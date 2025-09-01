# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

# 开始修改代码，首先新增关键信息保存的功能

# Update 7.15 
# 1. 关键信息的保存也返回成功与否的flag，参考宏观检索
# 2. 监控<key_info_retrieve>的tag，从而把关键信息库中的key都送给模型
# 


import sys
import os
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
# 该函数的目的是去除输入中的左侧填充padding部分，返回一个有效的 token 序列。
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
                train_tp = kwargs.get('train_tp', None)
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')

        trust_remote_code = kwargs.get('trust_remote_code', False)
        load_format = 'dummy' if config.load_format.startswith('dummy') else config.load_format

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=int(os.getenv("RANK", "0")) // tensor_parallel_size,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            response = pad_2d_list_to_length(response, self.pad_token_id,
                                             max_length=self.config.response_length).to(idx.device)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if 'multi_modal_inputs' in non_tensor_batch.keys():
                    non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                                self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response,
                                                    eos_token=eos_token_id,
                                                    dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

import re
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from verl.utils.torch_functional import pad_sequence_to_length

class vLLMRolloutWithMicroTool(vLLMRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

        self.gen_str = "\n<|im_start|>assistant\n<think>"
        self.gen_ids = self.tokenizer.encode(self.gen_str)

        self.gen_str_answer = "\n<|im_start|>assistant\n<answer>"
        self.gen_ids_answer = self.tokenizer.encode(self.gen_str_answer)

        # 新增一些初始化参数
        # 添加answer阶段的生成前缀
        self.answer_gen_str = "\n<answer>\n"
        self.answer_gen_ids = self.tokenizer.encode(self.answer_gen_str)
        


    # 将json格式的工具调用字符串转换成python可执行的代码
    def format_tool_call(self, tool_call_str: str):
        """Convert JSON function call description to Python executable code string."""
        try:
            call_json = json.loads(tool_call_str)
            func_name = call_json['name']
            arguments = call_json.get('arguments', {})
            
            args_str = ', '.join(f"{k}={repr(v)}" for k, v in arguments.items())
            return f"{func_name}({args_str})"
        except Exception as e:
            return f"Parse tool call failed: {e}"

    def validate_tool_calls(self, output_str):
        if "macro_tool_call" in output_str:
            start_pattern = r'<macro_tool_call>'
            end_pattern = r'</macro_tool_call>'
        elif "micro_tool_call" in output_str:
            start_pattern = r'<micro_tool_call>'
            end_pattern = r'</micro_tool_call>'
        elif "key_info_save" in output_str:
            start_pattern = r'<key_info_save>'
            end_pattern = r'</key_info_save>'
        else:
            return False

        start_tags = re.findall(start_pattern, output_str)
        end_tags = re.findall(end_pattern, output_str)

        
        if len(start_tags) != len(end_tags):
            return False
            
        start_positions = [m.start() for m in re.finditer(start_pattern, output_str)]
        end_positions = [m.start() for m in re.finditer(end_pattern, output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True

    ################################################################################################
    # 下面这两个检测是否保存关键信息的函数写好了
    def validate_key_infos(self, output_str):
        start_tags = re.findall(r'<key_info_save>', output_str)
        end_tags = re.findall(r'</key_info_save>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
            
        start_positions = [m.start() for m in re.finditer(r'<key_info_save>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</key_info_save>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True

    def extract_key_infos(self, output_str):
        if not self.validate_key_infos(output_str):
            return []

        try:
            pattern = r'<key_info_save>((?:(?!</key_info_save>).)*)</key_info_save>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []

    ################################################################################################

    ################################################################################################
    def extract_key_info_retrieve(self, output_str):
        # tag_str = "<extract_key_info_retrieve>"
        tag_str = "<get_micro_func_schemas>"
        if tag_str in output_str:
            return True
        else:
            return False

    ################################################################################################

    ################################################################################################
    # 下面这两个检测是否微观调用的函数写好了
    def validate_micro_tool_calls(self, output_str):
        start_tags = re.findall(r'<micro_tool_call>', output_str)
        end_tags = re.findall(r'</micro_tool_call>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
            
        start_positions = [m.start() for m in re.finditer(r'<micro_tool_call>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</micro_tool_call>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True

    def extract_micro_tool_calls(self, output_str):
        if not self.validate_micro_tool_calls(output_str):
            return []

        try:
            pattern = r'<micro_tool_call>((?:(?!</micro_tool_call>).)*)</micro_tool_call>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []

    ################################################################################################

    ################################################################################################
    # 下面这两个检测是否宏观调用的函数写好了
    def validate_macro_tool_calls(self, output_str):
        start_tags = re.findall(r'<macro_tool_call>', output_str)
        end_tags = re.findall(r'</macro_tool_call>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
            
        start_positions = [m.start() for m in re.finditer(r'<macro_tool_call>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</macro_tool_call>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True

    def extract_macro_tool_calls(self, output_str):
        if not self.validate_macro_tool_calls(output_str):
            return []

        try:
            pattern = r'<macro_tool_call>((?:(?!</macro_tool_call>).)*)</macro_tool_call>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []

    ################################################################################################
    def extract_tool_calls(self, output_str):
        """
        返回: (tool_calls, call_type)
        call_type: 'macro', 'micro', 'key_info', 'none'
        """
        if not self.validate_tool_calls(output_str):
            return [], 'none'

        try:
            # 检测宏观工具调用
            macro_pattern = r'<macro_tool_call>((?:(?!</macro_tool_call>).)*)</macro_tool_call>'
            macro_matches = re.findall(macro_pattern, output_str, re.DOTALL)
            if macro_matches:
                return [match.group(1).strip() for match in macro_matches], 'macro'

            # 检测微观工具调用
            micro_pattern = r'<micro_tool_call>((?:(?!</micro_tool_call>).)*)</micro_tool_call>'
            micro_matches = re.findall(micro_pattern, output_str, re.DOTALL)
            if micro_matches:
                return [match.group(1).strip() for match in micro_matches], 'micro'
            
            # 检测关键信息保存
            key_info_pattern = r'<key_info_save>((?:(?!</key_info_save>).)*)</key_info_save>'
            key_info_matches = re.findall(key_info_pattern, output_str, re.DOTALL)
            if key_info_matches:
                return [match.group(1).strip() for match in key_info_matches], 'key_info'
            
            return [], 'none'
            
            # pattern = r'<tool_call>((?:(?!</tool_call>).)*)</tool_call>'
            # matches = re.finditer(pattern, output_str, re.DOTALL)
            
            # return [match.group(1).strip() for match in matches]


        except Exception as e:
            return [], 'none'
    

    def batch_execute(self, env_list: List[str], tool_calls_list: List[List[str]]):
        def exe_tool_call(env, call):
            url = f'{self.config.sandbox_url}/execute'

            call_str = self.format_tool_call(call)
            if call_str.startswith("Parse tool call failed"):
                return call_str
            
            try:
                data = {
                    'env': env,
                    'call': call_str
                }                
                response = requests.post(url, json=data, timeout=10)
                if response.status_code != 200:
                    return f"error: {response.status_code}"
                response = response.json()
                ret_str = ''
                if response['result']:
                    ret_str += f'result: \n{response["result"]}\n'
                if response['output']:
                    ret_str += f'output: \n{response["output"]}\n'
                if response['error']:
                    ret_str += f'error: \n{response["error"]}\n'
                return ret_str.strip()
            except requests.exceptions.Timeout:
                return "error: execution timed out"
            except Exception as e:
                return str(e)

        # flatten all tasks
        all_tasks = []
        task_indices = []
        for env_idx, (env, tool_calls) in enumerate(zip(env_list, tool_calls_list)):
            for call_idx, tool_call in enumerate(tool_calls):
                all_tasks.append((env, tool_call))
                task_indices.append((env_idx, call_idx))

        # parallel execute all tasks
        all_results = [None] * len(all_tasks)
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_index = {executor.submit(exe_tool_call, env, call): i 
                            for i, (env, call) in enumerate(all_tasks)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                all_results[index] = future.result()

        # reorganize results to original structure
        results_list = [[None for _ in range(len(tool_calls_list[i]))] for i, _ in enumerate(env_list)]
        for (env_idx, call_idx), result in zip(task_indices, all_results):
            results_list[env_idx][call_idx] = result

        return results_list

    # feng new
    def batch_execute_macro(self, env_list: List[str], tool_calls_list: List[List[str]]):
        """处理宏观工具调用"""
        return self.batch_execute(env_list, tool_calls_list)  # 复用现有逻辑
    
    # feng new
    def batch_execute_micro(self, env_list: List[str], tool_calls_list: List[List[str]]):
        """处理微观工具调用 - 从key_info_storage中检索"""
        # 感觉需要根据idx获取对应的key_info_storage，然后从中检索；

        results_list = []
        assert len(tool_calls_list) == len(env_list)

        for i in range(len(tool_calls_list)): # 相当于遍历每一个样本
            tool_calls = tool_calls_list[i]
        
            key_info_storage = env_list[i]
            # results = []
            results = ""

            for tool_call in tool_calls: # 这里还需要加一层遍历
                try:
                    micro_calls = json.loads(tool_call)
                    micro_querys = micro_calls.values()

                    for mq in micro_querys:
                        if mq in key_info_storage:
                            # results.append(str(key_info_storage[mq]))
                            results = results + mq +": "+ str(key_info_storage[mq])+";"
                            print("Successful micro retrieval processing! key-value 匹配一致！")
                            print(f"模型的请求是：{mq}")
                            print(f"关键信息库中的key是:{key_info_storage.keys()}")
                            print()
                        else:
                            results = results + f"Key '{mq}' not found in key information storage" + ";"
                            # results.append(f"Key '{mq}' not found in key information storage")
                            print("Successful micro retrieval processing! key-value 匹配错误！")
                            print(f"模型的请求是：{mq}")
                            print(f"关键信息库中的key是:{key_info_storage.keys()}")
                            print()

                except Exception as e:
                    results = results + f"Error processing micro retrieval: {tool_call}" + ";"
                    # results.append(f"Error processing micro retrieval: {e}")
                    print(f"Error processing micro retrieval: {e}")
                    print(f"模型的请求是：{tool_call}")
                    print()
                results_list.append(results)
        
        return results_list


    def batch_execute_key_info_retrieve(self, active_micro_env_list):
        keys_list = []
        for key_info_dict in active_micro_env_list:
            # Check if the current item is a valid dictionary
            if isinstance(key_info_dict, dict):
                # Retrieve the keys of the dictionary and append them to the keys_list
                keys_list.append(list(key_info_dict.keys()))
            else:
                # If it's not a valid dictionary, append an empty list or handle the error as needed
                keys_list.append([])  # You can append an empty list or handle the error differently
        

        return keys_list

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:        
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()


        # 初始化和输入准备
        ori_input_ids = prompts.batch['input_ids']  # (bs, prompt_length) # 这里还只是词汇表的索引，没转成embedding
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask'] # 这里只区分实际输入和padding
        position_ids = prompts.batch['position_ids'] # 位置编码的id
        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = ori_input_ids.size(0)

        idx_list = [] # 记录实际有用的input ids
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            # _pre_process_inputs， 该函数的目的是去除输入中的左侧填充部分，返回一个有效的 token 序列。
            idx_list.append(_pre_process_inputs(self.pad_token_id, ori_input_ids[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        with self.update_sampling_params(**kwargs):
            # prepare n copies for each input
            curr_inputs = [] # ：每个输入会被复制 n 次，以生成多个样本。
            for input_ids in idx_list: # 对每一个样本
                for _ in range(self.sampling_params.n): # 重复采样n次
                    curr_inputs.append(input_ids.copy())
            init_inputs = [ids.copy() for ids in curr_inputs] # 单独保留了一份原始的输入样本，以便后续进行操作


            # if there are envs, prepare n copies for each env
            env_list = None  # 环境就是指的外部知识
            if 'env' in prompts.non_tensor_batch:  # 如果有环境这个字段，复制n次
                env_list = [] 
                for env in prompts.non_tensor_batch['env']:
                    #print(env)
                    for _ in range(self.sampling_params.n):
                        env_list.append(env)


            # track the status of each input
            # 这是一个列表，其中每个元素都等于self.sampling_params.max_tokens，即每个输入样本的最大 token 数。这个列表用于跟踪每个输入样本当前的最大 token 数限制。
            curr_max_tokens = [self.sampling_params.max_tokens] * len(curr_inputs) 
            active_indices = list(range(len(curr_inputs))) # 当前所有输入的索引


            # 用于存储关键信息的字典，里面的每个元素是dict 
            key_info_storage_list = []
    
            # 初始化一下关键信息数据库，也进行复制
            for i in range(len(active_indices)):
                key_info_storage_list.append({})


            # print(f"len evn list: {len(env_list)}")
            # print()
            # print(f"len active_indices: {len(active_indices)}")
            # print(f"len key_info_storage_list: {len(key_info_storage_list)}")
            assert len(env_list) == len(key_info_storage_list), "error! 两个环境的大小不一样"

            # collect the result mask of each rollout, 1 for non-result, 0 for tool call result or pad
            result_mask_list = [[] for _ in range(len(curr_inputs))]

            # generate until all inputs are completed 该部分代码进行逐步生成，直到满足停止条件（如达到最大轮次或生成完所有序列）。
            # recall特有的循环生成，正常生成的话是不需要循环的
            for step in range(self.config.max_turns):
                if len(active_indices) == 0: # active_indices 是一个 索引列表，它存储着当前还在生成过程中的输入样本的索引。如果 active_indices 为空，说明所有输入都已处理完毕，因此可以结束生成
                    break

                # only process the active inputs
                active_inputs = [curr_inputs[i] for i in active_indices]
                active_max_tokens = [curr_max_tokens[i] for i in active_indices] # 用于控制生成时的最大长度。
                
                with self.update_sampling_params(
                    n=1, 
                    max_tokens=min(512, max(active_max_tokens)),
                    stop_token_ids=[151644],
                    top_p=0.99,
                ):  # 512 at most, and add <|im_start|> as stop for corner case
                    vllm_inputs = [{
                        'prompt_token_ids': raw_prompt_ids
                    } for raw_prompt_ids in active_inputs]
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    ) # 调用 生成模型 来生成对应的输出。这是模型生成的核心步骤。

                # collect all tool calls
                macro_tool_calls_list: List[List[str]] = [] # 意思就是表明这个list中的每个元素也还是list
                micro_tool_calls_list: List[List[str]] = [] 
                key_infos_list: List[List[str]] = [] 
                
                # call_indices: List[int] = [] # 这一行注释掉，改成下面的两行
                macro_call_indices = [] # 记录宏观检索激活的样本
                micro_call_indices = []

                # process each output
                new_active_indices = [] # 这个列表用来记录哪些输入样本仍然是“活动的”需要继续生成


                key_info_active_indices = [] # 记录那些样本的关键信息库需要更新
                key_info_retrieve_indices = [] # 记录哪些样本需要返回关键信息库的keys

                # feng: 我们从这里开始改，遍历每一个样本的时候
                for i, idx in enumerate(active_indices): # 遍历每一个输入样本
                    output_ids = outputs[i].outputs[0].token_ids #即模型生成的 token IDs
                    finish_reason = outputs[i].outputs[0].finish_reason # finish_reason 表示生成结束的原因，通常有几种可能性：'stop': 正常停止。'length': 达到最大生成长度。 其他自定义的结束条件。
                    stop_reason = outputs[i].outputs[0].stop_reason # 表示具体的停止原因，比如是否因为 EOS（End of Sentence）标记，或者某个特殊的停止条件（如 <|im_start|>）。
                    
                    if finish_reason == 'stop' and (stop_reason == None or stop_reason == self.tokenizer.pad_token_id):
                        # 如果生成是因为达到了停止标记（EOS token），并且 stop_reason 为空或是填充符号（pad_token_id），则表示正常生成结束。
                        curr_inputs[idx] += output_ids # 将生成的 token IDs 添加到当前输入的 curr_inputs。
                        result_mask_list[idx] += [1] * len(output_ids) # 表示这些 token 是有效生成的内容

                        output_str = self.tokenizer.decode(output_ids)

                        

                        # 这里需要改一下了，一种情况变三种情况了，看看是否需要微观、宏观工具调用以及关键信息存储
                        # 首先看是否需要宏观工具调用
                        macro_tool_calls: List[str] = self.extract_macro_tool_calls(output_str) # 检查是否有工具调用
                        # if len(macro_tool_calls) > 1:
                        #     print(f"模型生成了{len(macro_tool_calls)}组<macro_tool_calls>标签！")
                        if macro_tool_calls: # 如果提取到宏观工具调用 (macro_tool_calls)，则把它们加入 macro_tool_calls_list，并将该输入标记为需要继续生成。
                            macro_tool_calls_list.append(macro_tool_calls)
                            macro_call_indices.append(idx)
                        else:
                            pass # no tool calls

                        # 然后再看是否需要微观工具调用
                        micro_tool_calls: List[str] = self.extract_micro_tool_calls(output_str) # 检查是否有工具调用
                        # print(f"模型生成了{len(micro_tool_calls)}组<micro_tool_calls>标签！")
                        # if len(micro_tool_calls) > 1:
                        #     print(f"模型生成了{len(micro_tool_calls)}组<micro_tool_calls>标签！")
                            # print(micro_tool_calls)
                            # sys.exit(1)
                        if micro_tool_calls: # 如果提取到宏观工具调用 (macro_tool_calls)，则把它们加入 macro_tool_calls_list，并将该输入标记为需要继续生成。
                            micro_tool_calls_list.append(micro_tool_calls)
                            micro_call_indices.append(idx)
                        else:
                            pass # no tool calls

                        # 最后看看是否需要保存关键信息
                        key_infos: List[str] = self.extract_key_infos(output_str) 
                        # if len(key_infos) > 1:
                        #     print(f"模型生成了{len(key_infos)}组<key_info_save>标签！")
                            # print(f"模型的回复是：{output_str}")
                            # sys.exit(1)
                        if key_infos: # 如果提取到宏观工具调用 (macro_tool_calls)，则把它们加入 macro_tool_calls_list，并将该输入标记为需要继续生成。
                            key_infos_list.append(key_infos)
                            key_info_active_indices.append(idx)
                        else:
                            pass 
                        

                        # 新增，检查一下是否输出了<key_info_retrieve>，从而返回所有的keys。并准备开启answer阶段，
                        key_info_retrieve: bool = self.extract_key_info_retrieve(output_str) 

                        if key_info_retrieve:
                            key_info_retrieve_indices.append(idx)



                        # if macro_tool_calls and micro_tool_calls:
                        #     print(f"macro_tool_calls is {macro_tool_calls}")
                        #     print(f"micro_tool_calls is {micro_tool_calls}")
                        #     print("error")
                        #     print(f"output_str is {output_str}")
                        #     import sys
                        #     sys.exit(1)

                        # 只要涉及到上面的两种情况之一，该样本就需要继续生成没毛病吧
                        # 新增了如果有关键信息保存，也应该继续生成，因为后面还有answer阶段
                        if macro_tool_calls or micro_tool_calls or key_infos or key_info_retrieve:
                            
                            new_active_indices.append(idx)

                        # tool_calls: List[str] = self.extract_tool_calls(output_str) # 检查是否有工具调用
                        # if tool_calls: # 如果提取到工具调用 (tool_calls)，则把它们加入 tool_calls_list，并将该输入标记为需要继续生成。
                        #     tool_calls_list.append(tool_calls)
                        #     call_indices.append(idx)
                        #     new_active_indices.append(idx)
                        # else:
                        #     pass # no tool calls

                    elif finish_reason == 'length':
                        # output over max tokens
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    elif finish_reason == 'stop' and stop_reason == 151644: # 151644 is the id of <|im_start|>, is a illigal stop, we stop here
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    else:
                        raise ValueError(f"unknown stop reason. finish_reason: {finish_reason}, stop_reason: {stop_reason}")
                
                # print(f"len macro_call_indices is {len(macro_call_indices)}")
                # print(f"len micro_call_indices is {len(micro_call_indices)}")
                # print(f"len macro_tool_calls_list is {len(macro_tool_calls_list)}")
                # print(f"len micro_tool_calls_list is {len(micro_tool_calls_list)}")
                # print(f"len key_infos_list is {len(key_infos_list)}")
                # import sys
                # sys.exit(1)

                assert len(macro_call_indices) == len(macro_tool_calls_list)
                assert len(micro_call_indices) == len(micro_tool_calls_list)
                assert len(key_info_active_indices) == len(key_infos_list)
                


                # 然后就开始处理对应的工具调用
                # 首先处理save key info，处理关键信息的保存，他的处理需要跟tool calls独立开
                # 同时也将保存的结果返回给模型
                if key_infos_list: # 更新对应的key_info_storage_list
                    # Only tp_rank 0 executes the tools # 只有 rank 0 的设备执行工具调用。
                    if self.tp_rank == 0:
                        save_responses_list = []
                        for idx_info, key_infos in zip(key_info_active_indices, key_infos_list):
                            ret_str_list = []

                            for key_info in key_infos:     
                                ret_str = ''
                                if not key_info:  # 检查 key_info 是否为空
                                    ret_str += "Kye Info Save Error: Received empty key information."
                                    print("Kye Info Save Error: Received empty key information.")
                                    ret_str_list.append(ret_str)
                                    continue                       

                                try:
                                    # 解析并保存关键信息
                                    info_json = json.loads(key_info)
                                    key_info_storage_list[idx_info].update(info_json)
                                    ret_str += f"Key information saved successfully."
                                    print(f"Key information {info_json} saved successfully")
                                except json.JSONDecodeError as e:
                                    ret_str += f"Kye Info Save Error: JSON decoding failed. Exception: {e}"
                                    print(f"Kye Info Save Error: JSON decoding failed for key_info {key_info}. Exception: {e}")
                                except Exception as e:
                                    ret_str += f"Kye Info Save Error: {e}"
                                    print(f"Kye Info Save Error: {e}. key_infos: {key_info}")
                                ret_str_list.append(ret_str)
                            save_responses_list.append(ret_str_list)
                        # Prepare data for broadcasting
                        # 将工具调用、调用的索引、以及工具调用的响应结果放入 broadcast_data 中。这个数据将在分布式环境中广播给所有的设备。
                        broadcast_data = {
                            'key_info_save_list': key_infos_list,
                            'key_info_save_indices': key_info_active_indices,
                            'save_responses_list': save_responses_list
                        }
                    else:
                        broadcast_data = None

                    # 广播数据到所有设备
                    broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0)
                    
                    # All ranks process the broadcasted data 所有设备处理广播的数据
                    if broadcast_data is not None:
                        key_infos_list = broadcast_data['key_info_save_list']
                        key_info_active_indices = broadcast_data['key_info_save_indices']
                        save_responses_list = broadcast_data['save_responses_list']

                        # 这里对工具返回的结果进行处理
                        for idx, tool_calls, tool_responses in zip(key_info_active_indices, key_infos_list, save_responses_list):
                            tool_response_str = ''

                            for call, response in zip(tool_calls, tool_responses):
                                tool_response_str += f"<key_info_save_response>{call}\n{response}\n</key_info_save_response>\n"
                            # 这里需要仔细处理一下是否需要添加<|im_start|>  <|im_end|>
                            # 首先对于当前这个样本的idx，看看是否在macro_call_indices中
                            # 如果存在，说明还需要进行宏观检索并返回内容，那么只需要 提供<|im_start|> ，<|im_end|>在宏观response中收尾就行
                            # 如果不存在，那么就直接把<|im_start|>  <|im_end|>添加上就行
                            if idx in macro_call_indices:
                                tool_response_str = "\n<|im_start|>user\n" + tool_response_str

                                output_ids = self.tokenizer.encode(tool_response_str)
                                curr_inputs[idx] += output_ids
                                result_mask_list[idx] += [0] * len(output_ids)  # 这里对工具返回的结果做mask
                                # curr_inputs[idx] += self.gen_ids # self.gen_str = "\n<|im_start|>assistant\n<think>" 这部分也需要mask掉
                                # result_mask_list[idx] += [0] * len(self.gen_ids)
                            else:
                                tool_response_str = "\n<|im_start|>user\n" + tool_response_str + "<|im_end|>"
                            
                                output_ids = self.tokenizer.encode(tool_response_str)
                                curr_inputs[idx] += output_ids
                                result_mask_list[idx] += [0] * len(output_ids)  # 这里对工具返回的结果做mask
                                if idx not in key_info_retrieve_indices:
                                    curr_inputs[idx] += self.gen_ids # self.gen_str = "\n<|im_start|>assistant\n<think>" 这部分也需要mask掉
                                    result_mask_list[idx] += [0] * len(self.gen_ids)

    
 


                # 然后处理宏观工具调用
                # batch process tool calls
                if macro_tool_calls_list:
                    # Only tp_rank 0 executes the tools # 只有 rank 0 的设备执行工具调用。
                    if self.tp_rank == 0:
                        active_env_list = [env_list[i] for i in macro_call_indices]
                        macro_tool_responses_list = self.batch_execute_macro(active_env_list, macro_tool_calls_list)
                        
                        # Prepare data for broadcasting
                        # 将工具调用、调用的索引、以及工具调用的响应结果放入 broadcast_data 中。这个数据将在分布式环境中广播给所有的设备。
                        broadcast_data = {
                            'tool_calls_list': macro_tool_calls_list,
                            'call_indices': macro_call_indices,
                            'tool_responses_list': macro_tool_responses_list
                        }
                    else:
                        broadcast_data = None
                    
                    # 广播数据到所有设备
                    broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0)
                    
                    # All ranks process the broadcasted data 所有设备处理广播的数据
                    if broadcast_data is not None:
                        macro_tool_calls_list = broadcast_data['tool_calls_list']
                        call_indices = broadcast_data['call_indices']
                        macro_tool_responses_list = broadcast_data['tool_responses_list']

                        # 这里对工具返回的结果进行处理
                        for idx, tool_calls, tool_responses in zip(call_indices, macro_tool_calls_list, macro_tool_responses_list):
                            tool_response_str = ''
                            for call, response in zip(tool_calls, tool_responses):
                                tool_response_str += f"<macro_response>{call}\n{response}\n</macro_response>\n"
                            
                            # 这里也新增了一下
                            if idx in key_info_active_indices:
                                tool_response_str = tool_response_str + "<|im_end|>"
                            else:
                                tool_response_str = "\n<|im_start|>user\n" + tool_response_str + "<|im_end|>"
                            output_ids = self.tokenizer.encode(tool_response_str)

                            curr_inputs[idx] += output_ids
                            result_mask_list[idx] += [0] * len(output_ids)  # 这里对工具返回的结果做mask
                            if idx not in key_info_retrieve_indices:
                                curr_inputs[idx] += self.gen_ids # self.gen_str = "\n<|im_start|>assistant\n<think>" 这部分也需要mask掉
                                result_mask_list[idx] += [0] * len(self.gen_ids)
                
                # 再处理微观工具调用
                if micro_tool_calls_list: 
                
                    # Only tp_rank 0 executes the tools # 只有 rank 0 的设备执行工具调用。
                    if self.tp_rank == 0:
                        active_env_list = [key_info_storage_list[i] for i in micro_call_indices]
                        micro_tool_responses_list = self.batch_execute_micro(active_env_list, micro_tool_calls_list)
                        
                        # Prepare data for broadcasting
                        # 将工具调用、调用的索引、以及工具调用的响应结果放入 broadcast_data 中。这个数据将在分布式环境中广播给所有的设备。
                        broadcast_data = {
                            'tool_calls_list': micro_tool_calls_list,
                            'call_indices': micro_call_indices,
                            'tool_responses_list': micro_tool_responses_list
                        }
                    else:
                        broadcast_data = None
                    
                    # 广播数据到所有设备
                    broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0)
                    
                    # All ranks process the broadcasted data 所有设备处理广播的数据
                    if broadcast_data is not None:
                        micro_tool_calls_list = broadcast_data['tool_calls_list']
                        call_indices = broadcast_data['call_indices']
                        micro_tool_responses_list = broadcast_data['tool_responses_list']

                        # 这里对工具返回的结果进行处理
                        for idx, tool_calls, tool_responses in zip(call_indices, micro_tool_calls_list, micro_tool_responses_list):
                            # tool_response_str = ''
                            # for call, response in zip(tool_calls, tool_responses):
                            #     tool_response_str += f"{call}: {response}\n"

                            tool_response_str = f"\n<micro_response>{tool_responses}</micro_response>\n"
        
                            output_ids = self.tokenizer.encode(tool_response_str)

                            curr_inputs[idx] += output_ids
                            result_mask_list[idx] += [0] * len(output_ids)  # 这里对工具返回的结果做mask
                
                # 处理关键信息query返回
                if key_info_retrieve_indices:
                    
                    # Only tp_rank 0 executes the tools # 只有 rank 0 的设备执行工具调用。
                    if self.tp_rank == 0:
                        active_micro_env_list = [key_info_storage_list[i] for i in key_info_retrieve_indices]
                        key_info_retrieve_list = self.batch_execute_key_info_retrieve(active_micro_env_list) # 返回对应关键信息可以的keys
                        
                        # Prepare data for broadcasting
                        # 将工具调用、调用的索引、以及工具调用的响应结果放入 broadcast_data 中。这个数据将在分布式环境中广播给所有的设备。
                        broadcast_data = {
                            'key_info_retrieve_indices': key_info_retrieve_indices,
                            'key_info_retrieve_list': key_info_retrieve_list
                        }
                    else:
                        broadcast_data = None
                    
                    # 广播数据到所有设备
                    broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0)
                    
                    # All ranks process the broadcasted data 所有设备处理广播的数据
                    if broadcast_data is not None:
                        
                        key_info_retrieve_indices = broadcast_data['key_info_retrieve_indices']
                        key_info_retrieve_list = broadcast_data['key_info_retrieve_list']

                        # 这里对工具返回的结果进行处理
                        for idx, tool_responses in zip(key_info_retrieve_indices, key_info_retrieve_list):
                            
                            tool_response_str = f"Available functions for micro retrieval are: {str(tool_responses)}"


                            tool_response_str = "\n<|im_start|>user\n" + tool_response_str + "<|im_end|>"
                            output_ids = self.tokenizer.encode(tool_response_str)

                            curr_inputs[idx] += output_ids
                            result_mask_list[idx] += [0] * len(output_ids)  # 这里对工具返回的结果做mask

                            curr_inputs[idx] += self.gen_ids_answer # self.gen_str = "\n<|im_start|>assistant\n<answer>" 这部分也需要mask掉
                            result_mask_list[idx] += [0] * len(self.gen_ids_answer)


                
                # check if need to truncate, if yes, truncate, and remove from active; if no, update curr_max_tokens
                length_checked_active_indices = []
                for idx in active_indices:
                    assert len(curr_inputs[idx]) - len(init_inputs[idx]) == len(result_mask_list[idx]), f"curr_inputs: {len(curr_inputs[idx])}, init_inputs: {len(init_inputs[idx])}, result_mask_list: {len(result_mask_list[idx])}"
                    if len(curr_inputs[idx]) - len(init_inputs[idx]) >= self.config.response_length:
                        # 如果当前生成的 token 数量超过了最大响应长度（response_length），则执行截断操作。
                        curr_inputs[idx] = init_inputs[idx] \
                            + curr_inputs[idx][len(init_inputs[idx]):len(init_inputs[idx])+self.config.response_length]
                        result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                    else:
                        # 如果没有超出最大响应长度，更新 curr_max_tokens
                        curr_max_tokens[idx] = self.config.response_length - len(curr_inputs[idx]) + len(init_inputs[idx])
                        if idx in new_active_indices:
                            length_checked_active_indices.append(idx)
                # 最后，active_indices 被更新为所有符合长度要求的索引列表。只有那些生成过程中尚未完成的输入才会继续参与后续的生成。然后继续循环
                active_indices = length_checked_active_indices


            output_ids_list = []
            # collect the all rollouts
            for i, input_ids in enumerate(idx_list):
                for j in range(self.sampling_params.n):
                    # 遍历每一个样本
                    idx = i * self.sampling_params.n + j 
                    input_len = len(input_ids)
                    output_ids_list.append(curr_inputs[idx][input_len:])
                    # 这里就是提取每个样本response部分的ids

        # 这部分代码主要对生成的响应（output_ids）进行后处理，就是下面的三个数据，包括 padding 和 masking，都填充到统一的长度方便后续处理。
        response_attention_mask_list = []
        response_list = []
        result_mask_list_padded = []
        for output_ids, result_mask in zip(output_ids_list, result_mask_list):
            assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
            # to tensor 
            response = torch.tensor(output_ids, device=ori_input_ids.device)
            result_mask = torch.tensor(result_mask, device=ori_input_ids.device)
            # response attention mask, 1 for valid, 0 for invalid
            response_attention_mask = torch.ones_like(response, dtype=torch.int64)
            # 将 response_attention_mask 填充到 self.config.response_length 指定的长度，填充部分使用 0 来标记无效部分。
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
            response_attention_mask_list.append(response_attention_mask)
            # response, pad to response_length，再把response填充到response_length的长度
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            response_list.append(response)
            # result mask, 1 for non-result, 0 for result or pad 
            # 将 result_mask 填充到统一长度
            result_mask = pad_sequence_to_length(result_mask, self.config.response_length, 0)
            result_mask_list_padded.append(result_mask)
        response_attention_mask = torch.stack(response_attention_mask_list, dim=0)
        response = torch.stack(response_list, dim=0)
        result_mask = torch.stack(result_mask_list_padded, dim=0)


        # 这部分代码继续处理模型的输入，主要是 position_ids、attention_mask 和 loss_mask 等的拼接和准备，并构造了最终的输入批次（batch），用于后续的推理过程。
        if self.config.n > 1 and do_sample: # 输入会被 复制 n 次
            ori_input_ids = ori_input_ids.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        # seq 是将原始输入 ori_input_ids 和生成的响应 response 在最后一个维度（即序列长度维度）拼接起来，生成整个输入序列。
        seq = torch.cat([ori_input_ids, response], dim=-1)

        response_length = response.size(1) # 生成的token数量
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device) # 生成一个从 1 到 response_length 的整数序列。这个序列用于标记生成部分每个 token 的位置。
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1) # （batch_size, response_length）

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # position_ids[:, -1:]就等于4，假设delta_position_id原来是[1, 2, 3, 4]，那么response_position_ids就变成了[5, 6, 7, 8]
        # 然后再与input的position ids进行拼接，形成连冠的位置编码
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
                
        # concat attenion_mask for input and response
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # result mask: result part is 0, other part is 1
        # 这里修改最终的loss mask，同时取result_mask（工具调用的token）和response_attention_mask（padding的token）的并集
        # 例如result_mask = [1, 1, 0, 0, 1]，response_attention_mask = [1, 1, 1, 0, 1]，那么loss_mask = [1, 1, 0, 0, 1]
        loss_mask = result_mask * response_attention_mask
        # 普通情况下 loss_mask = response_attention_mask就行了
        
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict({
            'prompts': ori_input_ids,
            'responses': response,
            'input_ids': seq,  # here input_ids become the whole sentences
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids
        }, batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)