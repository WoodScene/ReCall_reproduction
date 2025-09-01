import re
import json
import requests
import time
from typing import List
from functools import wraps

def retry(max: int=10, sleep: int=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"[retry] try {i} times")
                    if i == max - 1:
                        raise Exception("Retry {} failed after {} times".format(func.__name__, max))
                    elif sleep:
                        time.sleep(sleep)
        return wrapper
    return decorator

class ReCall_ours_v4():
    system_prompt = """You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think></think> tags, while the final answer is enclosed within <answer></answer> tags.

During the reasoning process, you have access to a set of tools you can use to assist with the user query, referred to as macro retrievals. These macro retrievals are enclosed within <macro_tool_call></macro_tool_call> tags. In your response, you must first reason through the problem and then perform macro function calls to gather information or perform actions if needed. You may conduct multiple rounds of function calls, and in each round, you can call one or more functions. 

Here are the available functions in JSON Schema format for macro retrieval:  
```json\n{func_schemas}\n```

The results of the macro function calls will be given back to you after execution, and you can continue to call functions until you get the final answer for the user's question.

Additionally, since the user's query may involve multiple questions, during the reasoning process, whenever you obtain the answer to one of the questions, you should store the current round's answer as a key-value pair in a key information dictionary using the <key_info_save></key_info_save> tag. This allows you to retrieve the answer directly during the answering phase.

For example: <think> Based on the response from the macro function call, I get the flight ID and hotel name.\n  <key_info_save>{{"flight_ID": "FL456", "hotel_name": "Cozy Inn"}} </key_info_save> </think>

The result of saving the key information (whether successful or failed) will be provided after execution. If saving fails, you can adjust the format based on the error information and retry saving.
(Note: You can perform both macro function calls and key information saving simultaneously, but they must be used during the thinking process.)

Once you have gathered enough information and no longer need to make macro function calls or store key information, you can proceed to the answering phase. However, before outputting the <answer> tag, you can call a tool to get all the available function calls for micro retrieval by using the following format: "<think> Based on the response from the function call, I have all the necessary information to answer the user's question. </think> <get_micro_func_schemas>"

The available functions for micro retrieval will be provided to you after execution, which includes all the keys in the key information dictionary. Therefore, during the answering phase, you can only select from the returned functions and use micro retrieval to obtain the answer.

For macro function call, return a JSON object with the function name and arguments within <macro_tool_call></macro_tool_call> tags:
<macro_tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</macro_tool_call>

During the answering phase, the response should be enclosed within <answer></answer> tags and formatted using LaTeX notation. Note that there should only be one set of <answer></answer> tags.

Before answering, you must first perform micro retrieval to fetch the relevant key information. You should then base your answer on the results of the micro retrieval, rather than answering independently.

Micro retrieval should be enclosed within <micro_tool_call></micro_tool_call> tags. You can query multiple items at once or issue requests in batches. The results of the micro function calls will be provided after execution, and you can continue making additional calls until all of the user’s questions are answered. If the micro retrieval does not return the correct content (e.g., due to formatting errors), simply indicate that the results are incorrect, without generating an answer independently.

Additionally, ensure that the micro retrieval is as closely aligned with the corresponding question as possible, as shown in the example below. This approach will result in a more accurate and efficient response format:

Better format (answer1):
<answer> 
    <micro_tool_call>{{"query1": "flight_ID", "query2": "hotel_name"}}</micro_tool_call>
    <micro_response>flight_ID: FL456; hotel_name: Cozy Inn;</micro_response> 
    The cheapest flight ID is FL456, and the hotel name is Cozy Inn. 
    <micro_tool_call>{{"query1": "total_cost"}}</micro_tool_call>
    <micro_response>Error processing micro retrieval...</micro_response> 
    And the total cost of the trip could not be retrieved from the key information repository.
</answer>

Less optimal format (answer2):
<answer> 
    <micro_tool_call>{{"query1": "flight_ID", "query2": "hotel_name", "query3": "total_cost"}}</micro_tool_call>
    <micro_response>flight_ID: FL456; hotel_name: Cozy Inn; Error processing micro retrieval for total_cost</micro_response> 
    The cheapest flight ID is FL456, and the hotel name is Cozy Inn. 
    The total cost of the trip could not be retrieved from the key information repository.
</answer>


Note: Always remember that macro retrieval and key information saving must occur during the think phase, while micro retrieval should only take place during the answer phase.
"""



    def __init__(self, model_url, executor_url):
        self.model_url = model_url
        self.executor_url = executor_url
        
    def init_prompt(self, func_schemas, question):
        system_prompt = f"<|im_start|>system\n{self.system_prompt.format(func_schemas=func_schemas)}<|im_end|>"
        user_prompt = f"<|im_start|>user\n{question}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return system_prompt + "\n" + user_prompt + "\n" + assistant_prefix

    # 把助手的回答与当前的prompt进行拼接
    def cat_assistant_response(self, curr_prompt, assistant_response):
        return curr_prompt + assistant_response + "<|im_end|>"
    
    def cat_assistant_response_noend(self, curr_prompt, assistant_response):
        return curr_prompt + assistant_response
    
    # 将工具调用的结果与当前的prompt进行拼接，工具调用的结果插入到user的tag中
    def cat_macro_tool_results(self, curr_prompt, tool_calls, results, key_infos):
        tool_response_str = ""
        for tool_call, result in zip(tool_calls, results):
            tool_response_str += f"<tool_response>{tool_call}\n{result}\n</tool_response>\n"

        if len(key_infos) > 0:
            tool_response_str = tool_response_str + "<|im_end|>"
        else:
            tool_response_str = "\n<|im_start|>user\n" + tool_response_str + "<|im_end|>"

        # tool_response_str = f"<|im_start|>user\n{tool_response_str}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return curr_prompt + "\n" + tool_response_str + "\n" + assistant_prefix


    # 将工具调用的结果与当前的prompt进行拼接，工具调用的结果插入到user的tag中
    def cat_micro_tool_results(self, curr_prompt, tool_calls, results):
        tool_response_str = ""
        
        tool_response_str += f"\n<micro_response>{results}</micro_response>\n"
        
        
        return curr_prompt + tool_response_str 

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
    

    # 执行工具的调用，并返回结果
    def execute_macro_tool_calls(self, env: str, tool_calls: List[str]) -> List[str]:
        def exe_tool_call(env, call):
            url = self.executor_url + '/execute'

            call_str = self.format_tool_call(call)
            if call_str.startswith("error: parse tool call failed"):
                return call_str

            try:
                data = {
                    'env': env,
                    'call': call_str
                }
                response = requests.post(url, json=data, timeout=3)
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
        
        results = []
        for tool_call in tool_calls:
            result = exe_tool_call(env, tool_call)
            results.append(result)
        return results
    
    # 执行工具的调用，并返回结果
    def execute_micro_tool_calls(self, env: dict, tool_calls: List[str]) -> str:
        results = ""

        for tool_call in tool_calls: # 这里还需要加一层遍历
            try:
                micro_calls = json.loads(tool_call)
                micro_querys = micro_calls.values()

                for mq in micro_querys:
                    if mq in env:
                        # results.append(str(key_info_storage[mq]))
                        results = results + mq +": "+ str(env[mq])+";"
                        #print("Successful micro retrieval processing! key-value 匹配一致！")
                    else:
                        results = results + f"Key '{mq}' not found in key information storage" + ";"
                        # results.append(f"Key '{mq}' not found in key information storage")
                        # print("Successful micro retrieval processing! key-value 匹配错误！")
            except Exception as e:
                results = results + f"Error processing micro retrieval: {tool_call}" + ";"
                # results.append(f"Error processing micro retrieval: {e}")
                #print(f"Error processing micro retrieval: {e}")
        return results

    # 验证工具调用是否有效
    def validate_tool_calls(self, output_str):
        # 先分别查找这两个tag出现的次数
        start_tags = re.findall(r'<macro_tool_call>', output_str)
        end_tags = re.findall(r'</macro_tool_call>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
        
        # 在检查这两个标签的起始位置
        start_positions = [m.start() for m in re.finditer(r'<macro_tool_call>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</macro_tool_call>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True


    # 从模型的输出中提取工具调用
    # 就是看看模型在本次的输出内容中，想要调用几个工具，返回调用工具的个数
    def extract_macro_tool_calls(self, output_str):

        # 先验证一下工具的调用是否有效，例如出否出现了格式上的问题
        if not self.validate_tool_calls(output_str):
            return []

        try:
            # 提取<tool_call>...</tool_call> 区块中的所有内容
            pattern = r'<macro_tool_call>((?:(?!</macro_tool_call>).)*)</macro_tool_call>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []



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
    
    def extract_key_info_retrieve(self, output_str):
        # tag_str = "<extract_key_info_retrieve>"
        tag_str = "<get_micro_func_schemas>"
        if tag_str in output_str:
            return True
        else:
            return False



        
    @retry(max=5, sleep=1)
    def run(self, env, func_schemas, question):

        key_info_dic = {}
        
        curr_prompt = self.init_prompt(func_schemas, question)
        for _ in range(50):

            response = requests.post(
                f'{self.model_url}/generate', 
                json={
                    "text": curr_prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "repetition_penalty": 2,
                        "max_new_tokens": 512,
                        # "do_sample": True,

                    }
                }
            ).json()

            #print(response['text'])

            # 查看模型输出的结果中需要调用的工具个数
            macro_tool_calls: List[str] = self.extract_macro_tool_calls(response['text'])

            micro_tool_calls: List[str] = self.extract_micro_tool_calls(response['text'])
            
            # 关键信息保存
            key_infos = self.extract_key_infos(response['text'])

            # 新增，检查一下是否输出了<key_info_retrieve>，从而返回所有的keys。并准备开启answer阶段，
            key_info_retrieve_flag: bool = self.extract_key_info_retrieve(response['text'])

            # 做一个拼接
            # 这里得改一下
            if len(micro_tool_calls) > 0:
                curr_prompt = self.cat_assistant_response_noend(curr_prompt, response['text'])
            else:
                curr_prompt = self.cat_assistant_response(curr_prompt, response['text'])

            # print("current prompt is:")
            # print(curr_prompt)
            # print()

            if len(macro_tool_calls) == 0 and len(micro_tool_calls) == 0 and len(key_infos) == 0: # 不需要调用工具时，退出循环返回结果。
                break

            if len(key_infos) > 0:
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
                        key_info_dic.update(info_json)
                        ret_str += f"Key information saved successfully."
                        print(f"Key information {info_json} saved successfully")
                    except json.JSONDecodeError as e:
                        ret_str += f"Kye Info Save Error: JSON decoding failed. Exception: {e}"
                        print(f"Kye Info Save Error: JSON decoding failed for key_info {key_info}. Exception: {e}")
                    except Exception as e:
                        ret_str += f"Kye Info Save Error: {e}"
                        print(f"Kye Info Save Error: {e}. key_infos: {key_info}")
                    ret_str_list.append(ret_str)

                # 这里对工具返回的结果进行处理
                tool_response_str = ''

                for call, response in zip(key_infos, ret_str_list):
                    tool_response_str += f"<key_info_save_response>{call}\n{response}\n</key_info_save_response>\n"
                # 这里需要仔细处理一下是否需要添加<|im_start|>  <|im_end|>
                # 首先对于当前这个样本的idx，看看是否在macro_call_indices中
                # 如果存在，说明还需要进行宏观检索并返回内容，那么只需要 提供<|im_start|> ，<|im_end|>在宏观response中收尾就行
                # 如果不存在，那么就直接把<|im_start|>  <|im_end|>添加上就行
                if len(macro_tool_calls) > 0:
                    tool_response_str = "\n<|im_start|>user\n" + tool_response_str      
                    curr_prompt = curr_prompt + "\n" + tool_response_str + "\n"              
                else:
                    tool_response_str = "\n<|im_start|>user\n" + tool_response_str + "<|im_end|>"
                
                    curr_prompt = curr_prompt + "\n" + tool_response_str + "\n"
                        # 
                    if not key_info_retrieve_flag:
                        assistant_prefix = f"<|im_start|>assistant\n<think>"
                        curr_prompt = curr_prompt + assistant_prefix


            
            if len(macro_tool_calls) > 0:
                # 执行工具的调用并返回结果
                results: List[str] = self.execute_macro_tool_calls(env, macro_tool_calls)

                # 把工具调用的结果放入usr tag中并对prompt进行拼接
                curr_prompt = self.cat_macro_tool_results(curr_prompt, macro_tool_calls, results, key_infos)

            if len(micro_tool_calls) > 0:
                results: str = self.execute_micro_tool_calls(key_info_dic, micro_tool_calls)
                # 把工具调用的结果放入usr tag中并对prompt进行拼接
                curr_prompt = self.cat_micro_tool_results(curr_prompt, micro_tool_calls, results)


            if key_info_retrieve_flag:
                
                tool_responses = list(key_info_dic.keys())
                tool_response_str = f"Available functions for micro retrieval are: {str(tool_responses)}"
                tool_response_str = "\n<|im_start|>user\n" + tool_response_str + "<|im_end|>"

                curr_prompt = curr_prompt + "\n" + tool_response_str + "\n"

                assistant_prefix = f"<|im_start|>assistant\n<answer>"
                curr_prompt = curr_prompt + assistant_prefix

        return curr_prompt