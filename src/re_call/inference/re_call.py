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

class ReCall():
    system_prompt = """In this environment you have access to a set of tools you can use to assist with the user query. \
You may perform multiple rounds of function calls. \
In each round, you can call one or more functions. \

Here are available functions in JSONSchema format: \n```json\n{func_schemas}\n```

In your response, you need to first think about the reasoning process in the mind and then conduct function calling to get the information or perform the actions if needed. \
The reasoning process and function calling are enclosed within <think> </think> and <tool_call> </tool_call> tags. \
The results of the function calls will be given back to you after execution, \
and you can continue to call functions until you get the final answer for the user's question. \
Finally, if you have got the answer, enclose it within \\boxed{{}} with latex format and do not continue to call functions, \
i.e., <think> Based on the response from the function call, I get the weather information. </think> The weather in Beijing on 2025-04-01 is \\[ \\boxed{{20C}} \\].

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

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
    
    # 将工具调用的结果与当前的prompt进行拼接，工具调用的结果插入到user的tag中
    def cat_tool_results(self, curr_prompt, tool_calls, results):
        tool_response_str = ""
        for tool_call, result in zip(tool_calls, results):
            tool_response_str += f"<tool_response>{tool_call}\n{result}\n</tool_response>\n"
        tool_response_str = f"<|im_start|>user\n{tool_response_str}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return curr_prompt + "\n" + tool_response_str + "\n" + assistant_prefix

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
    def execute_tool_calls(self, env: str, tool_calls: List[str]) -> List[str]:
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
    
    # 验证工具调用是否有效
    def validate_tool_calls(self, output_str):
        # 先分别查找这两个tag出现的次数
        start_tags = re.findall(r'<tool_call>', output_str)
        end_tags = re.findall(r'</tool_call>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
        
        # 在检查这两个标签的起始位置
        start_positions = [m.start() for m in re.finditer(r'<tool_call>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</tool_call>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True


    # 从模型的输出中提取工具调用
    # 就是看看模型在本次的输出内容中，想要调用几个工具，返回调用工具的个数
    def extract_tool_calls(self, output_str):

        # 先验证一下工具的调用是否有效，例如出否出现了格式上的问题
        if not self.validate_tool_calls(output_str):
            return []

        try:
            # 提取<tool_call>...</tool_call> 区块中的所有内容
            pattern = r'<tool_call>((?:(?!</tool_call>).)*)</tool_call>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []


        
    @retry(max=5, sleep=1)
    def run(self, env, func_schemas, question):
        curr_prompt = self.init_prompt(func_schemas, question)
        for _ in range(50):
            response = requests.post(
                f'{self.model_url}/generate', 
                json={
                    "text": curr_prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 512
                    }
                }
            ).json()

            # 做一个拼接
            curr_prompt = self.cat_assistant_response(curr_prompt, response['text'])

            # 查看模型输出的结果中需要调用的工具个数
            tool_calls: List[str] = self.extract_tool_calls(response['text'])
            if len(tool_calls) == 0: # 不需要调用工具时，退出循环返回结果。
                break
            # 执行工具的调用并返回结果
            results: List[str] = self.execute_tool_calls(env, tool_calls)

            # 把工具调用的结果放入usr tag中并对prompt进行拼接
            curr_prompt = self.cat_tool_results(curr_prompt, tool_calls, results)

        return curr_prompt