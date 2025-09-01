import pandas as pd
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
# 打印 sys.path 检查是否正确添加了路径
# print(sys.path)
from re_call import ReCall

model_url = "http://0.0.0.0:30001"
sandbox_url = "http://0.0.0.0:2501"
data_path = "/group/40059/yuujiefeng/aaai2026/ReCall/data/ReCall-data/syntool_re_call/test.parquet"

# load some data
test_lines = []
test_data = pd.read_parquet(data_path)
for row in test_data.iterrows():
    curr_line = {}
    curr_line['question'] = row[1]['question']
    curr_line['answer'] = row[1]['reward_model']['ground_truth']
    curr_line['env'] = row[1]['extra_info']['env']
    curr_line['func_schemas'] = row[1]['extra_info']['func_schemas']
    test_lines.append(curr_line)

# initialize the re_call model
re_call = ReCall(model_url, sandbox_url)

# run the re_call model
test_data = test_lines[3]
#print(f"test_data['env']: {test_data['env']}")
#print("------------------")
# print(f"test_data['func_schemas']: {test_data['func_schemas']}")
#sys.exit(1)
response = re_call.run(test_data['env'], test_data['func_schemas'], test_data['question'])
print(response)
print("------------------")
print(test_data['answer'])
print("------------------")

#print(test_data)