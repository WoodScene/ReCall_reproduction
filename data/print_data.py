import pandas as pd
import numpy as np
import json

# 读取 parquet 文件
df = pd.read_parquet("./ReCall-data/musique_re_call/train.parquet") # 2w
# df = pd.read_parquet("./ReCall-data/syntool_re_call/train.parquet") # 1w
# 打印前几行数据看样子
print(df.head())  # 默认显示前5行
print(len(df))
#sys.exit(1)
# 如果想看所有列名
print(df.columns)

# 如果只想看一小部分列的内容
# print(df[["column1", "column2"]].head())

# 获取第一行数据并转换为字典
first_row_dict = df.iloc[0].to_dict()
print(first_row_dict)
# 递归处理不可 JSON 序列化的数据类型
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    else:
        return obj

serializable_data = make_json_serializable(first_row_dict)

# 保存为 JSON 文件
with open("first_row.json", "w", encoding="utf-8") as f:
    json.dump(serializable_data, f, ensure_ascii=False, indent=2)

print("First row saved to first_row.json")


