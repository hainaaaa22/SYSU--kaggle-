

# ========================


import pandas as pd
import pprint

print("开始探索循环数据结构")

# 1. 读取train_cycles.csv的前5行
try:
    cycles_head = pd.read_csv('battery-life-prediction-predicting-long-life-based-on-short\\train_cycles.csv', nrows = 5)

    print("预测循环数据集的列名(Columns):")
    print(cycles_head.columns.tolist())
    print("\n前2行数据详细预览：")
    pprint.pprint(cycles_head.head(2).to_dict(orient = 'records'))

except FileNotFoundError:
    print("未找到train_cycles.csv,请检查路径")