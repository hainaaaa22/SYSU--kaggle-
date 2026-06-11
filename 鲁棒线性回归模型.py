import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import HuberRegressor # 鲁棒线性回归
from sklearn.metrics import mean_squared_error

print("========== 开始第三阶段：电池级聚合与对数线性外推 ==========")

# 1. 加载上一阶段融合好的完整循环数据集
final_train = pd.read_csv('final_train.csv')
final_test = pd.read_csv('final_test.csv')

# 2. 按照电池id进行分组聚合，让一辆电池只占一行（同时提取“方差”统计量）
#此举是由于鲁棒线性模型只能接收一行数据，所以我们要将这10-100次循环的数据浓缩到某一个量，在这里是方差
print("正在将循环级数据聚合为电池级数据...")

agg_dict = {
    'cycle_life': 'first',
    'delta_q': 'first',
    'delta_ir': 'first',
    'q_discharge_10': 'first',
    'ir_10': 'first',
    'tmax_max': 'first',
    'chargetime_mean': 'first',
    'q_discharge': 'var',            #放电容量在100次循环中的方差
    'ir': 'var'                      #内阻在100次循环中的方差
}

#训练集聚合
cell_train = final_train.groupby('id').agg(agg_dict).reset_index()

#测试集聚合（测试集没有cycle_life列，需移除）
agg_dict_test = agg_dict.copy()
del agg_dict_test['cycle_life']
cell_test = final_test.groupby('id').agg(agg_dict_test).reset_index()

print(f"聚合后训练集形状：{cell_train.shape}")
print(f"聚合后测试集形状：{cell_test.shape}")
print("=" * 30)