import pandas as pd
import numpy as np

def extract_battery_features(cycles_path):
    #从循环细节数据集中为每个电池提取宏观电化学特征
    print(f"正在从{cycles_path}中提取物理特征")
    df = pd.read_csv(cycles_path)

    features_list = []

    # 按照电池id进行分组处理
    for cell_id, group in df.groupby('id'):
        #确保循环按1到100的顺序排列
        group = group.sort_values('cycle_index')

        #提取第10和第100次循环数据
        row_10 = group[group['cycle_index'] == 10]
        row_100 = group[group['cycle_index'] == 100]

        #防御性代码：如果刚好缺失了第10或第100次，则用最早和最晚的循环代替
        if row_10.empty: row_10 = group.head(1)
        if row_100.empty: row_100 = group.tail(1)

        q_10 = row_10['q_discharge'].values[0]
        q_100 = row_100['q_discharge'].values[0]
        ir_10 = row_10['ir'].values[0]
        ir_100 = row_100['ir'].values[0]

        #截取第10到第100次循环之间的片段，用来计算期间的统计量
        group_90 = group[(group['cycle_index'] >= 10) & (group['cycle_index'] <= 100)]

        #组装物理特征
        feat = {
            'id': cell_id,
            # 1. 放电容量特征(Q)
            'q_discharge_10': q_10,
            'q_discharge_100': q_100,
            'delta_q': q_100 - q_10,

            # 2. 内阻特征
            'ir_10': ir_10,
            'ir_100': ir_100,
            'delta_ir': ir_100 - ir_10,

            # 3. 温度与充电应力特征
            'tmax_max': group_90['tmax'].max(),              # 期间承受的最高温度
            'tavg_mean': group_90['tavg'].mean(),            # 期间的平均温度
            'chargetime_mean': group_90['chargetime'].mean() # 期间平均充电时间
        }
        features_list.append(feat)

    return pd.DataFrame(features_list)

# ========== 主程序开始 ==========

# 0. 取文件路径
path_to_train_cycles = "battery-life-prediction-predicting-long-life-based-on-short\\train_cycles.csv"
path_to_test_cycles = "battery-life-prediction-predicting-long-life-based-on-short\\test_cycles.csv"

# 1. 提取训练集和测试集的循环特征
train_features = extract_battery_features(path_to_train_cycles)
test_features = extract_battery_features(path_to_train_cycles)

# 2. 读取原始的主表数据
train_df = pd.read_csv(path_to_train_cycles)
test_df = pd.read_csv(path_to_test_cycles)

# 3. 通过id列将新提取的物理特征合并回主表中
final_train = pd.merge(train_df, train_features, on = 'id', how = 'left')
final_test = pd.merge(test_df, test_features, on = 'id', how = 'left')

print("\n---------- 特征工程融合完成 --------")
print(f"最终训练集形状：{final_train.shape}")
print(f"最终测试集形状：{final_test.shape}")

# 4. 将制作好的高质量数据集保存到本地
final_train.to_csv('final_train.csv', index = False)
final_test.to_csv('final_test.csv', index = False)
print("'final_train.csv' 和 'final_test.csv' 已保存在本地，准备用于下一步建模")