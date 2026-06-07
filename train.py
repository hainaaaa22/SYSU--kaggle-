import pandas as pd
import numpy as np



# ============================== 一、 数据加载与初步探索（EDA)==============================


# ========== 0. 引入工具 ==========
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# ========== 1. 加载数据 ==========
print("正在加载数据....")
try:
    train_df = pd.read_csv('battery-life-prediction-predicting-long-life-based-on-short\\train.csv')
    test_df = pd.read_csv('battery-life-prediction-predicting-long-life-based-on-short\\test.csv')
    sample_sub = pd.read_csv('battery-life-prediction-predicting-long-life-based-on-short\\sample_submission.csv')
    print("数据加载成功！\n")
except FileNotFoundError as e:
    print(f"找不到文件，请检查路径:{e}")

# ========== 2. 查看数据基本维度 ==========
print(f"训练集形状（行数，列数）:{train_df.shape}")
print(f"测试集形状（行数，列数）:{test_df.shape}")
print("-" * 30)

# ========== 3. 预览训练集前5行数据 ==========
print("训练集前5行预览:")
print(train_df.head())
print("-" * 30)

# ========== 4. 检查训练集中是否存在缺失值(NaN)
missing_values = train_df.isnull().sum()
missing_cols = missing_values[missing_values > 0]

if len(missing_cols) > 0:
    print("训练集中存在缺失值的列及数量:")
    print(missing_cols)
else:
    print("训练集中没有缺失值")


# ================================= 二、 数据预处理与构建Baseline模型 =================================


print("\n--- 开始第二阶段：数据预处理与Baseline模型")

# 1. 复制数据，养成不污染原始数据的好习惯
train_data = train_df.copy()
test_data = test_df.copy()

# 2. 剔除无用列和全缺失列
cols_to_drop = ['policy_c3', 'policy_c4']
train_data = train_data.drop(columns = cols_to_drop)
test_data = test_data.drop(columns = cols_to_drop)

# 处理id列
train_data = train_data.drop(columns = ['id'])
test_id = test_data['id']
test_data = test_data.drop(columns = ['id'])

# 3. 填补少量缺失值
# 取训练集中policy_c2的中位数，同时填补给训练集和测试集
median_c2 = train_data['policy_c2'].median()
train_data['policy_c2'] = train_data['policy_c2'].fillna(median_c2)
test_data['policy_c2'] = test_data['policy_c2'].fillna(median_c2)

# 4. 提取数值型特征
numeric_cols = train_data.select_dtypes(include = [np.number]).columns.tolist()
numeric_cols.remove('cycle_life')              #移除目标变量，不能将其作为特征

print(f"提取出{len(numeric_cols)}个数值特征用于训练")

X = train_data[numeric_cols]
y = train_data['cycle_life']
X_test = test_data[numeric_cols]

# 5. 划分本地验证集 (80%作为训练，20%作为考试)
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.2, random_state = 42)

# 6. 训练Baseline模型（使用随机森林模型）
print("正在训练随机森林模型....")
rf_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf_model.fit(X_train, y_train)

# 7. 在本地验证集上进行预测并计算得分
val_preds = rf_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"本地验证集RMSE得分：{rmse:.2f}")


# ============================== 三、 生成测试集预测与提交文件 ==============================


print("\n--- 开始第三阶段：生成测试集预测与提交文件 ---")

# 1. 使用训练好的模型对测试集(X_test)进行预测
print("正在生成测试集预测结果")
test_preds = rf_model.predict(X_test)

# 2. 将预测结果与之前保存的test_id组装成Kaggle要求的格式
submission_df = pd.DataFrame({
    'id': test_id,
    'cycle_life': test_preds
})

# 3. 预览一下准备提交的数据
print("\n提交数据前5行预览：")
print(submission_df.head())

# 4. 将结果保存为CSV文件
submission_name = 'submission.csv'
submission_df.to_csv(submission_name, index = False)

print(f"{submission_name}文件已生成")