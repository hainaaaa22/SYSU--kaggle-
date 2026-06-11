import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import HuberRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. 读取数据
train_df = pd.read_csv('battery-life-prediction-predicting-long-life-based-on-short\\train.csv')
test_df = pd.read_csv('battery-life-prediction-predicting-long-life-based-on-short\\test.csv')

print(f"训练集形状:{train_df.shape}")
print(f"测试集形状:{test_df.shape}")

# 2. 自动筛选特征列
drop_cols = ['id', 'campaign', 'protocol', 'cycle_life']
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X = train_df[feature_cols].copy()
y_log = np.log10(train_df['cycle_life'])
X_test = test_df[feature_cols].copy()

# 3. 数据预处理：填补可能存在的极少数NaN，并进行标准化
imputer = SimpleImputer(strategy = 'median')
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 4. 5折交叉验证
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
oof_preds = np.zeros(len(train_df))
test_preds_log = np.zeros(len(test_df))

print("\n正在进行5折交叉验证训练")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_log)):
    X_tr, y_tr = X_scaled[train_idx], y_log.iloc[train_idx]
    X_va, y_va = X_scaled[val_idx], y_log.iloc[val_idx]

    #HuberRegressor能有效自动识别并削弱训练集中由于短寿电池带来的噪声影响
    model = HuberRegressor(max_iter = 15000)
    model.fit(X_tr, y_tr)

    oof_preds[val_idx] = model.predict(X_va)
    test_preds_log += model.predict(X_test_scaled) / kf.n_splits

# 5. 计算诚实、无泄漏的本地验证集RMSE
true_life = train_df['cycle_life']
pred_life = 10 ** oof_preds
cv_rmse = np.sqrt(mean_squared_error(true_life, pred_life))

print("\n" + "=" * 30)
print(f"本地交叉验证RMSE得分：{cv_rmse:.2f}")
print("=" * 30)

final_test_preds = 10 ** test_preds_log
print(f"测试集预测最大寿命:{final_test_preds.max():.2f}")
print(f"测试集预测最小寿命:{final_test_preds.min():.2f}")

# 6. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'cycle_life': final_test_preds
})
submission.to_csv('submission1.csv', index = False)
print("\nsubmission1.csv已成功生成")