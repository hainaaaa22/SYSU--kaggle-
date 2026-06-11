import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV  # 自动寻找最佳正则化系数的岭回归
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

print("--- 开启精简物理特征 + 自动调参岭回归模型 ---")

# 1. 读取电池级主表
train_df = pd.read_csv('battery-life-prediction-predicting-long-life-based-on-short\\train.csv')
test_df = pd.read_csv('battery-life-prediction-predicting-long-life-based-on-short\\test.csv')

# 2. ⭐ 核心进化：只提炼黄金物理特征，彻底抛弃40个特征的噪声干扰
# 包含了 Nature 论文的 Delta Q 统计量、内阻、放电容量以及最高温度
elite_features = [
    'log_dq_10_2_var',  # 斯坦福论文NO.1黄金特征：Delta Q 的方差
    'log_dq_10_2_min_abs',  # Delta Q 的绝对最小值
    'dq_10_2_mean',  # Delta Q 的均值
    'dq_10_2_std',  # Delta Q 的标准差
    'dq_10_2_skew',  # Delta Q 的偏度
    'dq_10_2_kurtosis',  # Delta Q 的峰度
    'qd_10_minus_2',  # 第10次与第2次放电容量差
    'ir_10',  # 第10次的内阻
    'ir_10_minus_2',  # 内阻的变化量
    'tmax_max_1_10'  # 前10次循环中的最高温度
]

print(f"已将特征从 41 个精简为 {len(elite_features)} 个黄金物理特征！")

X = train_df[elite_features].copy()
y_log = np.log10(train_df['cycle_life'])  # 保持对数变换，维持线性外推基础
X_test = test_df[elite_features].copy()

# 3. 标准化（物理特征量纲不同，必须标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 4. 严谨的 5 折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds_log = np.zeros(len(test_df))

print("\n正在进行 5 折交叉验证训练...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_log)):
    X_tr, y_tr = X_scaled[train_idx], y_log.iloc[train_idx]
    X_va, y_va = X_scaled[val_idx], y_log.iloc[val_idx]

    # 使用 RidgeCV 自带留一交叉验证，能在线性空间内找到最完美的特征权重拉伸
    model = RidgeCV(alphas=np.logspace(-4, 4, 100))
    model.fit(X_tr, y_tr)

    oof_preds[val_idx] = model.predict(X_va)
    test_preds_log += model.predict(X_test_scaled) / kf.n_splits

# 5. 计算诚实的本地验证集 RMSE
true_life = train_df['cycle_life']
pred_life = 10 ** oof_preds
cv_rmse = np.sqrt(mean_squared_error(true_life, pred_life))

print("\n" + "=" * 40)
print(f"🎉 瘦身后的本地交叉验证 RMSE 得分: {cv_rmse:.2f} 次循环")
print("=" * 40)

# 6. 关键检查：观察预测差距是否被成功拉开
final_test_preds = 10 ** test_preds_log
print(f"👉 测试集预测的最大寿命: {final_test_preds.max():.2f}")
print(f"👉 测试集预测的最小寿命: {final_test_preds.min():.2f}")

# 7. 生成全新版的线上提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'cycle_life': final_test_preds
})
submission.to_csv('submission_ridge_elite.csv', index=False)
print("\n✅ 'submission_ridge_elite.csv' 已成功生成！")