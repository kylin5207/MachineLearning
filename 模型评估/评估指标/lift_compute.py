import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score

# 计算每个bin的precision
def calculate_precision(group):
    tp = ((group['pred'] == 1) & (group['true'] == 1)).sum()
    fp = ((group['pred'] == 1) & (group['true'] == 0)).sum()
    if tp + fp == 0:
        return 0  # 避免除以零
    precision = tp / (tp + fp)
    return precision

def calculate_lift(y_true, prob_pos, n_bins=10, threshold=0.5):
    """
    计算并返回每个分位数的Lift值

    参数:
    y_true : 真实标签数组
    prob_pos : 预测为正类的概率数组
    n_bins : 分位数的数量，默认为10

    返回:
    lift_values : 各分位数的Lift值
    """
    # 将真实标签和预测概率合并到DataFrame
    y_pred = np.where(prob_pos > threshold, 1, 0).ravel()
    df = pd.DataFrame({'true': y_true, 'prob': prob_pos, 'pred': y_pred})
    # 按照预测概率降序排序
    df = df.sort_values(by='prob', ascending=False)
    # 计算总体正例比例
    base_rate = df['true'].mean()

    # 计算每个分位数的累积响应率和Lift值
    df['bin'] = pd.qcut(df['prob'], q=n_bins, duplicates='drop', labels=False) + 1
    grouped = df.groupby('bin')

    nominator = grouped.apply(calculate_precision)
    lift_values = nominator / base_rate

    return lift_values

def total_lift(y_true, y_pred, threshold=0.5):
    """
    利用公式手动计算: (TP/(TP+FP))/(P/(P+N))=precision/positive_rate
    """
    # 计算分母：总体正例比例
    positive_rate = y_true.mean()
    print(f"denominator = {positive_rate}")

    class_results = np.where(y_pred > threshold, 1, 0).ravel()

    # 计算分子：precision
    precision = precision_score(y_true, class_results)
    print(f"nominator = {precision}")

    return precision / positive_rate

# 生成一些随机数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 训练一个简单的模型
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# 获取测试集的预测概率
prob_pos = model.predict_proba(X_test)[:, 1]

# 使用之前定义的函数计算Lift
print(f"======nbins = 10=====")
lift_values = calculate_lift(y_test, prob_pos, n_bins=10)
print(f"nbins=10, lift_values={lift_values}")

# 可视化Lift值
plt.figure(figsize=(10, 6))
lift_values.plot(kind='bar')
plt.title('Lift across deciles')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

print(f"====nbins = 1=====")
lift_values = calculate_lift(y_test, prob_pos, n_bins=1)
print(f"handy lift values with one bin = {lift_values}")
print(f"total = {total_lift(y_test, prob_pos)}")

