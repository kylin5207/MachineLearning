import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib
matplotlib.use('TkAgg')


# 加载乳腺癌数据集
data = load_breast_cancer()
X, y = data.data, data.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用逻辑回归作为分类器（您可以根据需要选择其他分类器）
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)

# 预测测试集中的概率
y_probs = classifier.predict_proba(X_test)[:, 1]

# 计算ROC曲线的数据
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

auc_score = roc_auc_score(y_test, y_probs)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve - Breast Cancer Classification')
plt.legend()
plt.grid(True)
plt.show()
