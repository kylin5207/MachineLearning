from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def plot_roc(tpr, fpr):
    plt.plot(fpr, tpr)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.grid()
    plt.title("ROC")
    plt.show()

# 创建一个简单的二分类数据集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 创建随机森林分类器并在训练数据上进行拟合
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)

# 对测试集进行概率预测，即获取模型预测正类的概率
y_scores = clf.predict_proba(X_test)[:, 1]  # 分数是正类的概率

# 计算ROC曲线和每个类的ROC面积
# 计算方式1，通过roc_curve得到tpr和fpr，然后进一步利用auc函数计算
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)
print(f"sklearn roc_curve and auc = {roc_auc}")

# 计算方式2，直接利用roc_auc_score计算
auc_val = roc_auc_score(y_test, y_scores)
print(f"sklearn roc_auc_score = {auc_val}")

# plot roc
plot_roc(tpr, fpr)