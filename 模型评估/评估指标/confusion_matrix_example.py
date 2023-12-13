"""
confusion_matrix 参数说明
- y_true:
  类型：array-like（数组形式的对象），形状 [n_samples]。
  描述：这是真实的目标值。它包含了每个样本的实际分类标签。

- y_pred:
  类型：array-like，形状 [n_samples]。
  描述：这是模型预测的目标值。它包含了每个样本的预测分类标签。

- labels （可选）:
  类型：array-like，形状 [n_classes]。
  描述：这是用于构建混淆矩阵的标签数组。如果未指定，则从 y_true 和 y_pred 中自动推断出标签。

- sample_weight （可选）:
  类型：array-like，形状 [n_samples]。
  描述：样本权重。如果提供，则每个样本的贡献将根据其权重进行缩放，这对于处理非均衡数据集或赋予某些样本更高的重要性非常有用。默认为 None。

"""
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

def compute_confusion_matrix(labels, pred_labels_list, true_labels_list):
    # y_pred
    y_pred = np.asarray(pred_labels_list)
    y_true = np.asarray(true_labels_list)
    matrix = confusion_matrix(y_true,
                              y_pred,
                              labels=labels)
    return matrix

def reconstruct_confusion_matrix(cm):
    """
    重新排列混淆矩阵元素位置
    :param cm:
    :return:
    """
    # 从混淆矩阵中提取元素
    TN, FP, FN, TP = cm.ravel()

    # 重新排列为 TP, FN, FP, TN
    new_cm = np.asarray([[TP, FN], [FP, TN]])
    return new_cm

def compute_acc_withCM(matrix):
    """
    根据混淆矩阵计算总体准确率
    Args:
        matrix: numpy.ndarray, confusion matrix.
    """
    return np.trace(matrix)/np.sum(matrix)

def compute_acc(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.shape[0]

def compute_aa(matrix):
    """
    计算每一类的准确率,AA=(TP/(TP+FN)+TN/(FP+TN))/2
    :param matrix:
    :return:
    """
    return np.mean(np.diag(matrix) / np.sum(matrix, axis=1))

def compute_kappa(y_true, y_pred, label):
    """
     计算kappa系数
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param label: 标签类型
    :return:
    """
    # 计算实际一致性
    po = compute_acc(y_true, y_pred)
    # 计算混淆矩阵
    matrix = compute_confusion_matrix(label, y_pred, y_true)
    # 计算偶然一致性
    pe = 0
    for i in range(len(matrix)):
        pe += np.sum(matrix[i]) * np.sum(matrix[:, i])
    pe = pe / np.sum(matrix) ** 2
    return (po - pe) / (1 - pe)

# 真实标签
y_true = np.asarray([1, 0, 1, 1, 0, 1])
labels = np.unique(y_true)

# 预测标签
y_pred = np.asarray([0, 0, 1, 1, 0, 1])

# 计算混淆矩阵
cm = compute_confusion_matrix(labels,y_pred, y_true)
# 默认顺序是[[TN, FP], [FN, TP]]
print(f"confusion matrix = {cm}")
print(f"reconstruct confusion matrix = {reconstruct_confusion_matrix(cm)}")

# 计算ACC
print(f"acc with matrix = {compute_acc_withCM(cm)}")
print(f"acc = {compute_acc(y_true, y_pred)}")
print(f"acc sklearn = {accuracy_score(y_true, y_pred)}")

# averacy acc
aa = compute_aa(cm)
print(f"averacy acc = {aa}")

# compute kappa
kappa = compute_kappa(y_true, y_pred, labels)
print(f"kappa = {kappa}")
