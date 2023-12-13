"""
sklearn.metrics.recall_score 参数说明
参数说明：
- y_true: 真实的目标值，即数据集中的实际标签。
- y_pred: 预测的目标值，即模型预测的标签。
- labels (可选): 需要考虑的标签列表。例如，在多分类问题中，你可以指定一组标签来计算这些特定标签的召回率。
- pos_label (可选): 在二分类问题中，用于表示正类的标签。默认情况下，pos_label 被设定为 1。
- average (可选): 用于多类别问题的平均计算方法。它的值可以是以下之一：
  - None: 不进行平均，返回每个类的召回率。
  - 'binary': 仅报告由 pos_label 指定的类的召回率，用于二分类问题。
  - 'micro': 通过计算总的真正例、假负例和假正例来全局计算指标。
  - 'macro': 计算每个类别的召回率，然后找到它们的未加权均值。不考虑标签不平衡。
  - 'weighted': 计算每个类别的召回率，然后按照每个类别的真实实例数加权平均。
    如果你在处理不平衡的多分类问题，可以选择使用 'weighted' 平均。
  - 'samples': 适用于多标签分类问题。
- zero_division (可选): 当存在分母为零的情况时的行为。可以是 0、1 或 'warn'。如果是 0，则在分母为零时返回 0；如果是 1，则返回 1；如果是 'warn'，则会输出一个警告并返回 0。


"""
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import numpy as np

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

def calculate_recall(y_true, y_pred, positive_label=1):
    """
    手动实现recall的计算：recall=TP/(TP+FN)
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param positive_label: 正例表示
    :return:
    """
    # 初始化真正例和假负例的计数
    true_positives = 0
    false_negatives = 0

    # 遍历所有的真实标签和预测标签
    for true, pred in zip(y_true, y_pred):
        if true == positive_label:
            if pred == positive_label:
                true_positives += 1  # 真正例
            else:
                false_negatives += 1  # 假负例

    # 避免除以零的情况
    if true_positives + false_negatives == 0:
        return 0
    else:
        return true_positives / (true_positives + false_negatives)

def calculate_precision(y_true, y_pred, positive_label=1):
    """
    手动实现precision的计算：precision=TP/(TP+FP)
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param positive_label: 正例表示
    :return:
    """
    # 初始化真正例和假正例的计数
    true_positives = 0
    false_positives = 0

    # 遍历所有的真实标签和预测标签
    for true, pred in zip(y_true, y_pred):
        if pred == positive_label:
            if true == positive_label:
                true_positives += 1  # 真正例
            else:
                false_positives += 1  # 假正例

    # 避免除以零的情况
    if true_positives + false_positives == 0:
        return 0
    else:
        return true_positives / (true_positives + false_positives)


def main():
    # 真实的标签
    y_true = [0, 1, 1, 0, 1, 0, 1, 1, 1, 0]

    # 预测的标签
    y_pred = [0, 1, 0, 0, 1, 0, 0, 1, 1, 0]

    # compute matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cm = reconstruct_confusion_matrix(cm)
    print(f"confusion matrix = {cm}")

    # 计算召回率
    handy_recall = calculate_recall(y_true, y_pred, positive_label=1)
    sk_recall = recall_score(y_true, y_pred)
    print("handy Recall:", handy_recall)
    print("sklearn Recall:", sk_recall)

    # 计算精准率
    handy_precision = calculate_precision(y_true, y_pred, positive_label=1)
    sk_precision = precision_score(y_true, y_pred)
    print("handy Precision:", handy_precision)
    print("sklearn Precision:", sk_precision)


if __name__ == "__main__":
    main()