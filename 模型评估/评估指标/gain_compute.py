import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np

def cumulative_gain_curve(y_true, y_score, pos_label=1):
    """This function generates the points necessary to plot the Cumulative Gain

    Note: This implementation is restricted to the binary classification task.

    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.

        y_score (array-like, shape (n_samples)): Target scores, can either be
            probability estimates of the positive class, confidence values, or
            non-thresholded measure of decisions (as returned by
            decision_function on some classifiers).

        pos_label (int, default=1): Label considered as positive and
            others are considered negative

    Returns:
        percentages (numpy.ndarray): An array containing the X-axis values for
            plotting the Cumulative Gains chart.

        gains (numpy.ndarray): An array containing the Y-axis values for one
            curve of the Cumulative Gains chart.

    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The Cumulative
            Gain Chart is only relevant in binary classification.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)

    percentages = np.arange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains


def cumulative_gain_at_intervals(y_true, y_score, pos_label=1, interval=0.1):
    """
    计算间隔为interval的累积gain
    :param y_true: true labels of the data.
    :param y_score: target scores.
    :param pos_label: int, label considered as positive
    :param intervals: float, the interval at which to select gains (0.1 for 10%).

    :return:
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort
    sorted_indices = np.argsort(y_score)[::-1]

    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true) / np.sum(y_true)  # Cumulative gain
    percentages = np.arange(1, len(y_true) + 1) / len(y_true)  # Percentage of data

    # Insert initial point (0,0)
    gains = np.insert(gains, 0, 0)
    percentages = np.insert(percentages, 0, 0)

    # Select gains at specified intervals
    interval_gain = []
    interval_percentage = []
    for percentage_point in np.arange(interval, 1 + interval, interval):
        idx = np.abs(percentages - percentage_point).argmin()
        interval_gain.append(gains[idx])
        interval_percentage.append(percentage_point)

    return interval_percentage, interval_gain


# 加载数据
data = load_breast_cancer()
X = data.data
y = data.target

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练一个随机森林分类器
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 预测概率
y_probas = model.predict_proba(X_test)

# handy compute gain
percentages, gains = cumulative_gain_curve(y_test, y_probas[:, 1], pos_label=1)
print(gains)
plt.plot(percentages, gains)
plt.show()

# 计算interval为0.1的累积gain
percentages, gains = cumulative_gain_at_intervals(y_test, y_probas[:, 1], pos_label=1)
plt.plot(percentages, gains)
plt.show()

# 绘制Cumulative Gain图
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()