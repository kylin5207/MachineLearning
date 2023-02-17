"""
SGDClassifier的两种拟合方式
-  fit：从头开始对训练数据进行一次完整的迭代
- partial_fit: 用于增量训练，即逐步地将新的数据添加到模型中进行训练，可以在多次调用中逐步传入数据
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def get_next_batch(X, y, batch_size=20):
    """逐步获取批量数据的生成器函数"""
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size

    # 分批获取数据
    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        yield X[start:end], y[start:end]

    # 如果数据集不能完全被batch_size整除，最后一个batch会包含剩余的所有样本
    if n_batches * batch_size < n_samples:
        yield X[n_batches * batch_size:], y[n_batches * batch_size:]


def full(X, y):
    sgd_full = SGDClassifier(loss="log_loss", max_iter=10, learning_rate="constant", eta0=0.1)
    sgd_full.fit(X, y)

    return sgd_full


def partial(X, y):
    sgd_partial = SGDClassifier(loss="log_loss", max_iter=10, learning_rate="constant", eta0=0.1)

    for X_batch, y_batch in get_next_batch(X, y):
        sgd_partial.partial_fit(X_batch, y_batch, classes=[0, 1])

    return partial


def plot_compare(y, pred_full, pred_partial):
    x = np.arange(y.shape[0])

    index_false = np.where(y == 0)
    index_true = np.where(y == 1)

    plt.figure(figsize = (10,10))
    plt.subplot(2,1,1)
    plt.scatter(x[index_false], pred_full[index_false], color="red", alpha=1, edgecolors='k')
    plt.scatter(x[index_false], pred_partial[index_false], color="green", alpha=0.2, edgecolors='k')
    plt.title("y_label equals False")

    plt.subplot(2,1,2)
    plt.scatter(x[index_true], pred_full[index_true], color="red", alpha=1, edgecolors='k')
    plt.scatter(x[index_true], pred_partial[index_true], color="green", alpha=0.2, edgecolors='k')
    plt.title("y_label equals True")
    plt.show()


def main():
    # load data
    breast = load_breast_cancer()
    X, y = breast.data, breast.target
    # scaler
    X = StandardScaler().fit_transform(X)

    # shuffle
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]

    sgd_full = full(X, y)
    sgd_partial = full(X, y)
    pred_full = sgd_full.predict_proba(X)[:, 1]
    pred_partial = sgd_partial.predict_proba(X)[:, 1]

    # compare auc
    full_auc = roc_auc_score(y, pred_full)
    partial_auc = roc_auc_score(y, pred_partial)
    print(f"auc compare, fit = {full_auc}, partial_fit = {partial_auc}")

    # compare loss
    full_loss = log_loss(y, pred_full)
    partial_loss = log_loss(y, pred_partial)
    print(f"loss compare, fit = {full_loss}, partial_fit = {partial_loss}")

    # plot to compare
    plot_compare(y, pred_full, pred_partial)


if __name__ == "__main__":
    main()

