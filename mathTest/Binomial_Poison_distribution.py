import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.linear_model import LogisticRegression
"""
二项分布与柏松分布
"""

def compute_binomial(n, p):
    """ 计算二项分布的概率质量函数
    Args:
        n: int, n次实验
        p: int, 正的概率
    """
    value_list = []

    for i in range(n+1):
        value_list.append(scipy.stats.binom.pmf(i, n, 0.5))

    return value_list

def compute_possion(n, lamda):
    """
    计算柏松分布概率质量函数
    """
    value_list = []

    for i in range(n+1):
        value_list.append(scipy.stats.poisson.pmf(i, lamda))

    return value_list


def main():
    n = 10
    p = 0.5
    lamda = 5
    X = np.arange(n+1)
    binomial_value = compute_binomial(n, p)
    possion_value = compute_possion(n, lamda)

    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.bar(X, binomial_value, color='blue', edgecolor='black', alpha=0.5)
    plt.plot(X, binomial_value, color='blue', linewidth=3)
    plt.axvline(n * p, linestyle='--', color='blue')
    plt.title(r"$Binomial N(10, p=0.5)$")

    plt.subplot(1, 2, 2)
    plt.bar(X, possion_value, color='red', edgecolor='black', alpha=0.5)
    plt.plot(X, possion_value, color='red', linewidth=3)
    plt.title(r"$poisson Poi(\lambda=5)$")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
