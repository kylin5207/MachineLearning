"""
牛顿迭代法计算凸函数的极小值
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def f(x):
    '''待求解函数  f(x)=2x^2-10x'''
    return 2 * x**2 - 10 * x


def df(x):
    '''函数导数 f'(x)=4x-10'''
    return 4 * x - 10


def ddf(x):
    '''函数二阶导数 f''(x) = 4'''
    return 4


def plot(x, y, x_iters):
    print(x_iters)
    y_iters = f(x_iters)
    y_iters_max = y_iters.max()
    y_iters_min = y_iters.min()

    plt.plot(x, y)
    plt.xlabel("x")
    plt.title(r"$f(x)=2x^2-10x$")
    plt.scatter(x_iters, y_iters, s=45)

    for i, x_i in enumerate(x_iters):
        y_i = y_iters[i]
        print(f"x_i = {x_i}, min = {min(y_iters_min, y_i)}, max={min(y_iters_max, y_i)}")

        plt.vlines(x_i, ymin=min(y_iters_min, y_i), ymax=min(y_iters_max, y_i), linestyles="--", colors='gray')

        # 在垂线下方显示x对应的值
        plt.annotate(f'iter{i}={x_i}', (x_i, 0), xytext=(x_i, -3),
                     textcoords='offset points', ha='center', fontsize=12, color='red')

    plt.show()


def main():
    x = np.arange(-10, 10, 0.1)
    y = f(x)

    x0 = -10 # 初始x0
    xt = [x0] # 记录每次迭代的点x
    max_iter = 20 # 最大迭代次数
    tol = 1e-4 # 停止迭代阈值

    # newton iteration
    for i in range(max_iter):
        x0 = x0 - df(x0) / ddf(x0)

        if i > 0 and abs(x0 - xt[-1]) < tol:
            break
        else:
            xt.append(x0)

    xt = np.array(xt)

    plot(x, y, xt)


if __name__ == "__main__":
    main()