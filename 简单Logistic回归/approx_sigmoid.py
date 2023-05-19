import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

"""
Sigmoid的三种计算方法
"""

def exact_sigmoid(X, plot=False):
    y = expit(X)

    # plot
    if plot:
        plt.plot(X, y)
        plt.ylim(0, 1)
        plt.title(r"$y = \frac{1}{1+e^{-x}}$")
        plt.show()
    return y

def taler3_sigmoid(X, plot=False):
    y = 0.5 + 0.25*X - 0.0104*np.power(X,3)

    # plot
    if plot:
        plt.plot(X, y)
        plt.ylim(0, 1)
        plt.title(r"$y = 0.5+0.25x+0.0104x^3$")
        plt.show()

    return y


def tanh_plot(X, plot=False):
    y = np.tanh(X)

    # plot
    if plot:
        plt.plot(X, y)
        plt.title(r"$ y = tanh(x)$")
        plt.show()
    return y


def minimax_sigmoid(X, plot=False):
    y = 0.5 * (1 + np.tanh(0.5 * X))

    # plot
    if plot:
        plt.plot(X, y)
        plt.ylim(0, 1)
        plt.title(r"$ y = \frac{1}{2}(1+tanh(\frac{x}{2}))$")
        plt.show()
    return y


def minmax2_sigmoid(X, plot=False):
    y = -0.004 * np.power(X, 3) + 0.197 * X + 0.5
    print(-0.004 * np.power(4, 3) + 0.197 * 4 + 0.5)
    # plot
    if plot:
        plt.plot(X, y)
        plt.title(r"$y = -0.004x^3 + 0.197x+0.5$")
        plt.show()
    return y


def piecewise_sigmoid(X, plot=False):
    def f1(x):
        return 0

    def f2(x):
        return x+0.5

    def f3(x):
        return 1

    conditions = [X < -0.5, (X >= -0.5) & (X <= 0.5), X > 0.5]
    # functions = [f1, f2, f3]
    functions = [0, f2, 1]
    y = np.piecewise(X, conditions, functions)

    # plot
    if plot:
        plt.plot(X, y)
        plt.title(r"$y=piecewise$")
        plt.show()
    return y

def main():
    X = np.linspace(-5, 5, 100)

    ## exact and minmax sigmoid
    # y_exact = exact_sigmoid(X)
    # y_minmax = minimax_sigmoid(X)
    # # compare
    # plt.plot(X, y_exact, color="blue", linestyle="-.", label=r"$exact = \frac{1}{1+e^{-x}}$")
    # plt.plot(X, y_minmax, color="red", linestyle="--", label=r"$minmax = \frac{1}{2}(1+tanh(\frac{x}{2}))$", alpha=0.7)
    # plt.ylim(0, 1)
    # plt.title("Sigmoid Compare")
    # plt.legend()
    # plt.show()

    ## exact and talor sigmoid
    # y_exact = exact_sigmoid(X)
    # y_talor = taler3_sigmoid(X, plot=True)
    # # compare
    # plt.plot(X, y_exact, color="blue", linestyle="-.", label=r"$exact = \frac{1}{1+e^{-x}}$")
    # plt.plot(X, y_talor, color="red", linestyle="--", label=r"$y = 0.5+0.25x+0.0104x^3$", alpha=0.7)
    # plt.ylim(0, 1)
    # plt.title("Sigmoid Compare")
    # plt.legend()
    # plt.show()

    ## exact and minmax2 sigmoid
    # y_exact = exact_sigmoid(X)
    # y_minmax2 = minmax2_sigmoid(X, plot=True)
    # # compare
    # plt.plot(X, y_exact, color="blue", linestyle="-.", label=r"$exact = \frac{1}{1+e^{-x}}$")
    # plt.plot(X, y_minmax2, color="red", linestyle="--", label=r"$y = -0.004x^3 + 0.197x+0.5$", alpha=0.7)
    # plt.ylim(0, 1)
    # plt.title("Sigmoid Compare")
    # plt.legend()
    # plt.show()

    ## exact and
    y_exact = exact_sigmoid(X)
    y_piecewise = piecewise_sigmoid(X, plot=True)
    # compare
    plt.plot(X, y_exact, color="blue", linestyle="-.", label=r"$exact = \frac{1}{1+e^{-x}}$")
    plt.plot(X, y_piecewise, color="red", linestyle="--", label="y=piecewise", alpha=0.7)
    # plt.ylim(0, 1)
    plt.title("Sigmoid Compare")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

