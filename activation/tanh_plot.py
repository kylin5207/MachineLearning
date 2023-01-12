import numpy as np
import matplotlib.pyplot as plt

"""
tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)) = 2sigmoid(2x) - 1

图像分析：
- 图像和sigmoid函数非常像，其实就是直接在竖直方向拉伸两倍，然后在y轴向下平移了1个单位，使得函数的中心回到了0，然后在水平方向上拉伸两倍。
解决了sigmoid函数收敛变慢的问题，相对于sigmoid提高了收敛速度。

- 尽管tanh函数和sigmoid函数存在梯度消失的问题，但是与之类似，如果函数的梯度过大又会导致梯度爆炸的问题，显然tanh和sigmoid的导函数非常有界

梯度消失：
好比你在往下走楼梯，楼梯的梯度很小，你感觉不到在下楼。放在ml里面，就是在梯度下降公式里w_new = w_old - rate*gradient。
当导数部分很小很小（可能接近于0—）， 导致训练极度缓慢（变化很小），这种现象就叫梯度消失，一般是由训练层的激活函数导致的。

所以尽量避免使用导函数值小的函数，sigmoid函数的梯度随着x的增大或减小和消失，而ReLU不会。
但是也要看具体情况，sigmoid函数值在[0,1],ReLU函数值在[0,+无穷]，所以sigmoid函数可以描述概率，ReLU适合用来描述实数。

"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(y):
    return y * (1-y)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def derivative_tanh(y):
    return 1 - y ** 2

def plot(x, y, title):
    plt.plot(x, y)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    x = np.linspace(-5, 5, 100)
    sig = sigmoid(x)
    plot(x, sig, r"$sigmoid = \frac{1}{1+e^{-x}}$")

    tanh_value = tanh(x)
    plot(x, tanh_value, r"$tanh=\frac{e^x - e^{-x}}{e^x + e^{-x}}$")

    # compare
    plt.plot(x, sig, label=r"$sigmoid = \frac{1}{1+e^{-x}}$")
    plt.plot(x, tanh_value, label=r"$tanh=\frac{e^x - e^{-x}}{e^x + e^{-x}}$")
    plt.legend()
    plt.show()

    # derivative compare
    derivative_sig = derivative_sigmoid(sig)
    derivative_tanh = derivative_tanh(tanh_value)
    plt.plot(x, derivative_sig, label=r"$sigmoid^{'}(x)=sigmoid(x) \times (1-sigmoid(x))$")
    plt.plot(x, derivative_tanh, label=r"$tanh^{'}(x)= 1 - tanh^2(x)$")
    plt.legend()
    plt.show()

    # s3 = sigmoid_log(x)
    # plot(x, s3, r"$log\_sigmoid=log\frac{1}{1+e^{-x}}$")
    #
    # plt.plot(x, s1, label=r"$sigmoid = \frac{1}{1+e^{-x}}$")
    # plt.plot(x, s2, label=r"$sigmoid_{hard}=\frac{x+1}{2}$")
    # plt.title("Sigmoid vs. Sigmoid_hard")
    # plt.legend()
    # plt.show()