import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

def f(x):
    return 3 * x**2 - 4 * x

def gradient(x):
    return 6*x - 4

def plot(x, y, line):
    plt.plot(x, y, "b-", label=r"$y=f(x)$")
    plt.plot(x, line, "r--", label=r"g=2x-3")
    plt.xlabel("x")
    plt.legend()
    plt.grid()
    plt.show()

x = np.arange(0, 3, 0.1)
y = f(x)
# 计算x=1处的导数
x_value = 1
gradient_x = gradient(1)
y_value = f(x_value)

# 计算导数曲线
line = gradient_x * (x-x_value) + y_value

plot(x, y, line)

