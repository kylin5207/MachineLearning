"""
numpy.vectorize(func):
会使用numpy的frompyfunc()函数将输入的Python函数转换为一个ufunc(Universal Function)对象，
然后将该ufunc对象应用于输入的numpy数组的每个元素上，得到一个与输入数组形状相同的输出数组。
因为ufunc对象被优化为能够在C语言级别上执行，所以在numpy数组上运行向量化函数的速度非常快，通常比使用Python循环执行要快得多。
"""

import matplotlib.pyplot as plt
import numpy as np

def calculate(x):
    if x < -0.5:
        return 0
    elif x > 0.5:
        return 1
    else:
        print(x+0.5)
        return x + 0.5

vec_calculate = np.vectorize(calculate, otypes=[float])

x = np.linspace(-1, 1, 1000)
y = vec_calculate(x)

plt.plot(x, y)
plt.show()