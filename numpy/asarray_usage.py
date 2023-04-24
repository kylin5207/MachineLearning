"""
numpy.asarray(a, dtype=None, order=None) 是 Python 的 NumPy 库中的一个函数，用于将输入对象转换为 ndarray 对象（即多维数组）。
该函数可以接受各种类型的输入，包括列表、元组、其他数组和一些其他可迭代对象，
并返回与输入具有相同数据类型和形状的数组。如果输入已经是 ndarray，则该函数将简单地返回其本身。
参数：
    a 是要转换为数组的输入对象
    dtype 是输出数组的可选数据类型
    order 是输出数组中存储元素的可选顺序。
"""

import numpy as np

list1 = [1, 2, 3, 4]
arr1 = np.asarray(list1)
print(arr1)  # 输出：[1 2 3 4]

tuple1 = ((1, 2), (3, 4))
arr2 = np.asarray(tuple1)
print(arr2)  # 输出：[[1 2]
             #       [3 4]]

