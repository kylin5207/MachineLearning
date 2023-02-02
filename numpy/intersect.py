"""
交集计算
"""

import numpy as np

a = np.arange(10, 20)
b = np.arange(15, 23)
print(f"a = {a}")
print(f"b = {b}")

# np.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)
# Find the intersection of two arrays.
# Return the sorted, unique values that are in both of the input arrays.
# 返回两个数组中排序好的交集数据
# 参数：
# assume_unique: If True, the input arrays are both assumed to be unique, which can speed up the calculation
intersect_2 = np.intersect1d(a, b, assume_unique=True)
print(f"intersect of [a, b] = {intersect_2}")

# multi arrays intersect 多数组求交集
# 使用reduce(function, sequence[, initial])函数：
#   对sequence连续使用function, 如果不给出initial, 则第一次调用传递sequence的两个元素, 以后把前一次调用的结果和sequence的下一个元素传递给function. 如果给出initial, 则第一次传递initial和sequence的第一个元素给function.
from functools import reduce
c = np.arange(17, 29)
print(f"c = {c}")
intersect_3 = reduce(np.intersect1d, (a, b, c))
print(f"intersect of [a, b, c] = {intersect_3}")

