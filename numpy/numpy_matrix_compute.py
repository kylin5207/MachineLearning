import numpy as np

"""
*和@对比：
· 运算符*在矩阵运算中的功能是逐元素的乘法， hadamard product
· 运算法@在矩阵运算中的功能是矩阵乘法， a@b和np.matmul(a,b)的结果是一致的，此外，二维数组下numpy.dot和numpy.matmul的结果是一样的
我们在看python程序时，经常可以看到@运算符和*运算符，其中@运算符在传统python中通常是作为装饰器使用的。但是在Python 3.5之后，它又具备了矩阵乘法运算的功能
"""

print(">>>>>>>>one dimension>>>>>>")
v1 = np.arange(5)
v2 = np.ones_like(v1)
print(f"v1 = {v1}")
print(f"v2 = {v2}")

print("===== v1 * v2 =====")
print(v1 * v2)
print("===== v1 @ v2 =====")
print(v1 @ v2)
print("===== np.matmul =====")
print(np.matmul(v1, v2))
print("===== np.dot =====")
print(np.dot(v1, v2))


print(">>>>>>>>two dimension>>>>>>")
a = np.arange(1, 10).reshape(3,3)
print("====a===")
print(a)

b = np.ones(shape=(3,3))
print("====b====")
print(b)

# 运算符*在矩阵运算中的功能是逐元素的乘法， hadamard product
print("==== a * b ====")
print(a * b)

# a@b计算矩阵乘法
print("==== a @ b ====")
print(a @ b)
print("==== np.matmul ====")
print(np.matmul(a, b))
print("==== np.dot ====")
print(np.dot(a, b))