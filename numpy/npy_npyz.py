"""
使用NumPy来保存数据集
NumPy的np.save()和np.savez()函数提供了方便的方法来保存和加载数据集，以NumPy数组的形式存储在磁盘上。这对于保存训练数据、标签等数据集非常有用。
"""
import numpy as np

# 使用np.save()保存单个NumPy数组, 这会将名为"data.npy"的文件保存到当前工作目录中，其中包含了你的NumPy数组。
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.save('data.npy', data)
# 加载保存的数据
loaded_data = np.load('data.npy')  # 对于 np.save() 保存的文件
print(loaded_data)


# 使用np.savez()保存多个NumPy数组，并可以使用键名来检索这些数组
# 这会将名为"data.npz"的文件保存到当前工作目录中，其中包含了多个你指定的NumPy数组，你可以使用键名来检索这些数组。
data1 = np.array([1, 2, 3])
data2 = np.array([4, 5, 6])
data3 = np.array([7, 8, 9])
np.savez('data.npz', data1=data1, data2=data2, data3=data3)
loaded_data = np.load('data.npz')  # 对于 np.savez() 保存的文件
print(loaded_data['data1'])
print(loaded_data['data2'])
print(loaded_data['data3'])