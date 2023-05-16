import numpy as np
"""
numpy.array_split方法用于将一个NumPy数组分割成多个子数组。
numpy.array_split(ary, indices_or_sections, axis=0)

参数说明：
ary: 要分割的NumPy数组。
indices_or_sections: 可以是一个整数或一维整数数组。如果是整数，表示要分割的子数组个数；如果是一维整数数组，表示分割点的索引位置。
                     小于1时会报错：number sections must be larger than 0
axis（可选）：表示要沿着哪个轴进行分割。默认值为0，表示沿着第一个轴分割。
"""

data = np.random.randn(350, 20)
sample_count = data.shape[0]
batch_size = 100
# 生成索引
_z = np.arange(sample_count)

# 分割数组
splited = np.array_split(data, sample_count // batch_size + 1)
print(len(splited))
print(splited)


