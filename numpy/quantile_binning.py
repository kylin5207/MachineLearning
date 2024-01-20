"""
利用numpy实现等频分箱
"""

import numpy as np
import pandas as pd

def binning_with_numpy(x: np.ndarray, bucket_num: int):
    """利用numpy实现等频分箱：先用numpy.quantile计算分位点，再用numpy.digitize完成分箱映射。
    """
    quantiles = [i / bucket_num for i in range(1, bucket_num + 1, 1)]
    print(f"quantiles = {quantiles}")

    # np.quantile得到分位点
    split_points = np.unique(np.quantile(x, quantiles))
    print(f"split_points = {split_points}")

    # 得到每个样本划分到对应bin的索引(numpy.digitize 会将输入数组中的每个元素映射到分割点数组定义的区间中)
    # 例如，如果分割点数组是 [1, 3, 5, 7]，则定义了以下区间：(-∞, 1), [1, 3), [3, 5), [5, 7), [7, +∞)。
    bins = np.digitize(x, split_points)

    return bins


if __name__ == "__main__":
    bucket_num = 4

    # generate data
    x = np.random.randn(20)
    print(f"x = {x}")

    # binning with numpy
    numpy_bins = binning_with_numpy(x, bucket_num)
    print(f"numpy bins = {numpy_bins}")

    # binning with pandas
    # 如果数据中有重复值，可能会导致桶的大小不完全一样。
    # 如果数据中有 NaN 值，它们会被自动排除在分箱之外。
    # labels=False 参数返回每个数据点的桶索引，而不是桶的标签。
    pandas_bins = pd.qcut(x, bucket_num, labels=False)
    print(f"pandas bins = {pandas_bins}")

