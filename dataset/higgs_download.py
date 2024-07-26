"""
higgs数据集
从官方下载后：https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
按照下列步骤解压
"""

import lzma
import pandas as pd

# 解压缩函数
def decompress_xz(input_path, output_path):
    with lzma.open(input_path) as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())
    print(f"Extracted {output_path}")

# 解压缩文件
input_path = 'HIGGS.xz'
output_path = 'higgs.csv'
decompress_xz(input_path, output_path)

# 读取CSV文件
df = pd.read_csv(output_path)

# 打印数据前几行
print(df.head())
