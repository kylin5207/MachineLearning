import scipy.stats as stats
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

# 两个变量的数据
x = X["mean radius"]

# 直接调包计算计算Spearman相关系数
rho, p_value = stats.spearmanr(x, y)
print(f"rho by spearmanr = {rho}")

# 计算rankdata
x_rank = stats.rankdata(x, method="average")
y_rank = stats.rankdata(y, method="average")
print(stats.pearsonr(x_rank, y_rank))
print(X.columns)

data = pd.DataFrame([[1,2,], [2,1], [2,2]], columns=list('ab'))
print(data)
print(pd.DataFrame(stats.rankdata(data, axis=0), columns=data.columns))
