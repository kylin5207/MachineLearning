# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:21:12 2019
绘制sdSwiss_roll图像，就像卷卷心一样
@author: Kylin
"""

from sklearn.datasets import make_swiss_roll
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d #一定要引入这个包，不然会报错
import matplotlib.pyplot as plt

#1. 生成swiss_roll数据集
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

#2. 绘制3d卷卷心图像
axes = [-11.5, 14, -2, 23, -12, 15]
ax = plt.axes(projection="3d")
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])
ax.set_title("swiss roll 3d Fig")