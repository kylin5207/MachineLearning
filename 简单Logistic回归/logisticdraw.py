# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:39:42 2019
绘制sigmoid=1/(1+e-x)，其和logit=log(p/(1-p))互为反函数
@author: Kylin
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
p = np.linspace(0, 1, 100)

logit = np.log(p / (1 - p))
logistic = 1 / (1+np.exp(-x))

plt.subplot(1, 2, 1)
plt.plot(x, logistic)
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"$logistic = \frac{1}{1+e^{-x}}$")

plt.subplot(1, 2, 2)
plt.plot(x, logit)
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"$logistic = \frac{p}{1 - p}$")

plt.show()
