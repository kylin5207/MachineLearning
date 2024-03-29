#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
模型存储
非常实用
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os
from sklearn.externals import joblib


if __name__ == "__main__":
    data = pd.read_csv('iris.data', header=None)
    x = data[[0, 3]]
    y = pd.Categorical(data[4]).codes

    if os.path.exists('iris.model'):
        print('=======加载已有模型=======')
        lr = joblib.load('iris.model')

    else:
        print('=======训练模型=======')
        lr = Pipeline([('sc', StandardScaler()),
                       ('poly', PolynomialFeatures(degree=3)),
                       ('clf', LogisticRegression())])

        lr.fit(x, y.ravel())
        # 存储模型
        joblib.dump(lr, 'iris.model')

    # 预测
    y_hat = lr.predict(x)
    print('y_hat = \n', y_hat)
    print('accuracy = %.3f%%' % (100*accuracy_score(y, y_hat)))
