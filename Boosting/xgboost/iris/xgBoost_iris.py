# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split   # cross_validation
import pandas as pd

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    path = u'iris.data'  # 数据文件路径
    #利用pandas加载数据集
    #前4列是特征，最后一列是类标
    data = pd.read_csv(path, names=["sepal-length", "sepal-width",
                                     "petal-length", "petal-width", "label"])
    #为了便于处理数据，将label标记为int型数据
    data["label"] = pd.Categorical(data["label"]).codes
    
    
    #为了可视化，仅使用花萼长度和花瓣长度作为特征
    x = np.array(data[["sepal-length", "sepal-width",
                       "petal-length", "petal-width"]])
    y = np.array(data["label"])
    
    #划分数据集，训练集中有100个样本，测试集中有50个样本
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=50)

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    #多分类器objective:multi:softmax，顺便给出类别个数num_class
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}

    bst = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)
    y_hat = bst.predict(data_test)
    result = y_test.reshape(1, -1) == y_hat
    print('正确率:\t', float(np.sum(result)) / len(y_hat))
    
    # show xgboost tree
    digraph = xgb.to_graphviz(bst, num_trees=0)
    digraph.format = 'png'
    digraph.view('./xgb_demo')
