# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:02:24 2019
#对每对特征构建决策树
@author: Kylin
"""


# !/usr/bin/python
# -*- coding:utf-8 -*-
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import tree

# iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = u"Times New Roman" # 黑体 FangSong/KaiTi
    mpl.rcParams['axes.unicode_minus'] = False

    # 1. 利用pandas加载数据集
    path = 'iris.data'  # 数据文件路径
    data = pd.read_csv(path, names=["sepal-length", "sepal-width",
                                     "petal-length", "petal-width", "label"])
    # 为了便于处理数据，将label标记为int型数据
    data["label"] = pd.Categorical(data["label"]).codes

    # x_prime为特征数组，y为类标数组
    x_prime = np.array(data[["sepal-length", "sepal-width", "petal-length", "petal-width"]])
    y = np.array(data["label"])

    # 2. 根据不同的特征对，训练决策树
    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    plt.figure(figsize=(10, 9), facecolor='#FFFFFF')
    for i, pair in enumerate(feature_pairs):
        # 2.1 准备数据
        x = x_prime[:, pair]

        # 2. 划分训练测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
        
        # 3. 决策树学习
        clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3, max_depth=3)
        dt_clf = clf.fit(x_train, y_train)

        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=None, filled=True, rounded=True)  # 重要参数可定制
        graph = graphviz.Source(dot_data)
        graph.render(view=True, format="pdf", filename="decisiontree_plot")

        # 画图
        N, M = 500, 500  # 横纵各采样多少个值
        x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
        x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
        x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
        y_show_hat = dt_clf.predict(x_show)  # 预测值
        y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同
        
        # 训练集上的预测结果
        y_hat = dt_clf.predict(x)
        y = y.ravel()
        c = np.count_nonzero(y_hat == y)    # 统计预测正确的个数
        print('特征：  ', iris_feature[pair[0]], ' + ', iris_feature[pair[1]])
        print('\t预测正确数目：', c)
        print('\t准确率: %.2f%%' % (100 * float(c) / float(len(y))))

        # 4. 显示与绘制
        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

        # 绘制子图
        plt.subplot(2, 3, i+1)
        plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值
        plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.ravel(), edgecolors='k', cmap=cm_dark)  # 样本
        plt.xlabel(iris_feature[pair[0]], fontsize=14)
        plt.ylabel(iris_feature[pair[1]], fontsize=14)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.grid()

    plt.suptitle(u'Iris Category Prediction By Decision Tree with Two Features', fontsize=18)
    plt.subplots_adjust(top=0.92)
    plt.show()
