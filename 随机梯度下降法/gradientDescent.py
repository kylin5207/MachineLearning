# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:35:03 2019
使用梯度下降法拟合单变量线性回归
@author: Kylin
"""
import numpy as np
import matplotlib.pyplot as plt


class gradientdescent(object):
    """
    简单梯度下降法
    参数：
    -----------
    eta:float类型，学习率【0,1】
    n_iter:int类型，训练次数
    -----------
    属性：
    -----------
    w_:1darray, 有两个系数，单变量线性回归的系数向量
    errors_:list, 每次训练时错误分类的个数
    -----------
    """
    def __init__(self, eta=0.01, n_iter=1000):
        """
        学习率默认0.01， 训练次数默认为50次
        """
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """
        训练并拟合数据
        参数：
        -------------
        X：数组，shape={n_samples}
            训练向量，n_samples是样本数
        y：shape={n_samples}
            目标值
        -------------
        返回值：
        self.w_和self.cost_
        """
        self.w_ = np.zeros(2)
        self.cost_ = []
        n = X.shape[0]
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = output - y
            print("errors", errors)
            J0 = errors.sum()/n
            # n是样本数
            # 注意(X.dot(errors))计算出来的是一个(n,)维数组
            J1 = (X.dot(errors)).sum()/n
            self.w_[0] -= self.eta * J0
            self.w_[1] -= self.eta * J1
            cost = (errors**2).sum() /(2.0*n)
            self.cost_.append(cost)
        return self.w_, self.cost_
    
    def net_input(self, X):
        """计算所有X的净输入"""
        net = np.dot(X, self.w_[1]) + self.w_[0]
        print("net:", net)
        return net


if __name__ == "__main__":
    X = np.linspace(0, 4, 5)
    y = X * 3 + 1 + np.random.randn(5)
    n = 5
    print("X", X)
    print("y", y)
    
    gd = gradientdescent()
    weights, cost = gd.fit(X, y)
    liner = weights[0] + weights[1]*X
    print("假定线性函数是：y=3x+1")
    print("The regression liner is:")
    print("y = {:.5f} + {:.5f}x".format(weights[0], weights[1]))
    
    fig, ax = plt.subplots(2,1)
    fig.subplots_adjust(hspace=0.5, wspace=1)
    ax[0].scatter(X, y, color='red')
    ax[0].plot(X, liner)
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    
    iternum = np.arange(gd.n_iter)
    ax[1].plot(iternum, cost)
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Cost")
    plt.suptitle("Linear Regression with Gradient Descent Algorithm")
    plt.show()
    
    
