# encoding:utf-8
from math import pi, exp

import numpy as np
from Basic import NaiveBayes

# 记录 根号2pi，避免重复计算
sqrt_pi = (2 * pi) ** 0.5


class NBFunctions:
    # 定义正态分布的密度函数
    '''
    @staticmethod
    def gaussian(x, mu, sigma):
        return exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt_pi * sigma)
    '''

    # 向量化
    @staticmethod
    def gaussian(x, mu, sigma):
        return np.exp(-(x - mu) ** 2 / (2 * sigma)) / (sqrt_pi * sigma ** 0.5)

    # 批量处理
    @staticmethod
    def gaussian_batch(xx, mu, sigma):
        yy = []
        for x in xx:
            yy.append(exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt_pi * sigma))
        return yy

    # 定义进行极大似然估计的函数
    # 返回一个存储着计算条件概率密度函数的列表
    @staticmethod
    def gassian_maximum_likelihood(labelled_x, n_category, dim):
        # 计算mu ，平均值
        mu = [np.sum(labelled_x[c][dim]) / len(labelled_x[c][dim]) for c in range(n_category)]
        # 计算sigma 方差
        sigma = [np.sum((labelled_x[c][dim] - mu[c]) ** 2) / len(labelled_x[c][dim]) for c in range(n_category)]

        # 利用极大似然估计得到mu和sigma，定义生成计算条件概率密度的函数func
        def func(_c):
            def sub(xx):
                return NBFunctions.gaussian(xx, mu[_c], sigma[_c])

            return sub

        # 利用func返回目标列表
        return [func(_c=c) for c in range(n_category)]


class GaussianNB(NaiveBayes):
    def feed_data(self, x, y, sample_weight=None):
        # 简单的利用python自带的float方法将输入数据数值化
        x = np.array([list(map(lambda c: float(c), sample)) for sample in x])
        # 数值化类别向量
        labels = list(set(y))
        label_dic = {label: i for i, label in enumerate(labels)}
        y = np.array([label_dic[yy] for yy in y])
        cat_conter = np.bincount(y)
        labels = [y == value for value in range(len(cat_conter))]
        labelled_x = [x[label].T for label in labels]
        # 更新模型的各个属性
        self._x, self._y = x.T, y
        self._labelled_x, self._label_zip = labelled_x, labels
        self._cat_counter = cat_conter
        self._label_dic = [{i: _l} for _l, i in label_dic.items()]
        self.feed_sample_weight(sample_weight)

    # 定义处理样本权重的函数
    def feed_sample_weight(self, sample_weight=None):
        if sample_weight is not None:
            local_weight = sample_weight * len(sample_weight)
            for i, label in enumerate(self._label_zip):
                self._labelled_x[i] *= local_weight[label]

    def _fit(self, lb):
        n_category = len(self._cat_counter)
        p_catefory = self.get_prior_probability(lb)
        # 利用极大似然法估计获得条件概率函数，使用数组变量data进行存储
        data = [
            NBFunctions.gassian_maximum_likelihood(self._labelled_x, n_category, dim) for dim in range(len(self._x))
        ]
        self._data = data
        '''
        def func(input_x, tar_category):
            rs = 1
            for d, xx in enumerate(input_x):
                # 由于data中存储的事函数，所有需要调用他来进行条件概率的计算
                rs *= data[d][tar_category](xx)
            return rs * p_catefory[tar_category]
        '''

        # 向量化
        def func(input_x, tar_category):
            # 将输入转化为二维数组
            input_x = np.atleast_2d(input_x).T
            rs = np.ones(input_x.shape[1])
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category](xx)
            return rs * p_catefory[tar_category]

        return func

    @staticmethod
    def _transfer_x(self, x):
        return x
