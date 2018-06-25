# encoding:utf-8
import time

import matplotlib.pyplot as plt
import numpy as np
from Basic import NaiveBayes
from GaussianNB import GaussianNB, NBFunctions
from MultinomialNB import MultinomialNB
from Util import DataUtil
from pylab import mpl


class MergedNB(NaiveBayes):
    '''
    初始化结构
    self._whether_discrete:记录各个维度的变量是否为离散型变量
    self._whether_continuous:记录各个维度的变量是否为连续性变量
    self._multinomial,self._guassian:离散型、连续性朴素贝叶斯模型
    '''

    def __init__(self, whether_continuous):
        self._multinomial, self._guassian = MultinomialNB(), GaussianNB()
        if whether_continuous is None:
            self._whether_discrete = self._whether_continuous = None
        else:
            self._whether_continuous = np.array(whether_continuous)
            self._whether_discrete = ~self._whether_continuous

    # 分别利用MultinomialNB和GuassianNB的数据预处理方法进行数据预处理
    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
        # 这里的quantize_data方法正式之前离散型朴素贝叶斯数值化数据过程的抽象
        x, y, wc, features, feat_dics, label_dic = DataUtil.quantize_data(x, y, wc=self._whether_continuous,
                                                                          separate=True)
        # 若没有指定哪些维度连续，则采用quantize_data中朴素的方法判定那些维度连续
        if self._whether_continuous is None:
            self._whether_continuous = wc
            # 通过numpy中对逻辑非的支持进行快速运算
            self._whether_discrete = ~self._whether_continuous
        self.label_dic = label_dic
        discrete_x, continuous_x = x
        cat_counter = np.bincount(y)
        self._cat_counter = cat_counter

        labels = [y == value for value in range(len(cat_counter))]
        # 训练离散型朴素贝叶斯
        labelled_x = [discrete_x[ci].T for ci in labels]
        self._multinomial._x, self._multinomial._y = x, y
        self._multinomial._labelled_x, self._multinomial._label_zip = (labelled_x, list(zip(labels, labelled_x)))
        self._multinomial._cat_counter = cat_counter
        self._multinomial._feat_dics = [_dic for i, _dic in enumerate(feat_dics) if self._whether_discrete[i]]
        self._multinomial._n_possibilities = [len(feats) for i, feats in enumerate(features) if
                                              self._whether_discrete[i]]
        self._multinomial.label_dic = label_dic
        # 训练连续性朴素贝叶斯
        labelled_x = [continuous_x[label].T for label in labels]
        self._guassian._x, self._guassian._y = continuous_x.T, y
        self._guassian._labelled_x, self._guassian._label_zip = labelled_x, labels
        self._guassian._cat_counter, self._guassian.label_dic = cat_counter, label_dic
        # 处理样本权重
        self.feed_sample_weight(sample_weight)

    # 分别利用MultinomialNB和GuassianNB处理样本权重的方法来处理样本权重
    def feed_sample_weight(self, sample_weight=None):
        self._multinomial.feed_sample_weight(sample_weight)
        self._guassian.feed_sample_weight(sample_weight)

    # 分别利用MultinomialNB和GuassianNB训练函数来进行训练
    def _fit(self, lb):
        self._multinomial.fit()
        self._guassian.fit()
        p_category = self._multinomial.get_prior_probability(lb)
        discrete_func, continuous_func = self._multinomial['func'], self._guassian['func']

        # 将MultinomialNB和GuassianNB的决策函数直接合成MergeNB的决策函数
        # 由于这两个决策函数都乘了先验概率，需要出掉一个先验概率
        '''
        def func(input_x, tar_category):
            input_x = np.array(input_x)
            return discrete_func(input_x[self._whether_discrete].astype(np.int), tar_category) * continuous_func(
                input_x[self._whether_continuous], tar_category) / p_category[tar_category]
        '''

        # 向量化
        def func(input_x, tar_category):
            # 将输入转换为二维数组
            #input_x = np.atleast_2d(input_x)
            input_x = np.array(input_x)
            return discrete_func(input_x[:, self._whether_discrete].astype(np.int), tar_category) * continuous_func(
                input_x[:, self._whether_continuous], tar_category) / p_category[tar_category]

        return func

    # 实现转换混合型函数的方法，需主要利用MultinomialNB的相应变量
    def _transfer_x(self, x):
        _feat_dics = self._multinomial['feat_dics']
        for xx in x:
            idx = 0
            for d, discrete in enumerate(self._whether_discrete):
                # 如果是连续维度，直接调用float方法将其转化为浮点数
                if not discrete:
                    xx[d] = float(xx[d])
                # 如果是离散维度，利用数据字典进行数值化
                else:
                    xx[d] = _feat_dics[idx][xx[d]]
                if discrete:
                    idx += 1
        return x

    @staticmethod
    def plot_discrete_feature_pics(nb):
        # 进行一些设置，使得matplotlib可以显示中文
        # 将字体设为仿宋
        mpl.rcParams['font.sans-serif'] = ['FangSong']
        mpl.rcParams['axes.unicode_minus'] = False
        # 利用Multinomial搭建过程中记录的变量获取条件概率
        data = nb._multinomial['data']
        # 定义颜色字典，将类别yes（购买）设为天蓝色，类别no（不购买）设为橙色
        colors = {'yes\n': 'lightSkyBlue', 'no\n': 'orange'}
        # 利用转换字典定义其“反字典”，后面可视化会用上
        _rev_feat_dics = [{_val: _key for _key, _val in _feat_dics.items()} for _feat_dics in
                          nb._multinomial._feat_dics]
        # 遍历各维度进行可视化
        # 利用MultinomialNB搭建过程中记录的变量，获取画图需要的信息
        for _j in range(nb._multinomial._x[0].shape[1]):
            sj = nb._multinomial._n_possibilities[_j]
            tmp_x = np.arange(1, sj + 1)
            # 利用matplotlib对latex的支持来写标题，两个$之间即是latex语句
            # 标题为第几个feature(_j+1)，共有几种取值(sj)
            title = '$j = {};s_j = {}$'.format(_j + 1, sj)
            plt.figure()
            plt.title(title)
            # 根据条件概率的大小画出柱状图
            # 针对每个feature的每个category画图
            for _c in range(len(nb.label_dic)):
                plt.bar(tmp_x - 0.35 * _c, data[_j][_c, :], width=0.35, facecolor=colors[nb._multinomial.label_dic[_c]],
                        edgecolor='white', label='class:{}'.format(nb._multinomial.label_dic[_c]))
                # 利用上文的反字典将横坐标转化成特征的各个取值
                plt.xticks([i for i in range(sj + 2)], [''] + [_rev_feat_dics[_j][i] for i in range(sj)] + [''])
                plt.ylim(0, 1.0)
                plt.legend()
                # 保存画好的图像
                plt.savefig('bank_pics/d{}'.format(_j + 1))

    @staticmethod
    def plot_continuous_feature_pics(nb):
        labelled_x = nb._guassian._labelled_x
        n_category = len(nb._guassian.label_dic)
        labels = {0: 'yes', 1: 'no'}
        for dim in range(len(nb._guassian._x)):
            for c in range(n_category):
                # 计算mu ，平均值
                mu = np.sum(nb._guassian._labelled_x[c][dim]) / len(nb._guassian._labelled_x[c][dim])
                # 计算sigma 方差
                sigma = np.sum((nb._guassian._labelled_x[c][dim] - mu) ** 2) / len(nb._guassian._labelled_x[c][dim])
                xx = np.arange(mu - 3 * sigma, mu + 3 * sigma, 0.01 * sigma)
                # 要拿到mu和sigma以确定绘图区间，故无法直接使用函数
                # y1 = nb._guassian._data[dim][c](xx[1])
                yy = NBFunctions.gaussian_batch(xx, mu, sigma)
                # print(yy[1],y1)
                plt.plot(xx, yy, label='category = {}'.format(labels[c]))
                plt.title('$j = {};mu = {}; sigma = {}$'.format(dim, mu, sigma))
            plt.legend()
            plt.show()


if __name__ == '__main__':
    whether_continuous = [False] * 16
    continuous_lst = [0, 5, 9, 11, 12, 13, 14]
    for cl in continuous_lst:
        whether_continuous[cl] = True

    train_num = 40000
    data_time = time.time()
    (x_train, y_train), (x_test, y_test) = DataUtil.get_dataset(
        'bank1.0', 'bank1.0.txt', train_num=train_num)
    data_time = time.time() - data_time
    learning_time = time.time()
    nb = MergedNB(whether_continuous=whether_continuous)
    nb.fit(x_train, y_train)
    learning_time = time.time() - learning_time
    estimation_time = time.time()
    nb.evaluate(x_train, y_train)
    nb.evaluate(x_test, y_test)
    estimation_time = time.time() - estimation_time
    print(
        "Data cleaning   : {:12.6} s\n"
        "Model building  : {:12.6} s\n"
        "Estimation      : {:12.6} s\n"
        "Total           : {:12.6} s".format(
            data_time, learning_time, estimation_time,
            data_time + learning_time + estimation_time
        )
    )
    # MergedNB.plot_discrete_feature_pics(nb)
    MergedNB.plot_continuous_feature_pics(nb)
