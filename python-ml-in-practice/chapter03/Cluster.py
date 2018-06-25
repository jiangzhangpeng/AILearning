# encoding:utf-8
import math

import numpy as np


class Cluster:
    """
    初始化结构
    self._x, self._y 记录数据集的变量
    self._counters 类别计数器，记录第i类数据的个数
    self._sample_weight 记录样本权重的的属性
    self._con_chaos_cache self._ent_cache self._gini_cache 记录中间结果
    self._base 记录对数的底
    """

    def __init__(self, x, y, sample_weight=None, base=2):
        # 这里是我们要输入的向量（矩阵）
        self._x, self._y = x.T, y
        # 利用样本权重对类别向量y进行计数
        if sample_weight is None:
            self._counters = np.bincount(self._y)
        else:
            self._counters = np.bincount(self._y, weights=sample_weight * len(sample_weight))
        self._sample_weight = sample_weight
        self._con_chaos_cache = self._ent_cache = self._gini_cache = None
        self._base = base

    # 定义计算信息熵的函数
    def ent(self, ent=None, eps=1e-12):
        # 如果已经计算过且调用时没有给外给各类别的样本的个数，就直接调用结果
        if self._ent_cache is not None and ent is None:
            return self._ent_cache
        _len = len(self._y)
        # 如果调用时没有给出各个样本的个数，则利用结构本身的计数器来获取相应个数
        if ent is None:
            ent = self._counters
        # 使用eps来让算法的数值稳定性更好
        _ent_cache = max(eps, -sum([_c / _len * math.log(_c / _len, self._base) if _c != 0 else 0 for _c in ent]))
        # 如果调用时没有给各类别样本的个数，就把计算结果记录下来
        if ent is None:
            self._ent_cache = _ent_cache
        return _ent_cache

    # 定义计算H(y|A)和Gini(y|A)的函数
    def con_chaos(self, idx, criterion='ent', features=None):
        # 根据不同的准则，调用不同的方法
        if criterion == 'ent':
            _method = lambda cluster: cluster.ent()
        elif criterion == 'gini':
            _method = lambda cluster: cluster.gini()
        # 根据输入获取相应的维度向量
        data = self._x(idx)
        # 如果调用时没有给出该维度的取值空间features，则调用set方法获取该取值空间
        # 由于调用set方法比较耗时，在决策树实现时应努力将features传入
        if features is None:
            features = set(data)
        # 获得该维度特征值所对应的数据的下表
        # 调用self._con_chaos_catch记录下相应的结果以加速后面定义的相关函数
        tmp_labels = [data == feature for feature in features]
        self._con_chaos_cache = [np.sum(_label) for _label in tmp_labels]
        # 利用下标获取相应的类别向量
        label_list = [self._y[label] for label in tmp_labels]
        rs, chaos_lst = 0, []
        # 遍历各下标对应的类别向量
        for data_label, tar_label in zip(tmp_labels, label_list):
            # 获取对应的数据
            tmp_data = self._x.T[data_label]
            # 根据相应的数据、类别向量和样本权重计算出不确定性
            if self._sample_weight is None:
                _chaos = _method(Cluster(tmp_data, tar_label, base=self._base))
            else:
                _new_weights = self._sample_weight[data_label]
                _chaos = _method(Cluster(tmp_data, tar_label, _new_weights / np.sum(_new_weights), base=self._base))
                # 依概率加权，同时把各个初始条件不确定性记录下来
                rs += len(tmp_data) / len(data) * _chaos
                chaos_lst.append(_chaos)
        return rs, chaos_lst

    # 定义计算信息增益的函数 参数get_chaos_list用于控制输出
    def info_gain(self, idx, criterion='ent', get_chaos_list=False, features=None):
        # 根据不同的准则，获取相应的“条件不确定性”
        if criterion in ('ent', 'ratio'):
            _con_chaos, _chaos_list = self.con_chaos(idx, 'ent', features)
            _gain = self.ent() - _con_chaos
            if criterion == 'ratio':
                _gain /= self._con_chaos_cache
        elif criterion == 'gini':
            _con_chaos, _chaos_list = self.con_chaos(idx, 'gini', features)
            _gain = self.gini() - _con_chaos
        return (_gain, _chaos_list) if get_chaos_list else _gain

    # 定义计算二分类问题条件不确定性的函数
    # 参数tar即是二分标准，参数continuous则告诉我们该维度的特征是否连续
    def bin_con_chaos(self, idx, tar, criterion='gini', continuous=False):
        if criterion == 'ent':
            _method = lambda cluster: cluster.ent()
        elif criterion == 'gini':
            _method = lambda cluster: cluster.gini()
        data = self._x[idx]
        # 根据二分标准划分数据，注意要分离散和连续两种情况
        tar = data == tar if not continuous else data < tar
        tmp_labels = [tar, -tar]
        self._con_chaos_cache = [np.sum(_label) for label in tmp_labels]
        label_list = [self._y[label] for label in tmp_labels]
        rs, chaos_list = 0, []
        for data_label, tar_label in zip(tmp_labels, label_list):
            tmp_data = self._x.T[data_label]
            if self._sample_weight is None:
                _chaos = _method(Cluster(tmp_data, tar_label, base=self._base))
            else:
                _new_weights = self._sample_weight[data_label]
                _chaos = _method(Cluster(tmp_data, tar_label, _new_weights / np.sum(_new_weights), base=self._base))
            rs += len(tmp_data) / len(data) * _chaos
            chaos_list.append(_chaos)
        return rs, chaos_list


if __name__ == '__main__':
    pass
