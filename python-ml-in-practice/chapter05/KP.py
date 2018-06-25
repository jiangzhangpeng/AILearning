# encoding:utf-8
import sys

import numpy as np

sys.path.append(r'D:\Workspaces\Github\AILearning\python-ml-in-practice\util')
from Bases import KernelBase


class KernelPerceptron(KernelBase):
    def __init__(self):
        KernelBase.__init__(self)
        # 对于和感知机而言，循环体重所需的额外参数是学习速率，默认为1
        self._fit_args, self._fit_args_names = [1], ['lr']

    # 更新dw
    def _update_dw_cache(self, idx, lr, sample_weight):
        self._dw_cache = lr * self._y[idx] * sample_weight[idx]

    # 更新db
    def _update_db_cache(self, idx, lr, sample_weight):
        self._db_cache = self._dw_cache
        # self._db_cache = lr * self._y[idx] * sample_weight[idx]

    # 利用alpha和训练样本中的类别向量y来更新w和b
    def _update_params(self):
        self._w = self._alpha * self._y
        self._b = np.sum(self._w)

    def _fit(self,sample_weight,lr):
        #获取加权误差向量
        _err = (np.sign(self._prediction_chache)!= self._y)*sample_weight
        #引入随机性以进行随机梯度下降
        _indices = np.random.permutation(len(self._y))
        #获取错的最严重的样本所对应的下标
        _idx = _indices[np.argmax(_err[_indices])]
        #若该样本被正确分类，则所有样本都已经正确分类，此时返回真值，退出循环体
        if self._prediction_cache[_idx] == self._y[_idx]:
            return True
        #否则，进行随机梯度下降算法
        self._alpha[_idx] += lr
        self._update_dw_cache(_idx,lr,sample_weight)
        self._update_db_cache(_idx,lr,sample_weight)
        self._update_pred_cache(_idx)
        
