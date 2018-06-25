# encoding:utf-8
import sys

import numpy as np

sys.path.append(r'D:\Workspaces\Github\AILearning\python-ml-in-practice\util')
from Bases import KernelBase


class SVM(KernelBase):
    def __init__(self):
        KernelBase.__init__(self)
        # 对于核SVM而言，循环体重所需的额外的参数是容许误差e（默认10**-3）
        self._fit_args, self._fit_args_names = [1e-3], ['tol']
        self._c = None

    # 实现SMO算法中，挑选第一个变量的方法
    def _pick_first(self, tol):
        con1 = self._alpha > 0
        con2 = self._alpha <= self._c
        # 算出损失向量并拷贝成三分
        err1 = self._y * self._prediction_cache - 1
        err2 = err1.copy()
        err3 = err1.copy()
        # 将对应的数位置为0
        err1[con1 | (err1 >= 0)] = 0
        err2[(~con1 | ~con2) | (err2 == 0)] = 0
        err3[con2 | (err3 <= 0)] = 0
        # 算出总的损失向量并取出最大的一项
        err = err1 ** 2 + err2 ** 2 + err3 ** 2
        idx = np.argmax(err)
        # 若该项的损失小于e则返回空值
        if err[idx] < tol:
            return
            # 否则 返回对应下表
        return idx

    # 实现SMO算法中，跳出第二个变量的方法（事实上是随机挑选）
    def _pick_second(self, idx1):
        idx = np.random.randint(len(self._y))
        while idx == idx1:
            idx = np.random.randint(len(self._y))
        return idx

    # 获取新的alpha2的下界
    def _get_lower_bound(self, idx1, idx2):
        if self._y[idx1] != self._y[idx2]:
            return max(0., self._alpha[idx2] - self._alpha[idx1])
        return max(0., self._alpha[idx2] + self._alpha[idx1] - self._c)

    # 获取alpha2的上界
    def _get_upper_bound(self, idx1, idx2):
        if self._y[idx1] != self._y[idx2]:
            return min(self._c, self._c + self._alpha[idx2] - self._alpha[idx1])
        return min(self._c, self._alpha[idx2] + self._alpha[idx1])

    # 更新dw
    def _update_dw_cache(self, idx1, idx2, da1, da2, y1, y2, e1, e2):
        self._dw_cache = np.array([da1 * y1, da2 * y2])

    # 更新db
    def _update_db_cache(self, idx1, idx2, da1, da2, y1, y2, e1, e2):
        gram_12 = self._gram[idx1][idx2]
        b1 = -e1 - y1 * self._gram[idx1][idx1] * da1 - y1 * gram_12 * da2
        b2 = -e2 - y1 * gram_12 * da1 - y2 * self._gram[idx2][idx2] * da2
        self._db_cache = (b1 + b2) * 0.5

    # 利用alpha和蓄念样本中的类别向量y来更新w和b
    def _update_params(self):
        self._w = self._alpha * self._y
        _idx = np.argmax((self._alpha != 0) & (self._alpha != self._c))
        self._b = self._y[_idx] - np.sum(self._alpha * self._y * self._gram[_idx])

    # 定义局部更新alpha的方法
    def _update_alpha(self, idx1, idx2):
        l, h = self._get_lower_bound(idx1, idx2), self._get_upper_bound(idx1, idx2)
        y1, y2 = self._y[idx1], self._y[idx2]
        e1 = self._prediction_cache[idx1] - self._y[idx1]
        e2 = self._prediction_cache[idx2] - self._y[idx2]
        eta = self._gram[idx1][idx1] + self._gram[idx2][idx2] - 2 * self._gram[idx1][idx2]
        a2_new = self._alpha[idx2] + (y2 * (e1 - e2)) / eta
        if a2_new > h:
            a2_new = h
        elif a2_new < l:
            a2_new = l
        a1_old, a2_old = self._alpha[idx1], self._alpha[idx2]
        da2 = a2_new - a2_old
        da1 = -y1 * y2 * da2
        self._alpha[idx1] += da1
        self._alpha[idx2] = a2_new
        # 根据delta alpha1(da1) 、deltal alpha2(da2)来更新dw和db并局部更新yhat
        self._update_dw_cache(idx1, idx2, da1, da2, y1, y2)
        self._update_db_cache(idx1, idx2, da1, da2, y1, y2, e1, e2)
        self._update_pred_cache(idx1, idx2)

    # 初始化惩罚因子C
    def _prepare(self, **kwargs):
        self._c = kwargs.get('c', KernelConfig.default_c)

    def _fit(self, sample_weight, tol):
        idx1 = self._pick_first(tol)
        # 若没有能选出第一个变量，则所有样本的误差都小于e，此时返回真值，退出训练循环体
        if idx1 is None:
            return True
        idx2 = self._pick_second(idx1)
        self._update_alpha(idx1, idx2)
