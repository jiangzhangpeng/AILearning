# encoding:utf-8
import sys

sys.path.append(r'D:\Workspaces\Github\AILearning\python-ml-in-practice')
from util.Bases import ClassifierBase
import numpy as np
import matplotlib.pyplot as plt


class Perceptron(ClassifierBase):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.w = self.b = None

    def fit(self, x, y, sample_weight=None, lr=0.01, epoch=10 ** 3):
        x, y = np.atleast_2d(x), np.array(y)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.array(sample_weight) * len(y)

        # 初始化参数
        self._w = np.zeros(x.shape[1])
        self._b = 0
        for _ in range(epoch):
            y_pred = self.predict(x)
            # 获取加权误差向量

            _err = (y_pred != y) * sample_weight

            #书中的这部分感觉有问题
            # 引入随机性以进行随机梯度下降
            _indices = np.random.permutation(len(y))
            _idx = _indices[np.argmax(_err[_indices])]
            # 若没有被误分的，则完成训练
            if y_pred[_idx] == y[_idx]:
                #return
                continue

            #若无错误分类

            # 否则，根据选出的样本更新参数
            _idx = np.argmax(_err)
            _delta = lr * y[_idx] * sample_weight[_idx]
            self._w += _delta * x[_idx]
            self._b = _delta

    def predict(self, x, get_raw_results=False):
        rs = np.sum(self._w * x, axis=1) + self._b
        if not get_raw_results:
            return np.sign(rs)
        return rs


if __name__ == '__main__':
    x1 = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    x2 = [2, 3, 4, 5, 6, 0, 1, 2, 3, 4]
    y = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]

    x = [[x1[i], x2[i]] for i in range(len(x1))]
    print(x)
    plt.scatter(x1, x2)
    #plt.show()

    per = Perceptron()
    per.fit(x, y)
    print(per._w, per._b)
