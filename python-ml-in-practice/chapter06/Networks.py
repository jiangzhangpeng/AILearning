# encoding:utf-8
import sys

sys.path.append(r'D:\Workspaces\Github\AILearning\python-ml-in-practice')
from util.Bases import ClassifierBase
from Layers import *
from Optimizers import *

import numpy as np


class NaiveNN(ClassifierBase):
    """
    初始化结构
    self._layers, self._weights, self._bias 记录所有layer，权值矩阵和偏置量
    self._w_optimizer， self._b_optimazer 权值矩阵和偏置量优化器
    self._current_dimension 记录着当前最后一个layer所含的神经元个数
    """

    def __init__(self):
        super(NaiveNN, self).__init__()
        self._layers, self._weights, self._bias = [], [], []
        self._w_optimizer = self._b_optimazer = None
        self._current_dimension = 0

    def add(self, layer):
        if not self._layers:
            # 如果是第一次加入layer，进行初始化
            self._layers, self._current_dimension = [layer], layer.shape[1]
            # 调用初始化权值矩阵和偏置量的方法
            self._add_params(layer.shape)
        else:
            _next = layer.shape[0]
            layer.shape = (self._current_dimension, _next)
            # 调用进一步处理layer的方法
            self._add_layer(layer, self._current_dimension, _next)

    def _add_layer(self, layer, *args):
        _current, _next = args
        self._add_params((_current, _next))
        self._current_dimension = _next
        self._layers.append(layer)

    def _add_params(self, shape):
        self._weights.append(np.random.randn(*shape))
        self._bias.append(np.zeros(1, shape[1]))

    def fit(self, x, y, lr=0.001, optimizer='Adam', epoch=10):
        # 调用相应的方法来初始化优化器
        self._init_optimizer(optimizer, lr, epoch)
        layer_width = len(self._layers)
        # 训练的主循环
        # 需要注意的是，在每次迭代中我们是用训练集中所有的样本进行训练的
        for counter in range(epoch):
            self._w_optimizer.update()
            self._b_optimazer.update()
            # 调用相应的方法来进行前向传播算法，把所得到的激活值存储下来
            _activations = self._get_activations(x)
            # 调用costlayer的bp_first方法来进行bp算法的第一步
            _deltas = [self._layers[-1].bp_first(y, _activations)]
            # BP算法主体
            for i in range(-1, -len(_activations), -1):
                _deltas.append(self._layers[i - 1].bp(_activations[i - 1], self._weights[i], _deltas[-1]))
            # 利用各个局部梯度来更新模型参数
            # 注意由于最后一个是costlayer对应的占位符，所以无需进行更新
            for i in range(layer_width - 1, 0, -1):
                self._opt(i, _activations[i - 1], _deltas[layer_width - i - 1])
            self._opt(0, x, _deltas[-1])

    def predict(self, x, get_raw_results=False):
        y_pred = self._get_prediction(np.atleast_2d(x))
        if get_raw_results:
            return y_pred
        return np.argmax(y_pred, axis=1)

    def _get_prediction(self, x):
        # 直接去前向传到算法得到的最后一个激活值
        return self._get_activations(x)[-1]

    def _init_optimizer(self, optimizer, lr, epoch):
        # 利用定义好的优化器工厂来初始化优化器
        # 注意由于最后一层是costlayer对应的占位符，所以无需把它输入优化器
        _opt_fac = OptFactory()
        self._w_optimizer = _opt_fac.get_optimizer_by_name(optimizer, self._weights[:-1], lr, epoch)
        self._b_optimazer = _opt_fac.get_optimizer_by_name(optimizer, self._bias[:-1], lr, epoch)

    def _get_activations(self, x):
        _activations = [self._layers[0].activate(x, self._weights[0], self._bias[0])]  # 第一层
        for i, layer in enumerate(self._layers[1:]):  # 第二层到最后一层
            _activations.append(layer.activate(_activations[-1], self._weights[i + 1], self._bias[i + 1]))
        return _activations

    def _opt(self, i, _activation, _delta):
        self._weights[i] += self._w_optimizer.run(i, _activation.T.dot(_delta))
        self._bias[i] += self._b_optimazer.run(i, np.sum(_delta, axis=0, keepdims=True))
