# encoding:utf-8
import numpy as np


class Layer:
    """
    初始化结构
    """

    def __init__(self, shape):
        self.shape = shape

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return str(self)

    def _activate(self, x):
        pass

    # 激活函数的导函数的定义留给子类定义
    def derivative(self, y):
        pass

    # 前向传到算法封装
    def activate(self, x, w, bias):
        return self._activate(x.dot(w) + bias)

    # 反向传播算法的封装，主要是利用上面定义的导函数derivative来完成局部梯度的计算
    def bp(self, y, w, prev_delta):
        return prev_delta.dot(w.T) * self.derivative(y)


class Sigmoid(Layer):
    def _activate(self, x):
        return 1 / (1 + np.exp(-x))

    # fi(x) = 1/(1+e**(-x))   fi'(x) = fi(x)*(1-fi(x))
    def derivative(self, y):
        return y * (1 - y)


class CostLayer(Layer):
    """
    初始化结构
     self._avilable_cost_functions 记录所有损失函数的字典
    self._avilable_transform_functions 记录所有特殊变换函数的字典
    self._cost_function_name,self._cost_function 记录损失函数及其名字的两个属性
    self._transform,self._transform_function 记录特殊变换函数及其名字的两个属性
    """

    def __init__(self, shape, cost_function='MSE', transform=None):
        super(CostLayer, self).__init__(shape)
        self._avilable_cost_functions = {'MSE': CostLayer._mse, 'SVM': CostLayer._svm,
                                         'CrossEntropy': CostLayer._cross_entropy}
        self._avilable_transform_functions = {'Softmax': CostLayer._softmax, 'Sigmoid': CostLayer._sigmoid}
        self._cost_function_name = cost_function
        self._cost_function = self._avilable_cost_functions[cost_function]
        # transform哪里来的？
        if transform is None and cost_function == 'CrossEntropy':
            self._transform = 'Softmax'
            self._transform_function = CostLayer._softmax
        else:
            self._transform = transform
            self._transform_function = self._avilable_transform_functions.get(transform, None)

    def __str__(self):
        return self._cost_function_name

    def _activate(self, x, predict):
        # 如果不是用特殊变换函数的话，直接返回输入值即可
        if self._transform_function is None:
            return x
        # 否则，调用变换函数以获得结果
        return self._transform_function(x)

    # 由于CostLayer有自己特殊的BP算法，所以这个方法不会调用，自然也无需定义
    def _derivative(self, y, delta=None):
        pass

    @staticmethod
    def safe_exp(x):
        return np.exp(x - np.max(x, axis=1, keepdims=True))

    @staticmethod
    def _softmax(y, diff=False):
        if diff:
            return y * (1 - y)
        exp_y = CostLayer.safe_exp(y)
        return exp_y / np.sum(exp_y, axis=1, keepdims=True)

    @staticmethod
    def _sigmoid(y, diff=False):
        if diff:
            return y * (1 - y)
        return 1 / (1 + np.exp(y))

    # 定义计算整合梯度的方法，注意这里返回的是负梯度
    def bp_first(self, y, y_pred):
        # 如果是sigmoid  softmax或者crossentropy 采用delta**(m) = v**(m) -y进行优化
        if self._cost_function_name == 'CrossEntropy' and (
                self._transform == 'Softmax' or self._transform == 'Sigmoid'):
            return y - y_pred
        # 否则，就只能用普适公式进行计算
        dy = -self._cost_function(y, y_pred)
        if self._transform_function is None:
            return dy
        return dy * self._transform_function(y_pred, diff=True)

    # 定义损失的计算方法
    @property
    def calculate(self):
        return lambda y, y_pred: self._cost_function(y, y_pred, False)

    # 定义距离损失函数及其导数
    @staticmethod
    def _mse(y, y_pred, diff=True):
        if diff:
            return -y + y_pred
        return 0.5 * np.average((y - y_pred) ** 2)

    # 定义crossentropy损失函数及其导数
    @staticmethod
    def _cross_entropy(y, y_pred, diff=True, eps=1e-8):
        if diff:
            return -y / (y_pred + eps) + (1 - y) / (1 - y_pred + eps)
        return np.average(-y * np.log(y_pred + eps) - (1 - y) * np.log(1 - y_pred + eps))
