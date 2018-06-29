# encoding:utf-8
import numpy as np


class Optimizer:
    """
    初始化结构
    self.lr 记录学习速率的属性  默认0.01
    self._cache 存储中间结果的参数 在不同的算法中表现会不同
    """

    def __init__(self, lr=0.01, cache=None):
        self.lr = lr
        self._cache = cache

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    # 接收欲更新的参数并进行相应处理，注意可能传入多个参数
    # 默认行为是创建若干个和传入的参数形状相等的0矩阵，并把他们存在self._cache中
    def feed_variables(self, variables):
        self._cache = [np.zeros(var.shape) for var in variables]

    # 利用负梯度dw=-delta w和优化器自身的属性来返回最终更新步伐的方法
    # 注意之而立的i是指优化器中的第i个参数
    def run(self, i, dw):
        pass

    # 完成参数更新后，更新自身属性的方法
    def update(self):
        pass


# Vanilla Update
class MBGD(Optimizer):
    def run(self, i, dw):
        return self.lr * dw


# Momentum Update
class Momentum(Optimizer):
    """
    初始化结构（Momentum update版本）
    self._momentum 记录惯性p的属性
    self._floor, self._ceiling 惯性的最小、最大值
    self._step 每一步迭代后惯性的增量
    self._cache对于momentum update而言，记录的就是行进速度
    self._is_nesterov 处理nesterov momentum update的属性

    """

    def __init__(self, lr=0.01, cache=None, epoch=100, floor=0.5, ceiling=0.999):
        Optimizer.__init__(self, lr, cache)
        self._momentum = floor
        self._step = (ceiling - floor) / epoch
        self._floor, self._ceiling = floor, ceiling
        self._is_nesterov = False

    def run(self, i, dw):
        dw *= self.lr
        velocity = self._cache
        velocity[i] *= self._momentum
        velocity[i] += dw
        # 如果不是nesterov momentum update
        if not self._is_nesterov:
            return velocity[i]
        # 否则，调用公式来计算更新步伐
        return self._momentum * velocity[i] + dw

    def update(self):
        if self._momentum < self._ceiling:
            self._momentum += self._step


class NAG(Momentum):
    def __init__(self, lr=0.01, cache=None, epoch=100, floor=0.5, ceiling=0.99):
        Momentum.__init__(self, lr, cache, epoch, floor, ceiling)
        self._is_nesterov = True


class RMSProp(Optimizer):
    """
    初始化结构（RMSProp版本）
    self.decay_rate 记录梯度衰减值，一般去0.9,0.99,0.999
    self.eps 算法平滑项，增加算法的稳定性
    self._cache 对于rmsprop来讲，记录中间变量（累积梯度delta）
    """

    def __init__(self, lr=0.01, cache=None, decay_rate=0.9, eps=1e-8):
        Optimizer.__init__(self, lr, cache)
        self.decay_rate, self.eps = decay_rate, eps

    def run(self, i, dw):
        self._cache[i] = self._cache[i] * self.decay_rate + (1 - self.decay_rate) * dw ** 2
        return self.lr * dw / (np.sqrt(self._cache[i] + self.eps))


class Adam(Optimizer):
    """
    初始化结构（Adam版本）
    self.beta1, self.beta2 存储beta1，beta2，一般取beta1为0.9，beta2为0.999
    self._cache 对于adam而言，该属性记录中间变量
    """

    def __init__(self, lr=0.01, cache=None, beta1=0.9, beta2=0.999, eps=1e-8):
        Optimizer.__init__(self, lr, cache)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

    def feed_variables(self, variables):
        self._cache = [[np.zeros(var.shape) for var in variables], [np.zeros(var.shape) for var in variables]]

    def run(self, i, dw):
        self._cache[0][i] = self._cache[0][i] * self.beta1 + (1 - self.beta1) * dw
        self._cache[1][i] = self._cache[1][i] * self.beta2 + (1 - self.beta2) * (dw ** 2)
        return self.lr * self._cache[0][i] / (np.sqrt(self._cache[1][i] + self.eps))

class OptFactory:
    #将所有能用的优化器存进一个词典
    avilable_optimizers = {
        'MBGD':MBGD,
        'Momentum':Momentum,
        'NAG':NAG,
        'RMSProp':RMSProp,
        'Adam':Adam
    }

    #定义一个通过优化器名字获取优化器的方法
    def get_optimizer_by_name(self,name,variables,lr,epoch):
        try:
            _optimizer = self.avilable_optimizers[name](lr)
            if variables is not None:
                _optimizer.feed_variables(variables)
            if epoch is not None and isinstance(_optimizer,Momentum):
                _optimizer.epoch = epoch
            return _optimizer
        except KeyError:
            raise NotImplementedError('Undefined optimizer {} found!'.format(name))

