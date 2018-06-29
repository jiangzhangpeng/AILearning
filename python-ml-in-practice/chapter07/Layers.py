# encoding:utf-8
from math import ceil

import numpy as np
import tensorflow as tf


class Layer:
    """
    初始化结构
    self.shape 记录该layer和上个layer神经元的个数，具体而言，shape[0]为上个layer所含神经元个数，
    shape[1]为该layer所含神经元个数
    self.is_fc， self.is_sub_layer 记录该layer是否为fc、特殊层结构的属性
    self.apply_bias 记录是否对该layer加偏置量的属性
    """

    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.is_fc = self.is_sub_layer = False
        self.apply_bias = kwargs.get('apply_bias', True)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return str(self)

    @property
    def root(self):
        return self

    # 定义兼容特殊层结构和cnn的、前向传到算法的封装
    def activate(self, x, w, bias=None, predict=False):
        # 如果当前层是fc，先将输入铺平
        if self.is_fc:
            x = tf.reshape(x, [-1, int(np.prod(x.get_shape()[1:]))])
        # 如果是特殊层结构，就调用相应的方法获取结果
        if self.is_sub_layer:
            return self._activate(x, predict)
        # 如果不加偏置量的话，就执行性矩阵相乘和激活函数作用
        if not self.apply_bias:
            return self._activate(tf.matmul(x, w), predict)
        # 否则，进行正常的前向传到算法
        return self._activate(tf.matmul(x, w) + bias, predict)

    # 前向传到算法核心，留待子类定义
    def _activate(self, x, predict):
        pass


class Sigmoid(Layer):
    def _activate(self, x, predict):
        return tf.nn.sigmoid(x)


class ReLU(Layer):
    def _activate(self, x, predict):
        return tf.nn.relu(x)


# SubLayer 继承 Layer以合理复用代码
class SubLayer(Layer):
    """
    初始化结构
    self.parent 记录该layer的父层的属性
    self.description 用于可视化的属性，记录着该sublayer的描述
    """

    def __init__(self, parent, shape):
        Layer.__init__(self, shape)
        self.parent = parent
        self.description = ''

    # 复制获取root layer的property
    # 递归获取到root
    def root(self):
        _root = self.parent
        while _root.parent:
            _root = _root.parent
        return _root

    @property
    def info(self):
        return 'Layer:{:<16s}-{}{}'.format(self.name, self.shape[1], self.description)


class Dropout(SubLayer):
    # self._prob 训练过程中神经元被留下的概率
    def __init__(self, parent, shape, drop_prob=0.5):
        # 神经元被drop的概率必须大于0小于1
        if drop_prob < 0 or drop_prob >= 1:
            raise ValueError('dropout probability error!')
        SubLayer.__init__(self, parent, shape)
        # 被留下的概率是1-drop的概率
        self._prob = tf.constant(1 - drop_prob, dtype=tf.float32)
        self.description = '(drop prob:{})'.format(drop_prob)

    def _activate(self, x, predict):
        # 如果是训练过程，那么就按照设定的，被留下的概率进行dropout
        if not predict:
            return tf.nn.dropout(x, self._prob)
        # 如果是预测过程，则直接返回输入值
        return x


class Normalize(SubLayer):
    """
    初始化结构
    self._eps 记录增强数值稳定性所用的最小值的属性
    self._activation 记录自身的激活函数的属性
    self._momentum 记录动量值m的属性
    """

    def __init__(self, parent, shape, activation='Identical', eps=1e-8, momentum=0.9):
        SubLayer.__init__(self, parent, shape)
        self._eps, self._activation = eps,
        self.tf_rm = self.tf_rv = None
        self.tf_gamma = tf.Variable(tf.ones(self.shape[1]), name='norm_scale')
        self.tf_beta = tf.Variable(tf.zeros(self.shape[1]), name='norm_beta')
        self._momentum = momentum
        self.description = '(eps:{},momentum:{})'.format(eps, momentum)

    def _activate(self, x, predict):
        # 如果均值、方差尚未进行初始化，则根据x输入进行初始化
        if self.tf_rm is None or self.tf_rv is None:
            shape = x.get_shape()[-1]
            self.tf_rm = tf.Variable(tf.zeros(shape), trainable=False, name='norm_mean')
            self.tf_rv = tf.Variable(tf.ones(shape), trainable=False, name='norm_var')

        if not predict:
            # 利用tensorflow相应函数计算当前批次的均值、方差
            _sm, _sv = tf.nn.moments(x, list(range(len(x.get_shape()) - 1)))
            _rm = tf.assign(self.tf_rm, self._momentum * self.tf_rm + (1 - self._momentum) * _sm)
            _rv = tf.assign(self.tf_rv, self._momentum * self.tf_rv + (1 - self._momentum) * _sv)
            # 利用tensorflow的相应函数直接得到batch normalization的结果
            with tf.control_dependencies([_rm, _rv]):
                _norm = tf.nn.batch_normalization(x, _sm, _sv, self.tf_beta, self.tf_gamma, self._eps)
        else:
            _norm = tf.nn.batch_normalization(x, self.tf_rm, self.tf_rv, self.tf_beta, self.tf_gamma, self._eps)

        # 如果指定了激活函数，就在用相应的激活函数作用在BN结果上以得到最终结果
        # 这里只定义了relu和sigmoid两种，如有其它进行拓展
        if self._activation == 'ReLU':
            return tf.nn.relu(_norm)
        elif self._activation == 'Sigmoid':
            return tf.nn.sigmoid(_norm)

        return _norm


# 重新定义costlayer
class CostLayer(Layer):
    # 定义一个方法以获取损失值
    def calculate(self, y, y_pred):
        return self._activate(y_pred, y)


# 定义crossentropy对应的costlayer  整合了softmax变换
class CrossEntropy(CostLayer):
    def _activate(self, x, y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y))


# 定义MSE对应的costlayer
class MSE(CostLayer):
    def _activate(self, x, y):
        return tf.reduce_mean(tf.square(x - y))


class ConvLayer(Layer):
    """
    初始化结构
    self.shape 记录着上个卷积层的输出和该layer的kernel的信息
    shape[0]为上个卷积层输出的形状，shape[1]为该卷积层的kernel信息
    """

    def __init__(self, shape, stride=1, padding='SAME', parent=None):
        if parent is not None:
            _parent = parent.root if parent.is_sub_layer else parent
            shape = _parent.shape
        Layer.__init__(self, shape)
        self.stride = stride
        # 通过tensorflow对padding的封装，定义padding信息
        if isinstance(padding, str):
            # 'VALID'意味着输出的高、宽受kernel高、宽影像
            if padding.upper() == 'VALID':
                self.padding = 0
                self.pad_flag = 'VALID'
            # 'SAME'意味着输出的高宽与kernel的高宽无关，只受stride影像
            else:
                self.padding = self.pad_flag = 'SAME'
        # 如果输入的一个整数，那么就按照VALID的情形设置padding相关属性
        else:
            self.padding = int(padding)
            self.pad_flag = 'VALID'

        self.parent = parent
        if len(shape) == 1:
            self.n_channels = self.n_filters = self.out_h = self.out_w = None
        else:
            self.feed_shape(shape)

    # 定义一个处理shape属性的方法
    def feed_shape(self, shape):
        self.shape = shape
        self.n_channels, height, width = shape[0]
        self.n_filters
        filter_height, filter_width = shape[1]
        # 根据padding的相关信息，计算输出的高、宽
        if self.pad_flag == 'VALID':
            self.out_h = ceil((height - filter_height + 1) / self.stride)
            self.out_w = ceil((width - filter_width + 1) / self.stride)
        else:
            self.out_h = ceil(height / self.stride)
            self.out_w = ceil(width / self.stride)


class ConvLayerMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        # 规定继承的顺序为convlayer -> layer
        conv_layer, layer = bases

        def __init__(self, shape, stride=1, padding='SAME'):
            conv_layer.__init__(self, shape, stride, padding)

        # 利用tensorflow的函数，定义计算卷积的方法
        def _conv(self, x, w):
            return tf.nn.con2d(x, w, strides=[self.stride] * 4, padding=self.pad_flag)

        # 以此进行卷积，激活的步骤
        def _activate(self, x, w, bias, predict):
            res = self._conv(x, w) + bias
            return layer._activate(self, res, predict)

        # 在正式进行前向传到算法之前，先用tensorflow的函数进行padding
        def activate(self, x, w, bias=None, predict=False):
            if self.pad_flag == 'VALID' and self.padding > 0:
                _pad = [self.padding] * 2
                x = tf.pad(x, [[0, 0], _pad, _pad, [0, 0]], 'CONSTANT')
            return _activate(self, x, w, bias, predict)

        # 将打包好的类返回
        for key, value in locals().items():
            if str(value).find('function') >= 0:
                attr[key] = value
        return type(name, bases, attr)


class ConvReLU(ConvLayer, ReLU, metaclass=ConvLayerMeta):
    pass


class ConvPoolLayer(ConvLayer):
    def feed_shape(self, shape):
        shape = (shape[0], (shape[0][0], *shape[1]))
        ConvLayer.feed_shape(self, shape)

    def activate(self, x, w, bias=None, predict=False):
        pool_height, pool_width = self.shape[1][1:]
        # 处理padding
        if self.pad_flag == 'VALID' and self.padding > 0:
            _pad = [self.padding] * 2
            x = tf.pad(x, [[0, 0], _pad, _pad, [0, 0]], 'CONSTANT')
            # 利用self._activate方法进行池化
            return self._activate(None)(x, ksize=[1, pool_height, pool_width, 1],
                                        strides=[1, self.stride, self.stride, 1],
                                        padding=self.pad_flag)

    def _activate(self, x, *args):
        pass


# 实现极大池化
class MaxPool(ConvPoolLayer):
    def _activate(self, x, *args):
        return tf.nn.max_pool


# 实现平均池化
class AvgPool(ConvPoolLayer):
    def _activate(self, x, *args):
        return tf.nn.avg_pool


# 定义作为封装的元类
class ConvSubLayerMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        conv_layer, sub_layer = bases

        def __init__(self, parent, shape, *_args, **_kwargs):
            conv_layer.__init__(self, None, parent=parent)
            # 与池化层类似，特殊层输出数据的形状应该保持与输入数据形状一致
            self.out_h, self.out_w = parent.out_h, parent.out_w
            sub_layer.__init__(self, parent, shape, *_args, **_kwargs)
            self.shape = ((shape[0][0], self.out_h, self.out_w), shape[0])
            # 如果是cnn的normalize，则提前初始化gamma，beta
            if name == 'ConvNorm':
                self.tf_gamma = tf.Variable(tf.ones(self.n_filters), name='norm_scale')
                self.tf_beta = tf.Variable(tf.zeros(self.n_filters), name='norm_beta')

        # 利用nn中特殊层结构的相应方法获取结果
        def _activate(self, x, predict):
            return sub_layer._activate(self, x, predict)

        def activate(self, x, w, bias=None, predict=False):
            return _activate(self, x, predict)

        # 将打包好的类返回
        for key, value in locals().items():
            if str(value).find('function') >= 0 or str(value).find('property'):
                attr[key] = value
        return type(name, bases, attr)


# 定义cnn中的dropout，注意继承顺序
class ConvDrop(ConvLayer, Dropout, metaclass=ConvSubLayerMeta):
    pass


# 定义cnn中的normalize，注意继承顺序
class ConvNorm(ConvLayer, Normalize, metaclass=ConvSubLayerMeta):
    pass


class LayerFactory:
    # 使用一个字典记录所有的root layer
    available_root_layers = {
        'Sigmoid': Sigmoid, 'ReLU': ReLU, 'CrossEntropy': CrossEntropy,
        'MSE': MSE, 'MaxPool': MaxPool, 'AvgPool': AvgPool
    }
    # 使用一个字典记录下所有的特殊层
    availabel_special_layers = {
        'Dropout': Dropout, 'Normalize': Normalize,
        'ConvDrop': ConvDrop, 'ConvNorm': ConvNorm
    }
    # 使用一个字典记录下所有特殊层的默认参数
    special_layer_default_params = {
        'Dropout': (0.5,),
        'Normalize': ('Identical', 1e-8, 0.9),
        'ConvDrop': (0.5,),
        'ConvNorm': ('Identical', 1e-8, 0.9)
    }

    # 定义根据名字获取layer的方法
    def get_root_layer(self, name, *args, **kwargs):
        # 根据字典判断输入名字是否存在(root layer)
        if name in self.available_root_layers:
            # 若是，则返回相应的root layer
            layer = self.available_root_layers[name]
            return layer(*args, **kwargs)
        # 否则返回none
        return None

    # 根据名字获取热河layer的方法，包括（root special）
    def get_layer_by_name(self, name, parent, current_dimension, *args, **kwargs):
        # 先看看是否存在root layer
        _layer = self.get_root_layer(name, *args, **kwargs)
        # 若是，则返回相应的root layer
        if _layer:
            return _layer, None
        # 否则，就根据附赠和行相应字典进行初始化后，返回特殊层
        _current, _next = parent.shape[1], current_dimension
        layer_param = self.special_layer_default_params[name]
        _layer = self.availabel_special_layers[name]
        if args or kwargs:
            _layer = _layer(parent, (_current, _next), *args, **kwargs)
        else:
            _layer = _layer(parent, (_current, _next), *layer_param)
        return _layer, (_current, _next)


if __name__ == '__main__':
    x = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]
    print(np.shape(x))
    # x = tf.reshape(x, [-1, int(np.prod(x.get_shape()[1:]))])
    print(np.shape(x))
