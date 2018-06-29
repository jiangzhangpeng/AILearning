# encoding:utf-8
import sys

sys.path.append(r'D:\Workspaces\Github\AILearning\python-ml-in-practice')
from util.Bases import ClassifierBase
from chapter06.Layers import LayerFactory
import tensorflow as tf
import numpy as np


class NN(ClassifierBase):
    def __init__(self):
        super(NN, self).__init__()
        self.layers = []
        self._optimizer = None
        self._current_dimension = 0
        self._availabel_metrics = {
            key: value for key, value in zip(['acc', 'f1_score'], [NN.acc, NN.f1_score])
        }
        self.verbose = 0
        self._metrics, self._metrics_names, self._logs = [], [], {}
        self._layer_factory = LayerFactory()
        # 定义tensorflow中相应的变量
        self._tfx = self._tfy = None  # 记录每个batch的样本，标签属性
        self._tf_weights, self._tf_bias = [], []  # 记录w，b属性
        self._cost, self._y_pred = None  # 记录损失值，输出值属性
        self._train_step = None  # 记录参数更新步骤的属性
        self._sess = tf.Session()  # 记录tensorflow的session属性

    # 利用tensorflow相应的函数初始化参数
    @staticmethod
    def _get_w(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name='w')

    @staticmethod
    def _get_b(shape):
        return tf.Variable(np.zeros(shape, dtype=np.float32) + 0.1, name='b')

    #做一个初始化参数的封装，要注意兼容cnn
    def _add_params(self,shape,conv_channel=None,fc_shape=None,apply_bias = True):
        #如果是fc的话，根据平铺后的数据的形状来初始化数据
        if fc_shape is not None:
            w_shape = (fc_shape,shape[1])
            b_shape = shape[1]
        #如果是卷积曾的话，就要定义kernel而非权值矩阵
        elif conv_channel is not None:
            if len(shape[1])<=2:
                w_shape = shape[1][0],shape[1][1],conv_channel,conv_channel
            else:
                w_shape = (shape[1][1],shape[1][2],conv_channel,shape[1][0])
            b_shape = shape[1][0]
        #其余情况和nn无异
        else:
            w_shape = shape
            b_shape = shape[1]

        self._tf_weights.append(self._get_w(w_shape))
        if apply_bias:
            self._tf_bias.append(self._get_b(b_shape))
        else:
            self._tf_bias.append(None)
    #由于特殊层不会用到w和b，所以定义一个生成占位符的方法
    def _add_param_placeholder(self):
        self._tf_weights.append(tf.constant([0.0]))
        self._tf_bias.append(tf.constant([0.0]))
