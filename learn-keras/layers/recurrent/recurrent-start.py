# encoding:utf-8
import keras
import keras as K
from keras.layers import RNN


# 定义一个RNN单元在，作为网络层子类
class MinimalRNNCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer='uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), initializer='uniform', name='kernel')
        self.build = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]


def test1():
    # 在RNN层使用MinimalRNNCell
    cell = MinimalRNNCell(32)
    x = keras.Input((None, 5))
    layer = RNN(cell)
    y = layer(x)

    # 以下是使用单元格构建堆叠的RNN
    cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
    x = keras.Input((None, 5))
    layer = RNN(cells)
    y = layer(x)
