# encoding:utf-8
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, Input, Reshape, Permute, RepeatVector, Lambda
from keras.models import Sequential, Model
import keras as K


# Flatten
def test1():
    model = Sequential()
    # 此处keras2和1不同  参数方式不一样
    model.add(Conv2D(64, (3, 3), border_mode='same', input_shape=(32, 32, 3)))
    print(model.output_shape)
    model.add(Flatten())
    print(model.output_shape)


# Input
def test2():
    x = Input(shape=(32,))
    y = Dense(16, activation='relu')(x)
    model = Model(inputs=x, outputs=y)


# Reshape
def test3():
    model = Sequential()
    model.add(Reshape((4, 3), input_shape=(12,)))
    print(model.input_shape)
    print(model.output_shape)

    model.add(Reshape((6, 2)))
    print(model.output_shape)

    model.add(Reshape((-1, 2, 3)))
    print(model.output_shape)


# Permute
def test4():
    model = Sequential()
    # permute的turple中的维度是input_shape的turple索引，从1开始
    model.add(Permute((1, 3, 2), input_shape=(2, 4, 8)))
    print(model.input_shape)
    print(model.output_shape)


# RepeatVector
def test5():
    model = Sequential()
    model.add(Dense(32, input_shape=(2,)))
    print(model.output_shape)
    model.add(RepeatVector(3))
    print(model.output_shape)


# Lambda  对数据进行一些预处理
def test6():
    model = Sequential()
    model.add(Lambda(lambda x: x ** 2))

    # 添加一个网络层，返回输入的正数部分
    # 与负数部分的反面的连接

    def antirectifier(x):
        x -= K.mean(x, axis=1, keepdims=True)
        x = K.l2_normalize(x, axis=1)
        pos = K.relu(x)
        neg = K.relu(-x)
        return K.concatenate([pos, neg], axis=1)

    def antirectifier_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] *= 2
        return tuple(shape)

    model.add(Lambda(antirectifier,
                     output_shape=antirectifier_output_shape))


if __name__ == '__main__':
    test1()
