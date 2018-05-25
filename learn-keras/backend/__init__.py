# encoding:utf-8
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input

sess = tf.Session()
K.set_session(sess)


def test1():
    input = K.placeholder(shape=(2, 4, 5))
    print(input.shape)
    input = K.placeholder(shape=(None, 4, 5))
    print(input.shape)
    input = K.placeholder(ndim=3)
    print(input.shape)


def test2():
    val = np.random.random((3, 4, 5))
    var = K.variable(value=val)
    print(var)
    var = K.zeros(shape=(3, 4, 5))
    print(var)
    var = K.ones(shape=(3, 4, 5))
    print(var)


def test3():
    b = K.random_uniform_variable(shape=(3, 4), low=0, high=1)
    print(K.eval(b))
    c = K.random_normal_variable(shape=(3, 4), mean=0, scale=1)
    print(K.eval(c))
    d = K.random_normal_variable(shape=(3, 4), mean=0, scale=1)
    print(K.eval(d))

    a = b + c * K.abs(d)
    print(K.eval(a))


def test4():
    print(K.epsilon())
    K.set_epsilon(1e-8)
    print(K.epsilon())


def test5():
    print(K.floatx())
    K.set_floatx('float64')
    print(K.floatx())
    var = np.array([1.0, 2.0], dtype='float32')
    print(var.dtype)
    var = K.cast_to_floatx(var)
    print(var.dtype)


def test6():
    print(K.image_data_format())
    K.set_image_data_format('channels_first')
    print(K.image_data_format())


def test7():
    print(K.get_uid(prefix='T'))


def test8():
    val = np.array([[1, 2], [3, 4]])
    kvar = K.variable(val, dtype='float64', name='sample_variable')
    print(K.dtype(kvar))
    print(kvar)


def test9():
    np_var = np.array([1, 2])
    # print(K.is_keras_tensor(np_var))  # 一个 Numpy 数组不是一个符号张量。 不是符号张量，报错
    k_var = tf.placeholder('float32', shape=(1, 1))
    print(K.is_keras_tensor(k_var))  # 在 Keras 之外间接创建的变量不是 Keras 张量。
    keras_input = Input([10])
    keras_layer_output = Dense(10)(keras_input)
    print(K.is_keras_tensor(keras_layer_output))  # 任何 Keras 层输出都是 Keras 张量。


def test10():
    kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
    print(K.eval(kvar))


def test11():
    var = K.eye(4)
    print(K.eval(var))
    var1 = K.zeros_like(var)
    print(K.eval(var1))
    var2 = K.ones_like(var)
    print(K.eval(var2))


def test12():
    var = K.zeros((2, 2, 3, 4))
    print(K.count_params(var))


def test13():
    input = K.placeholder(shape=(2, 3))
    print(input.shape)
    input = K.transpose(input)
    print(input.shape)


def test14():
    var = np.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
    var = K.variable([[1, 2, 3], [4, 5, 6]])
    # print(var.shape)
    # print(K.eval(K.max(var, axis=1)))
    # print(K.eval(K.max(var)))
    # print(K.eval(K.sum(var,axis=1)))
    # print(K.eval(K.cumsum(var, axis=0)))
    # print(K.eval(K.log(var)))
    # print(K.eval(K.minimum([1, 8, 3], [4, 5, 6])))
    # print(K.eval(K.cos(0.0)))
    # print(K.eval(K.repeat(var,2)))
    # print(K.eval(K.repeat_elements(var,2,axis=1)))








if __name__ == '__main__':
    test14()
