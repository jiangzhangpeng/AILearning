# encoding:utf-8

from keras.layers import Cropping2D,Conv2D
from keras.models import Sequential


# Conv1D


# Cropping2D
def test1():
    model = Sequential()
    model.add(Cropping2D(cropping=((2, 2), (4, 4)), input_shape=(28, 28, 3)))
    #(2,2)表示上下各剪裁2,（4,4）表示左右各剪裁4
    print(model.output_shape)

    #注意padding  valid  same 默认valid，same保持处理前的大小
    model.add(Conv2D(64,(3,3),padding='valid'))
    print(model.output_shape)

    model.add(Cropping2D(cropping=((2,2),(2,2))))
    print(model.output_shape)


if __name__ == '__main__':
    test1()