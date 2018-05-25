# encoding:utf-8
from keras.layers import Input, Dense, Add, Subtract, add, subtract
from keras.models import Model

#Add
def test1():
    #定义第一个相加项
    input1 = Input(shape=(16,))
    x1 = Dense(8, activation='relu')(input1)
    #第二个相加项
    input2 = Input(shape=(32,))
    x2 = Dense(8, activation='relu')(input2)
    #相加 相当于keras.layers.add(x1,x2)
    added = Add()([x1, x2])
    #定义输出项
    out = Dense(4)(added)
    #创建模型
    model = Model(inputs=[input1,input2],outputs=out)
    #输出模型概况
    print(model.summary())

#Substract
def test2():
    # 定义第一个相减项
    input1 = Input(shape=(16,))
    x1 = Dense(8, activation='relu')(input1)
    # 第二个相减项
    input2 = Input(shape=(32,))
    x2 = Dense(8, activation='relu')(input2)
    # 相减 相当于keras.layers.subtract(x1,x2)
    subtracted = Subtract()([x1, x2])
    # 定义输出项
    out = Dense(4)(subtracted)
    # 创建模型
    model = Model(inputs=[input1, input2], outputs=out)
    # 输出模型概况
    print(model.summary())
if __name__ == '__main__':
    test2()