#encoding:utf-8
from keras.layers import LocallyConnected1D, LocallyConnected2D
from keras.models import Sequential

def test1():
    # 将长度为 3 的非共享权重 1D 卷积应用于
    # 具有 10 个时间步长的序列，并使用 64个 输出滤波器
    model = Sequential()
    model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
    # 现在 model.output_shape == (None, 8, 64)  10-3+1=8
    print(model.output_shape)
    # 在上面再添加一个新的 conv1d
    model.add(LocallyConnected1D(32, 3))
    # 现在 model.output_shape == (None, 6, 32) 8-3+1=6
    print(model.output_shape)

def test2():
    # 在 32x32 图像上应用 3x3 非共享权值和64个输出过滤器的卷积
    # 数据格式 `data_format="channels_last"`：
    model = Sequential()
    model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
    # 现在 model.output_shape == (None, 30, 30, 64)
    # 注意这一层的参数数量为 (30*30)*(3*3*3*64) + (30*30)*64
    print(model.output_shape)

    # 在上面再加一个 3x3 非共享权值和 32 个输出滤波器的卷积：
    model.add(LocallyConnected2D(32, (3, 3)))
    # 现在 model.output_shape == (None, 28, 28, 32)
    print(model.output_shape)


if __name__ == '__main__':
    test2()
    pass