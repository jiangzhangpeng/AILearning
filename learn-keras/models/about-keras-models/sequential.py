# encoding:utf-8
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets.mnist import load_data
from numpy import shape, array, reshape
import keras


def prepareTxtData():
    (x_train, y_train), (x_test, y_test) = load_data(
        'D:\Workspaces\Github\AILearning\learn-keras\getting-started\mnist.npz')
    # 数据处理
    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train / 255
    x_test = x_test.reshape((10000, 28 * 28))
    x_test = x_test / 255
    # label处理
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    f = open('data.txt', 'w')
    for i in range(shape(x_train)[0]):
        f.write(
            ' '.join([str(x) for x in list(x_train[i])]) + ' ' + ' '.join([str(x) for x in list(y_train[i])]) + '\n')
    f.close()


def createModel():
    model = Sequential()
    model.add(Dense(units=500, activation='relu', input_dim=784))
    model.add(Dropout(0.2))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_data_generator(file):
    print('kkkk')
    while True:
        f = open(file)
        lines = f.readlines()
        i = 0
        for line in lines:
            tmp = line.strip(' ').split(' ')
            x = reshape(array([float(t) for t in tmp[0:28 * 28]]), (1, 784))
            y = reshape(array([float(t) for t in tmp[28 * 28:]]), (1, 10))
            yield (x, y)
            i += 1
        f.close()
        print(i)


def train_and_evaluate():
    model = createModel()
    t = model.get_layer(index=0)
    print(t)
    hist = model.fit_generator(train_data_generator('data.txt'), steps_per_epoch=100, epochs=2,verbose=0)
    print(hist.history)
    t = model.get_layer(index=1)
    weights = t.get_weights()
    print(weights)



if __name__ == '__main__':
    # prepareTxtData()
    train_and_evaluate()
