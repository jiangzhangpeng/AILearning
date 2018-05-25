#encoding:utf-8
import keras as K
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.layers import Dense,Dropout
from tensorflow.examples.tutorials.mnist import input_data
from numpy import shape, reshape
import numpy as np
def loadDataSet():
    (x_train, y_train), (x_test, y_test) = load_data('D:\Workspaces\Github\AILearning\learn-keras\getting-started\mnist.npz')
    #数据处理
    x_train = x_train.reshape((60000, 28*28))
    x_train = x_train / 255
    x_test = x_test.reshape((10000, 28 * 28))
    x_test = x_test / 255
    #label处理
    y_train = K.utils.to_categorical(y_train, num_classes=10)
    y_test = K.utils.to_categorical(y_test, num_classes=10)
    return x_train[0:10001,:], y_train[0:10001,:], x_test[0:2001], y_test[0:2001]


def loadDataByTensor(filepath):
    data = input_data.read_data_sets(filepath,one_hot=True)
    x_train = data.train.images
    y_train = data.train.labels

    x_test = data.test.images
    y_test = data.test.labels

    return x_train,y_train,x_test,y_test


def createModel():
    model = Sequential()
    model.add(Dense(units=500,activation='relu',input_dim=784))
    model.add(Dropout(0.2))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=10,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def trainAndPredict(model,data):
    print(shape(data[0][0]))
    model.fit(data[0],data[1],batch_size=128,epochs=5)
    score = model.evaluate(data[2],data[3],batch_size=128)
    xs = data[0][0:100]
    ys = data[1][0:100]
    yhs = model.predict(xs)
    print([np.argmax(y) for y in ys])
    print([np.argmax(y) for y in yhs])

def test1():
    data = loadDataByTensor('D:\Workspaces\Workspace1\mnist')
    model = createModel()
    trainAndPredict(model, data)

def test2():
    data = loadDataSet()
    model = createModel()
    trainAndPredict(model, data)

if __name__ == '__main__':
    test2()

