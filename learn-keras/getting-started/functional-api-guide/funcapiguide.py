#encoding:utf-8
from keras.models import Model, load_model
from keras.layers import Dense,Input, Dropout
from keras.datasets.mnist import load_data
from keras.callbacks import EarlyStopping
import keras
from matplotlib import pyplot as plt

import keras.applications.inception_v3 as in3



def loadDataSet():
    (x_train, y_train), (x_test, y_test) = load_data('D:\Workspaces\Github\AILearning\learn-keras\mnist.npz')
    #数据处理
    x_train = x_train.reshape((60000, 28*28))
    x_train = x_train / 255
    x_test = x_test.reshape((10000, 28 * 28))
    x_test = x_test / 255
    #label处理
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    return x_train[0:10001,:], y_train[0:10001,:], x_test[0:2001], y_test[0:2001]


#定义输入层  返回一个张量
inputs = Input(shape=(28*28,))
x = Dense(500,activation='relu')(inputs)
x = Dropout(0.3)(x)
x = Dense(500,activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(10,activation='softmax')(x)

#定义model
model = Model(inputs=inputs,outputs=predictions)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

data = loadDataSet()
#验证集误差不再下降时，提前退出机制
early_stopping = EarlyStopping(monitor='val_loss',patience=2)
hist = model.fit(data[0],data[1],batch_size=128,validation_split=0.2,epochs=20,callbacks=[early_stopping],shuffle=True)
points = []
for i in range(len(hist.history['val_acc'])):
    points.append((i+1,hist.history['val_acc'][i]))
fig = plt.figure()
aux = fig.add_subplot(111)
aux.plot([i+1 for i in range(len(hist.history['val_acc']))],hist.history['val_acc'])
fig.show()


#训练结果的保存及加载
# model.save('funcapiguide.h5')
# del model
#
# modelnew = load_model('funcapiguide.h5')
# result = modelnew.evaluate(data[2],data[3])
# print(result[1])
