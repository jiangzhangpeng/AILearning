# encoding:utf-8
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
import  keras
model = Sequential()
model.add(Dense(64,input_dim=10,kernel_initializer='uniform'))
model.add(Activation('tanh'))
model.add(Activation('softmax'))
print(model.summary())

sgd=optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd,metrics=[keras.metrics.binary_crossentropy],loss=keras.losses.categorical_crossentropy)

model.compile(optimizer='sgd',metrics=[keras.metrics.binary_crossentropy],loss=keras.losses.categorical_crossentropy)