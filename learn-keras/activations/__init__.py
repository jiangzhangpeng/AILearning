# encoding:utf-8
from keras.models import Sequential
from keras.layers import Dense, Activation


model = Sequential()
model.add(Dense(64,input_dim=10,activation='tanh'))
#model.add(Activation('tanh'))
print(model.summary())