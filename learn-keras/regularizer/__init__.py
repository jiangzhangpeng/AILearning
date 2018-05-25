# encoding:utf-8

import keras.backend as K
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential


def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))


model = Sequential()
model.add(Dense(64, input_dim=784, kernel_regularizer=regularizers.l1(0.01), activation='relu'))
model.add(Dense(64, kernel_regularizer=l1_reg, activation='relu'))
model.add(Dense(10, activation='softmax'))
print(model.summary())
