# encoding:utf-8

import keras.backend as K
from keras import regularizers
from keras.constraints import max_norm
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import plot_model


def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))


model = Sequential()
model.add(Dense(64, input_dim=784, kernel_regularizer=regularizers.l1(0.01), activation='relu'))
model.add(Dense(64, kernel_regularizer=l1_reg, activation='relu', kernel_constraint=max_norm(2.0)))
model.add(Dense(10, activation='softmax'))
print(model.summary())
plot_model(model, to_file='model.png')

