# encoding:utf-8
from keras.layers import TimeDistributed, Dense, Conv2D
from keras.models import Sequential

model = Sequential()
model.add(TimeDistributed(Dense(12), input_shape=(10, 16)))
print(model.output_shape)
model.add(TimeDistributed(Dense(32)))
print(model.output_shape)
print(model.summary())

model1 = Sequential()
model1.add(TimeDistributed(Conv2D(64, (3, 3), padding='same'),
                           input_shape=(10, 299, 299, 3)))
print(model1.output_shape)
