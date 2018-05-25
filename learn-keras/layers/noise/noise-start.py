#encoding:utf-8
from keras.models import Sequential
from keras.layers import Dense,GaussianDropout, GaussianNoise

model = Sequential()
model.add(Dense(32,activation='relu',input_dim=10))
model.add(GaussianDropout(0.1))
model.add(GaussianNoise(0.01))
model.add(Dense(10,activation='softmax'))
print(model.summary())