#encoding:utf-8
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential

layer = Dense(32)
#layer.set_weights([0.3,0.8])
config = layer.get_weights()
print(config)

model = Sequential()
model.add(Dense(units=500,activation='relu',input_dim=784))
model.add(Dropout(0.2))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10,activation='softmax'))

layer1 = model.get_layer(index = 1)
print(layer1)
print(layer1.input)
print(layer1.output)

activation = Activation()