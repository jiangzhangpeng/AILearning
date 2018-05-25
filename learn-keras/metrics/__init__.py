#encoding:utf-8
import keras

model = keras.models.Sequential()
model.add(keras.layers.Dense(64,input_dim=784))
model.compile(loss=keras.losses.mean_squared_error,optimizer='sgd',metrics=[keras.metrics.categorical_accuracy])
print(model.summary())