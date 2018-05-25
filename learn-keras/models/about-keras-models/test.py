# encoding:utf-8

from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=100))
config = model.get_config()

# model1 = Model.from_config(config) #报错 无法运行
model1 = Sequential.from_config(config)
config1 = model1.get_config()

print('config:', config)
print('config1:', config1)
print('summary:', model.summary())
print('weights:', model.get_weights())

json_string = model.to_json()
print('json-string:', json_string)

model2 = model_from_json(json_string)
print(model2.get_config())

model.train_on_batch()
