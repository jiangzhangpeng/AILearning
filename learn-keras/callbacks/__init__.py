# encoding:utf-8
import keras

# ReduceLROnPlateau
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=0.001)
model = keras.models.Sequential()
model.add(keras.layers.Dense(64, input_dim=10))
model.add(keras.layers.Dense(10))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([], [], callbacks=[reduce_lr])


