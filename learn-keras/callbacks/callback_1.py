# encoding:utf-8
from keras.datasets.mnist import load_data
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras


def get_data():
    (x_train, y_train), (x_test, y_test) = load_data(
        'D:\Workspaces\Github\AILearning\learn-keras\getting-started\mnist.npz')
    # feature process
    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train / 255.0
    x_test = x_test.reshape((10000, 28 * 28))
    x_test = x_test / 255.0
    # label process
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    return x_train[0:10001, :], y_train[0:10001, :], x_test[0:2001], y_test[0:2001]


def create_model():
    model = Sequential()
    model.add(Dense(256, input_dim=28 * 28, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add((Dense(10, activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train):
    batch_print_callback = keras.callbacks.LambdaCallback(
        on_batch_begin=lambda batch, logs: print(batch))
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
    history = LossHistory()
    train_history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=0, validation_split=0.2,
                              callbacks=[history, checkpointer, batch_print_callback])
    print(history.losses)
    return model


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_data()
    model = create_model()
    train_model(model, x_train, y_train)