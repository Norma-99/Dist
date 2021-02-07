import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM,Reshape, InputLayer, Flatten


import numpy as np
import pandas as pd
import pickle

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


x_train, y_train = load_data('../../datasets/Bot/validation_dataset.pickle')
x_test, y_test = load_data('../../datasets/Bot/train_dataset.pickle')

"""***********MODELO FINAL**********"""


# x_train = np.reshape(x_train, x_train.shape + (1,))
# x_test = np.reshape(x_test, x_test.shape + (1,))

print(x_train.shape)

model=Sequential()
model.add(InputLayer(input_shape=x_train.shape[1:]))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics = ['accuracy'])

model.summary()

model.fit(x_train, y_train, validation_data=(x_test,y_test),batch_size=64, epochs=30)
