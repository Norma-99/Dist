import pickle
import numpy as np
import pandas as pd

import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.engine.input_layer import InputLayer


def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


x_train, y_train = load_data('./datasets/Bot/validation_dataset.pickle')
x_test, y_test = load_data('./datasets/Bot/train_dataset.pickle')

"""***********MODELO FINAL**********"""
def binary_model():
    # create model
    model = Sequential()
    model.add(Dense(31, activation='relu', input_shape=(31,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# build the model
model = binary_model()

# fit the model
model.compile(optimizer="SGD", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=15, verbose=2)





