import pickle
import numpy as np
import pandas as pd

import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


x_train, y_train = load_data('../datasets/train_dataset.pickle')
x_test, y_test = load_data('../datasets/validation_dataset.pickle')

def binary_model():
    # create model
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(32,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# build the model
model = binary_model()
model.summary()

# fit the model
model.compile(optimizer="SGD", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=8, verbose=2)