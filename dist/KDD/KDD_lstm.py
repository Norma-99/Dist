import tensorflow as tf
from tensorflow import keras
import pickle
from tensorflow.keras.layers import Dropout,Dense,LSTM,InputLayer
from tensorflow.keras.optimizers import Adam
import kerastuner as kt 

from tensorflow.python.keras.optimizers import Adamax
from tensorflow.python.keras.models import Sequential

import numpy as np

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


x_train, y_train = load_data('./datasets/KDD/validation_dataset.pickle')
x_test, y_test = load_data('./datasets/KDD/train_dataset.pickle')

"""***********RESHAPE PARA TRATO CON LSTM **********"""
x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))

print(x_train.shape)
print(y_train.shape)

model=Sequential()
model.add(InputLayer(input_shape=x_train.shape[1:]))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(200, return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adamax(learning_rate=0.01)
model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics = ['accuracy'])

model.summary()

model.fit(x_train, y_train, validation_data=(x_test,y_test),batch_size=50, epochs=20)