import tensorflow as tf
from functools import partial
from tensorflow.keras.layers import Dropout,Dense,LSTM,InputLayer
from tensorflow.python.keras.optimizers import Adamax, RMSprop
from tensorflow.keras.optimizers import Adam
import kerastuner as kt 

import numpy as np
import pandas as pd
import pickle



def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


x_train, y_train = load_data('./datasets/KDD/validation_dataset.pickle')
x_test, y_test = load_data('./datasets/KDD/train_dataset.pickle')


"""***********RESHAPE PARA TRATO CON LSTM **********"""
x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))


"""***********HYPERBAND**********"""

HIDDEN = 3

def model_builder(input_shape, hp):
    model = tf.keras.Sequential()
    model.add(InputLayer(input_shape=x_train.shape[1:]))

    hp_units_0 = hp.Int(f'units{0}', min_value=8, max_value=192, step=4)
    model.add(LSTM(hp_units_0,return_sequences=True))
    hp_Dropout_rate= hp.Float('dropout_0',min_value=0.0, max_value=0.8, step=0.05)
    model.add(Dropout(hp_Dropout_rate))
    
    hp_units_1 = hp.Int(f'units{1}', min_value=8, max_value=192, step=4)
    model.add(LSTM(units=hp_units_1,return_sequences=True))
    hp_Dropout_rate= hp.Float('dropout_1',min_value=0.0, max_value=0.8, step=0.05)
    model.add(Dropout(hp_Dropout_rate))

    hp_units_2 = hp.Int(f'units{2}', min_value=8, max_value=192, step=4)
    model.add(LSTM(units=hp_units_2,return_sequences=True))
    hp_Dropout_rate= hp.Float('dropout_2',min_value=0.0, max_value=0.8, step=0.05)
    model.add(Dropout(hp_Dropout_rate))

    hp_units_3 = hp.Int(f'units{3}', min_value=8, max_value=192, step=4)
    model.add(LSTM(units=hp_units_3,return_sequences=True))
    hp_Dropout_rate= hp.Float('dropout_3',min_value=0.0, max_value=0.8, step=0.05)
    model.add(Dropout(hp_Dropout_rate))

    hp_units_4 = hp.Int(f'units{4}', min_value=8, max_value=192, step=4)
    model.add(LSTM(units=hp_units_4))
    hp_Dropout_rate= hp.Float('dropout_4',min_value=0.0, max_value=0.8, step=0.05)
    model.add(Dropout(hp_Dropout_rate))

    model.add(Dense(1, activation='sigmoid'))


    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, 
                    loss="binary_crossentropy",
                    metrics=['accuracy'])
    return model

tuner = kt.Hyperband(
    #reshape 
	partial(model_builder, x_train.shape[0:]),
	objective = 'val_accuracy',
	max_epochs = 100,
	directory='./hyper/kdd_lstm')

tuner.search(
	x_train,
	y_train,
	epochs=100,
	validation_data= (x_test, y_test))

models = tuner.get_best_models(num_models=1)
models[0].summary()

print(models[0].get_layer(name="dropout").rate)
print(models[0].get_layer(name="dropout_1").rate)
print(models[0].get_layer(name="dropout_2").rate)
print(models[0].get_layer(name="dropout_3").rate)
print(models[0].get_layer(name="dropout_4").rate)