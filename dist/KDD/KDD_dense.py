import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import pickle

from tensorflow.python.keras.engine.input_layer import InputLayer



def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


x_train, y_train = load_data('../../datasets/KDD/validation_dataset.pickle')
x_test, y_test = load_data('../../datasets/KDD/train_dataset.pickle')

"""***********PRE-TRAINING**********"""
model=Sequential()
model.add(InputLayer(input_shape=x_train.shape[1:]))
# paso de la capa visible a la hidden layers
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.45))
model.add(Dense(88, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(188, activation='tanh'))
model.add(Dropout(0.35))
model.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics = ['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test,y_test),batch_size=68, epochs=20, verbose=2)
"""modelo decoder"""

















