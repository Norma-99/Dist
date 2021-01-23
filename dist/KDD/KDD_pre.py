from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import pickle


"""***********PREPROCESSING**********"""

def recopilar_datos(filepath):

    train_plus = pd.read_csv(filepath)
    """Recopilar dataset"""
    precedent_raw = train_plus.iloc[:, 1:41]
    precedent = pd.get_dummies(precedent_raw)
    target_raw = train_plus.iloc[:, 41]

    target=1*(target_raw=='normal')# para tener solo una salida
    """Recortamos la columna de ID y apicamos One-Hot para numerizar los valores categóricos"""
    return precedent, target

def normalise(matrix):
    return StandardScaler().fit_transform(matrix.values)
    
fliepath_train="../../datasets/KDD/KDDall.csv"  
x_train, y_train = recopilar_datos(fliepath_train)
x_test = x_train.iloc[102417:, :] 
y_test=y_train.iloc[102417:] 
#test dataset
x_train=x_train.iloc[1:102417,:]
y_train=y_train.iloc[1:102417] 
#train dataset


x_train=normalise(x_train)
x_test=normalise(x_test)

print(x_train.shape)
print(y_train.shape)



# Save information into .pickle format
validation_pair = x_test, y_test
training_pair = x_train, y_train
with open('../../datasets/KDD/validation_dataset.pickle', 'wb') as f:
    pickle.dump(validation_pair, f)

with open('../../datasets/KDD/train_dataset.pickle', 'wb') as f:
    pickle.dump(training_pair, f)
