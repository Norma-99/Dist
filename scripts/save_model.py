import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mutual_info_score

MODEL_PATH = '../models/adult_1nodes' #'../models/adult_3nodes'

def save_model_weights(model):
    # Save model with weights (tf.model.save -> h5)
    model.save_weights(MODEL_PATH) 

def load_model_weights(model):
    # Load model with weights (tf.model.load)
    # Extract weights from model (model.getWeights)
    weights = model.load_weights(MODEL_PATH)

def process_model_weights(weights):
    return weights.flatten()

def calculate_mutual_information(node_weights, cloud_weights):
    return mutual_info_score(node_weights,cloud_weights)


if __name__ == '__main__':
    node_weights = np.array([[0.2, 0.2, 0.0], [0.4, 0.4, 0.0]]) # Lo que me da dp de entrenar la red
    # cloud_weights = np.array([[0.2, 0.2, 0.0], [0.4, 0.4, 0.0]]) #Centralized case
    cloud_weights = np.array([[0.15, 0.3, 0.3], [0.55, 0.35, 0.0]]) # Lo que me da dp de los algoritmos

    print(calculate_mutual_information(process_model_weights(node_weights), process_model_weights(cloud_weights)))