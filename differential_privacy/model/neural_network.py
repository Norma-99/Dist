import os
from typing import Dict
import tensorflow as tf
from differential_privacy.dataset import Dataset
from differential_privacy.gradient import Gradient
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class NeuralNetwork:
    def __init__(self, tf_model, epochs, validation_dataset, trace_path):
        self.tf_model: tf.keras.Model = tf_model
        self.tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.epochs: int = epochs
        self.validation_dataset: Dataset = validation_dataset
        self.trace_path = trace_path

    def fit(self, data:Dataset) -> Gradient:
        initial_weights = self.tf_model.get_weights()
        self.tf_model.fit(*data.get(), batch_size=68, epochs=self.epochs, verbose=2)
        final_weights = self.tf_model.get_weights()
        return Gradient.from_delta(initial_weights, final_weights)

    def clone(self):
        new_model = tf.keras.models.clone_model(self.tf_model)
        new_model.set_weights(self.tf_model.get_weights())
        return NeuralNetwork(new_model, self.epochs, self.validation_dataset, self.trace_path)

    def evaluate(self, dataset: Dataset = None) -> Dict[str, float]:
        if dataset is not None:
            x, y = dataset.get()
            return self.tf_model.evaluate(x, y, return_dict=True)
        return self.tf_model.evaluate(*self.validation_dataset.get(), return_dict=True)

    def save_trace(self, trace_id: int):
        results = self.evaluate()
        if not os.path.isfile(self.trace_path):
            with open(self.trace_path, 'w') as f:
                f.write('trace_id,' + ','.join(results.keys()))
                f.write('\n')
        with open(self.trace_path, 'a') as f:
            f.write(str(trace_id) + ',' + ','.join(map(str, results.values())))
            f.write('\n')

    def apply_gradient(self, gradient: Gradient):
        new_weights = Gradient(self.tf_model.get_weights()) + gradient
        self.tf_model.set_weights(new_weights.get())

    def __repr__(self):
        return f'Model(layer_count={len(self.tf_model.layers)}, optimizer=adam, loss=binary_crossentropy)'
