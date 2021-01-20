import os
import numpy as np
from typing import List
from sklearn.metrics import normalized_mutual_info_score
from differential_privacy.dataset import Dataset
from differential_privacy.gradient import Gradient
from .gradient_folder import GradientFolder


FLOAT_BYTES = 4
RANDOM_MIN = 0.95
RANDOM_MAX = 1.05


class HybridGradientFolder(GradientFolder):
    def __init__(self, neural_network, dataset: Dataset):
        self.generalisation_dataset = dataset
        self.network = neural_network

    def fold(self, gradients: List[Gradient]) -> Gradient:
        grades = list(map(self._evaluate_gradient, gradients))
        grade_sum = sum(grades)
        result = gradients[0] * 0
        for gradient, grade in zip(gradients, grades):
            random_uniform = int.from_bytes(os.urandom(FLOAT_BYTES), byteorder='big') / 2 ** (FLOAT_BYTES * 8)
            random_multiplier = random_uniform * (RANDOM_MAX - RANDOM_MIN) + RANDOM_MIN
            result += gradient * grade * random_multiplier
        result = result * (1 / grade_sum)
        print('mutual information in the fog-embedded architecture: ')
        for gradient in gradients: 
            print(normalized_mutual_info_score(np.concatenate(result.get(), axis=None), np.concatenate(result.get(), axis=None)))
            print(normalized_mutual_info_score(np.concatenate(result.get(), axis=None), np.concatenate(gradients[0].get(), axis=None)))
        return result

    def _evaluate_gradient(self, gradient):
        network = self.network.clone()
        network.apply_gradient(gradient)
        results = network.evaluate(self.generalisation_dataset)
        return results['accuracy']
