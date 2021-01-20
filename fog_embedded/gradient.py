from typing import List
import numpy as np


class Gradient: 
    def __init__(self, layer_deltas):
        self._layer_deltas: List[np.ndarray] = layer_deltas

    @staticmethod
    def from_delta(initial_weights: List[np.ndarray], final_weights: List[np.ndarray]):
        return Gradient(final_weights) - Gradient(initial_weights)

    def __add__(self, other):
        return Gradient(
            [self_gradient + other_gradient
             for self_gradient, other_gradient in zip(self.get(), other.get())]
        )

    def __neg__(self):
        return Gradient([-gradient for gradient in self.get()])

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other: float):
        return Gradient([gradient * other for gradient in self.get()])

    def __eq__(self, other):
        return sum(map(np.sum, (self - other).get())) < 0.0001

    def get(self) -> List[np.ndarray]:
        return self._layer_deltas
