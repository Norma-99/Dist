import typing
from typing import List
from differential_privacy.dataset import Dataset
from differential_privacy.gradient import Gradient
from differential_privacy.folders.gradient_folder import GradientFolder

class ThresholdGradientFolder(GradientFolder):
    def __init__(self, generalisation_dataset: Dataset):
        self.generalisation_dataset = generalisation_dataset

    def fold(self, neural_network, gradients: List[Gradient]) -> Gradient:
        pass