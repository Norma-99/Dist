from typing import List
from differential_privacy.gradient import Gradient
from differential_privacy.folders.gradient_folder import GradientFolder


class MeanGradientFolder(GradientFolder):
    def fold(self, gradients: List[Gradient]) -> Gradient:
        gradient_weight = 1 / len(gradients)
        result = gradients[0] * 0
        for gradient in gradients:
            result += gradient * gradient_weight
        return result
