from typing import List
from differential_privacy.gradient import Gradient


class GradientFolder:
    def fold(self, gradients: List[Gradient]) -> Gradient:
        raise NotImplementedError()
