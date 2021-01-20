import numpy as np
from typing import List
from sklearn.metrics import normalized_mutual_info_score
from differential_privacy.gradient import Gradient
from differential_privacy.folders.gradient_folder import GradientFolder


class PassGradientFolder(GradientFolder):
    def fold(self, gradients: List[Gradient]) -> Gradient: 
        print('mutual information in the centralized architecture: ')
        print(normalized_mutual_info_score(np.concatenate(gradients[0].get(), axis=None), np.concatenate(gradients[0].get(), axis=None)))
        return gradients[0]
