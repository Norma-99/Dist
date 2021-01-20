from differential_privacy.folders import PassGradientFolder, MeanGradientFolder, PonderatedGradientFolder, ThresholdGradientFolder, HybridGradientFolder


class GradientFactory:
    @staticmethod
    def from_name(name: str, **kwargs):
        if name == 'mean':
            return MeanGradientFolder()
        elif name == 'ponderated':
            return PonderatedGradientFolder(**kwargs)
        elif name == 'threshold':
            return ThresholdGradientFolder()
        elif name == 'pass': 
            return PassGradientFolder()
        elif name == 'hybrid':
            return HybridGradientFolder(**kwargs)
        raise ValueError('Folder not found')
