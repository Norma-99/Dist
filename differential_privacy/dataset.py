import pickle
import numpy as np

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def get(self):
        return self.x, self.y

    def get_split(self, index: int, split_count: int):
        size = len(self.x) // split_count
        start = index * size
        end = start + size
        return Dataset(self.x[start:end], self.y[start:end])

    def get_generalisation_fragment(self, fraction: int):
        return self.get_split(0, fraction)

    def __eq__(self, other):
        return np.sum(self.x - other.x) == 0 and np.sum(self.y - other.y) == 0

    def __repr__(self):
        return f'Dataset({self.x.shape}, {self.y.shape})'

    @staticmethod
    def from_file(path: str):
        with open(path, 'rb') as f:
            return Dataset(*pickle.load(f))

    @staticmethod
    def join(datasets: dict):
        xs, ys = [], []
        for dataset in datasets.values():
            x, y = dataset.get()
            xs.append(x)
            ys.append(y)
        return Dataset(np.concatenate(xs, axis=0), np.concatenate(ys, axis=0))
