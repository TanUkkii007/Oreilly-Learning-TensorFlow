import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "./cifar-10-batches-py/"

def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict

def one_hot(vec, vals=10):
    '''
    vec: n-length vector contains 0-9 values
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

class CifarLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None
    
    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack(d["data"] for d in data)
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0,2,3,1).astype(float) / 255
        self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self
    
    def next_batch(self, batch_size):
        x = self.images[self._i:self._i + batch_size]
        y = self.labels[self._i:self._i + batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)]).load()
        self.test = CifarLoader(["test_batch"]).load()

