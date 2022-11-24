import numpy as np

def one_hot_encode(y, classes):
    res = np.zeros(shape=(y.shape[0], classes))
    for i in range(y.shape[0]):
        res[i][y[i]] = 1
    return res