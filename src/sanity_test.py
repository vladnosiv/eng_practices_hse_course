from classifier import MLPClassifier
from nn_modules import Linear, ReLU

from sklearn.datasets import make_blobs, make_moons
import numpy as np


def test_synth_moons():
    X, y = make_moons(400, noise=0.075)

    X_test, y_test = make_moons(400, noise=0.075)

    best_acc = 0
    for _ in range(25):
        p = MLPClassifier(modules=[
            Linear(X.shape[1], 64),
            ReLU(),
            Linear(64, 2)
        ], epochs=100)

        p.fit(X, y)
        score = np.mean(p.predict(X_test) == y_test)
        best_acc = max(score, best_acc)
        
    assert best_acc >= 0.85
    
    
def test_sanity_blobs():
    X, y = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5], [-2.5, 3]])
    X_test, y_test = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5], [-2.5, 3]])
    best_acc = 0
    for _ in range(5):
        p = MLPClassifier(modules=[
            Linear(X.shape[1], 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 3)
        ], epochs=150)

        p.fit(X, y)
        score = np.mean(p.predict(X_test) == y_test)
        best_acc = max(score, best_acc)
    
    assert best_acc >= 0.85
