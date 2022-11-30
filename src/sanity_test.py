from typing import NoReturn

from classifier import MLPClassifier

from nn_modules import Linear, ReLU

import numpy as np

from sklearn.datasets import make_blobs, make_moons


def test_synth_moons() -> NoReturn:
    x, y = make_moons(400, noise=0.075)

    x_test, y_test = make_moons(400, noise=0.075)

    best_acc = 0
    for _ in range(25):
        p = MLPClassifier(
            modules=[Linear(x.shape[1], 64), ReLU(), Linear(64, 2)],
            epochs=100,
        )

        p.fit(x, y)
        score = np.mean(p.predict(x_test) == y_test)
        best_acc = max(score, best_acc)

    assert best_acc >= 0.85


def test_sanity_blobs() -> NoReturn:
    x, y = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5], [-2.5, 3]])
    x_test, y_test = make_blobs(
        400,
        2,
        centers=[[0, 0], [2.5, 2.5], [-2.5, 3]],
    )
    best_acc = 0
    for _ in range(5):
        p = MLPClassifier(
            modules=[
                Linear(x.shape[1], 64),
                ReLU(),
                Linear(64, 64),
                ReLU(),
                Linear(64, 3),
            ],
            epochs=150,
        )

        p.fit(x, y)
        score = np.mean(p.predict(x_test) == y_test)
        best_acc = max(score, best_acc)

    assert best_acc >= 0.85
