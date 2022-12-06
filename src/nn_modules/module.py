from typing import NoReturn

import numpy as np


class Module:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def update(self, alpha: float) -> NoReturn:
        pass
