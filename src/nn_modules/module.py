from typing import NoReturn, Self

import numpy as np


class Module:
    def forward(self: Self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self: Self, d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def update(self: Self, alpha: float) -> NoReturn:
        pass
