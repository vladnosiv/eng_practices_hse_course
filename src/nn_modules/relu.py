import copy
from typing import NoReturn, Self

import numpy as np

from .module import Module


class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU.
    """

    def __init__(self: Self) -> NoReturn:
        self.d_y = None

    def forward(self: Self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.

        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        self.x = copy.deepcopy(x)
        return np.where(x > 0, x, 0)

    def backward(self: Self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        return d * (self.x >= 0)
