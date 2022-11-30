from typing import NoReturn, Self

import numpy as np

from .module import Module


class Softmax(Module):
    """
    Слой, соответствующий функции активации Softmax.
    """

    def __init__(self: Self) -> NoReturn:
        self.d_y = None

    def forward(self: Self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Softmax(x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.

        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        self.x = x
        return np.array(
            [np.exp(x[i]) / np.sum(np.exp(x[i])) for i in range(x.shape[0])],
        )

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
        return np.array(
            [
                -d[i] + np.exp(self.x[i]) / np.sum(np.exp(self.x[i]))
                for i in range(d.shape[0])
            ],
        )
