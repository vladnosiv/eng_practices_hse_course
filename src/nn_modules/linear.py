from typing import NoReturn, Self

import numpy as np

from .module import Module


class Linear(Module):
    """
    Линейный полносвязный слой.
    """

    def __init__(self: Self, in_features: int, out_features: int) -> NoReturn:
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int
            Размер выхода.

        Notes
        -----
        W и b инициализируются случайно.
        """
        self.in_features = in_features
        self.out_features = out_features

        k = 1 / in_features
        self.W = np.random.normal(
            loc=0, scale=np.sqrt(k), size=(in_features + 1, out_features),
        )

        self.x_previous = None
        self.delta = None

    def forward(self: Self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).

        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        self.x_previous = np.c_[np.ones(x.shape[0]), x]
        return self.x_previous @ self.W

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
        self.delta = d
        return np.delete(d @ self.W.T, 0, 1)

    def update(self: Self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
        self.W -= alpha * self.x_previous.T @ self.delta
