from typing import List, NoReturn, Self

from encoders import one_hot_encode

from nn_modules import Module, Softmax

import numpy as np


class MLPClassifier:
    def __init__(
            self: Self,
            modules: List[Module],
            epochs: int = 40,
            alpha: float = 0.01,
    ) -> NoReturn:
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и
            описывающий слои нейронной сети.
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обученияю
        alpha : float
            Cкорость обучения.
        """
        self.modules = modules
        self.epochs = epochs
        self.alpha = alpha

    def fit(
            self: Self,
            xs: np.ndarray,
            y: np.ndarray,
            batch_size: int = 32,
    ) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох.
        В каждой эпохе необходимо использовать cross-entropy loss для обучения,
        а так же производить обновления не по одному элементу,
        а используя батчи.

        Parameters
        ----------
        xs : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        batch_size : int
            Размер батча.
        """
        classes = np.max(y) + 1
        num_batches = len(xs) // batch_size + (len(xs) % batch_size != 0)
        x_batches = np.array_split(xs, num_batches)
        y_batches = np.array_split(y, num_batches)
        for _ in range(self.epochs):
            for xs, y in zip(x_batches, y_batches):
                inputs = xs

                for layer in self.modules:
                    inputs = layer.forward(inputs)

                softmax = Softmax()
                _ = softmax.forward(inputs)

                inputs = softmax.backward(one_hot_encode(y, classes))
                for i in range(len(self.modules) - 1, -1, -1):
                    layer = self.modules[i]
                    inputs = layer.backward(inputs)
                    layer.update(self.alpha)

    def predict_proba(self: Self, xs: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов xs.

        Parameters
        ----------
        xs : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)

        """
        inputs = xs
        for layer in self.modules:
            inputs = layer.forward(inputs)
        return inputs

    def predict(self: Self, xs: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Вектор предсказанных классов

        """
        p = self.predict_proba(xs)
        return np.argmax(p, axis=1)
