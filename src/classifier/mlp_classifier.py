from typing import List, NoReturn

import numpy as np

from encoders import one_hot_encode
from nn_modules import Module, Softmax


class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01):
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

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size=32) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох.
        В каждой эпохе необходимо использовать cross-entropy loss для обучения,
        а так же производить обновления не по одному элементу, а используя батчи.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        batch_size : int
            Размер батча.
        """
        classes = np.max(y) + 1
        num_batches = len(X) // batch_size + (len(X) % batch_size != 0)
        X_batches = np.array_split(X, num_batches)
        y_batches = np.array_split(y, num_batches)
        for epoch in range(self.epochs):
            for X, y in zip(X_batches, y_batches):
                inputs = X

                for layer in self.modules:
                    inputs = layer.forward(inputs)

                softmax = Softmax()
                _ = softmax.forward(inputs)

                inputs = softmax.backward(one_hot_encode(y, classes))
                for i in range(len(self.modules) - 1, -1, -1):
                    layer = self.modules[i]
                    inputs = layer.backward(inputs)
                    layer.update(self.alpha)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)

        """
        inputs = X
        for layer in self.modules:
            inputs = layer.forward(inputs)
        return inputs

    def predict(self, X) -> np.ndarray:
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
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)
