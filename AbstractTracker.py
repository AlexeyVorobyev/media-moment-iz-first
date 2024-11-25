import numpy as np
from abc import ABC, abstractmethod

type bounding_box_type = (float, float, float, float)

class AbstractTracker(ABC):
    @abstractmethod
    def init(
            self,
            frame: np.ndarray,
            bounding_box: bounding_box_type
    ) -> bool:
        """
        Инициализация трекера с первым изображением и областью (bounding box).

        :param frame: Изображение (кадр видео).
        :param bounding_box: Прямоугольник для отслеживания объекта (x, y, w, h).
        :return: True, если инициализация успешна, иначе False.
        """
        pass

    @abstractmethod
    def update(self, frame: np.ndarray) -> (bool, bounding_box_type):
        """
        Обновляет положение объекта на новом кадре.

        :param frame: Новый кадр изображения (или видеопотока).
        :return: Кортеж (success, boundingBox), где success — успешность обновления,
                 а boundingBox — новые координаты объекта.
        """
        pass

