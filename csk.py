import cv2
import numpy as np

from AbstractTracker import AbstractTracker, bounding_box_type


class CSK(AbstractTracker):
    """
    Реализация алгоритма CSK (Continuous Sparse Kernel) для отслеживания объектов.

    Алгоритм CSK использует ядра корреляции для отслеживания объектов в видеопоследовательности.
    Он применяет плотное выборочное покрытие фрагментов изображения и использует быстрое преобразование Фурье
    для эффективного вычисления корреляции.
    """

    def __init__(self):
        """
        Инициализация трекера CSK с настройкой гиперпараметров по умолчанию.
        """
        self.x1 = None
        self.y1 = None
        self.height = None
        self.width = None

        self.eta = 0.075  # Скорость обучения для обновления модели
        self.sigma = 0.2  # Ширина гауссового ядра
        self.lmbda = 0.01  # Параметр регуляризации

    def init(
            self,
            frame: np.ndarray,
            bounding_box: bounding_box_type
    ) -> bool:
        """
         Инициализация трекера с первым кадром и исходной рамкой.
        :param frame: Исходный кадр видео (цветное изображение).
        :param bounding_box: Исходная рамка (x1, y1, ширина, высота).
        :return: True, если инициализация выполнена успешно.
        """
        x1, y1, width, height = bounding_box
        # Сохраняем изначальные позиции bbox
        self.x1 = x1
        self.y1 = y1

        # Корректируем размеры, чтобы они были четными
        self.width = width if width % 2 == 0 else width - 1
        self.height = height if height % 2 == 0 else height - 1

        # Извлекаем начальный регион интереса и применяем предобработку
        self.x = self.crop(frame, x1, y1, self.width, self.height)

        # Генерируем карту отклика (гауссовая форма)
        self.y = self.target(self.width, self.height)
        self.prev = np.unravel_index(np.argmax(self.y, axis=None), self.y.shape)  # Maximum position

        # Обучаем начальную модель
        self.alphaf = self.training(self.x, self.y, self.sigma, self.lmbda)
        return True

    def update(self, frame: np.ndarray) -> (bool, bounding_box_type):
        """
        Обновление трекера с новым кадром.
        :param frame: Текущий кадр видео.
        :return: (bool, bounding_box_type): Успешность и обновленная рамка (x1, y1, ширина, высота).
        """
        # Извлекаем регион интереса на основе предыдущей позиции
        if self.x1 is None or self.y1 is None or self.width is None or self.height is None:
            return False, None
        z = self.crop(frame, self.x1, self.y1, self.width, self.height)

        # Вычисляем карту отклик
        responses = self.detection(self.alphaf, self.x, z, 0.2)
        curr = np.unravel_index(np.argmax(responses, axis=None), responses.shape)
        dy = curr[0] - self.prev[0]
        dx = curr[1] - self.prev[1]

        # Обновляем позицию на основе смещения максимума отклика
        self.x1 = self.x1 - dx
        self.y1 = self.y1 - dy

        # Убедимся, что новая рамка остается в пределах кадра
        if self.x1 < 0:
            self.x1 = 0
            # return False, None
        if self.y1 < 0:
            self.y1 = 0
            # return False, None
        if self.x1 + self.width > frame.shape[1]:
            self.x1 = frame.shape[1] - self.width
            # return False, None
        if self.y1 + self.height > frame.shape[0]:
            self.y1 = frame.shape[0] - self.height
            # return False, None

        # Обновляем модель с использованием линейной интерполяции
        xtemp = self.eta * self.crop(frame, self.x1, self.y1, self.width, self.height) + (1 - self.eta) * self.x
        self.x = self.crop(frame, self.x1, self.y1, self.width, self.height)

        self.alphaf = (self.eta * self.training(self.x, self.y, 0.2, 0.01)
                       + (1 - self.eta) * self.alphaf)  # Линейная интерполяция
        self.x = xtemp

        return True, (self.x1, self.y1, self.width, self.height)

    def dgk(self, x1: np.ndarray, x2: np.ndarray, sigma: float):
        """
        Вычисление плотного гауссового ядра между двумя фрагментами изображения.
        :param x1: Первый фрагмент изображения.
        :param x2: Второй фрагмент изображения.
        :param sigma: Ширина ядра гауссовой функции.
        :return: Матрица откликов ядра.
        """
        c = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(x1) * np.conj(np.fft.fft2(x2))))
        d = np.dot(np.conj(x1.flatten(order='K')), x1.flatten(order='K')) + np.dot(np.conj(x2.flatten(order='K')),
                                                                                   x2.flatten(order='K')) - 2 * c
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / np.size(x1))
        return k

    def training(self, x, y, sigma, lmbda):
        """
        Обучение модели корреляционного фильтра.

        :param x: Входной фрагмент.
        :param y: Карта отклика цели.
        :param sigma: Ширина ядра гауссовой функции.
        :param lmbda: Параметр регуляризации.
        :return: Преобразование Фурье обученной модели.
        """
        k = self.dgk(x, x, sigma)
        alphaf = np.fft.fft2(y) / (np.fft.fft2(k) + lmbda)
        return alphaf

    def detection(self, alphaf: np.ndarray, x: np.ndarray, z: np.ndarray, sigma: float) -> np.ndarray:
        """
        Выполнение обнаружения путем вычисления карты откликов.

        :param alphaf: Модель в частотной области.
        :param x: Опорный фрагмент.
        :param z: Фрагмент для поиска.
        :param sigma:  Ширина ядра гауссовой функции.
        :return: Карта откликов.
        """
        k = self.dgk(x, z, sigma)
        responses = np.real(np.fft.ifft2(alphaf * np.fft.fft2(k)))
        return responses

    def window(self, img: np.ndarray) -> np.ndarray:
        """
        Применение косинусного окна для уменьшения краевых эффектов.

        :param img: Входной фрагмент изображения.
        :return: Фрагмент изображения с применением окна.
        """
        height = img.shape[0]
        width = img.shape[1]

        j = np.arange(0, width)
        i = np.arange(0, height)
        J, I = np.meshgrid(j, i)
        window = np.sin(np.pi * J / width) * np.sin(np.pi * I / height)
        windowed = window * ((img / 255) - 0.5)

        return windowed

    def crop(self, img: np.ndarray, x1: float, y1: float, width: int, height: int) -> np.ndarray:
        """
        Вырезка и предобработка фрагмента изображения вокруг заданной области.

        :param img: Исходный кадр.
        :param x1: Координата центра по x.
        :param y1: Координата центра по y.
        :param width:
        :param height:
        :return: Вырезанный и предобработанный фрагмент изображения.
        """
        pad_y = [0, 0]
        pad_x = [0, 0]

        if (y1 - height / 2) < 0:
            y_up = 0
            pad_y[0] = int(-(y1 - height / 2))
        else:
            y_up = int(y1 - height / 2)

        if (y1 + 3 * height / 2) > img.shape[0]:
            y_down = img.shape[0]
            pad_y[1] = int((y1 + 3 * height / 2) - img.shape[0])
        else:
            y_down = int(y1 + 3 * height / 2)

        if (x1 - width / 2) < 0:
            x_left = 0
            pad_x[0] = int(-(x1 - width / 2))
        else:
            x_left = int(x1 - width / 2)

        if (x1 + 3 * width / 2) > img.shape[1]:
            x_right = img.shape[1]
            pad_x[1] = int((x1 + 3 * width / 2) - img.shape[1])
        else:
            x_right = int(x1 + 3 * width / 2)

        cropped = img[y_up:y_down, x_left:x_right]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        padded = np.pad(cropped, (pad_y, pad_x), 'edge')
        windowed = self.window(padded)
        return windowed

    def target(self, width: int, height: int) -> np.ndarray:
        """
        Генерация гауссовой формы карты отклика.

        :param width: Ширина области цели.
        :param height: Высота области цели.
        :return:
        """
        double_height = 2 * height
        double_width = 2 * width
        s = np.sqrt(double_height * double_width) / 16

        j = np.arange(0, double_width)
        i = np.arange(0, double_height)
        J, I = np.meshgrid(j, i)
        y = np.exp(-((J - width) ** 2 + (I - height) ** 2) / s ** 2)

        return y
