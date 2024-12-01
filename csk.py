import cv2
import numpy as np

from AbstractTracker import AbstractTracker, bounding_box_type
from dft import fft_2d, ifft_2d


class CSK(AbstractTracker):
    """
    Реализация алгоритма CSK (Continuous Sparse Kernel) для отслеживания объектов.

    Алгоритм CSK использует ядра корреляции для отслеживания объектов в видеопоследовательности.
    Он применяет плотное выборочное покрытие фрагментов изображения и использует быстрое преобразование Фурье
    для эффективного вычисления корреляции.
    """

    def __init__(self, debug: bool = False):
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
        self.debug = debug
        self.our_fft = False

    def init(
            self,
            frame: np.ndarray,
            bounding_box: bounding_box_type,
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

        # Создать карту отклика, которая служит эталоном для обучения корреляционного фильтра
        self.y = self.target(self.width, self.height)
        if self.debug:
            cv2.imshow('Gauss feedback map', self.y)

        # Определение центра карты отклика
        # Найти координаты пика y (максимального значения), которые задают начальное положение объекта.
        # self.prev хранит координаты центра отклика y
        self.prev = np.unravel_index(
            np.argmax(self.y, axis=None), self.y.shape
        )  # Позиции с максимальными значениями карты отклика
        if self.debug:
            print(f"Начальная максимальная позиция: {self.prev}")

        # Построить начальную модель корреляционного фильтра 𝛼, которая связывает ROI (x) с картой отклика (y).
        self.alphaf = self.training(self.x, self.y, self.sigma, self.lmbda)
        return True

    def update(self, frame: np.ndarray) -> (bool, bounding_box_type):
        """
        Обновление трекера с новым кадром.
        :param frame: Текущий кадр видео.
        :return: (bool, bounding_box_type): Успешность и обновленная рамка (x1, y1, ширина, высота).
        """

        # Проверка наличия начальных параметров
        if self.x1 is None or self.y1 is None or self.width is None or self.height is None:
            return False, None

        # Извлекаем регион интереса на основе предыдущей позиции
        z = self.crop(frame, self.x1, self.y1, self.width, self.height)

        # Создаёт карту отклика responses,
        # которая показывает вероятность нахождения объекта в каждом циклическом сдвиге ROI
        responses = self.detection(self.alphaf, self.x, z, self.sigma)

        # Ищем пик карты отклика, чтобы определить смещение объекта относительно предыдущей позиции
        # np.argmax(responses) определяет индекс пика карты отклика (максимальное значение).
        # np.unravel_index преобразует этот индекс в координаты (y, x).
        curr = np.unravel_index(np.argmax(responses, axis=None), responses.shape)
        # Разница между текущей позицией curr и предыдущей позицией prev даёт смещение
        dy = curr[0] - self.prev[0]
        dx = curr[1] - self.prev[1]

        # Обновляем координаты центра объекта с учётом найденного смещения
        self.x1 = self.x1 - dx
        self.y1 = self.y1 - dy
        if self.debug:
            print(f"Обновлённые координаты объекта: ({self.x1}, {self.y1})")

        # Убедимся, что новая рамка остается в пределах кадра
        if self.x1 < 0:
            self.x1 = 0
        if self.y1 < 0:
            self.y1 = 0
        if self.x1 + self.width > frame.shape[1]:
            self.x1 = frame.shape[1] - self.width
        if self.y1 + self.height > frame.shape[0]:
            self.y1 = frame.shape[0] - self.height

        # Обновляем модель с использованием линейной интерполяции
        # Сохраняем новое значение опорного кадра, полученное с помощью линейной интерполяции
        xtemp = self.eta * self.crop(frame, self.x1, self.y1, self.width, self.height) + (1 - self.eta) * self.x

        # Вырезаем новый кадр
        self.x = self.crop(frame, self.x1, self.y1, self.width, self.height)
        # Обучаем модель с новым кадром и получаем новое значение с помощью линейной интерполяции
        self.alphaf = (self.eta * self.training(self.x, self.y, self.sigma, self.lmbda)
                       + (1 - self.eta) * self.alphaf)  # Линейная интерполяция

        # Устанавливаем сохранённый опорный кадр
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
        # Вычисление циклической корреляции через FFT
        # np.fft.fft2(x1) и np.fft.fft2(x2) преобразуют двумерные фрагменты в частотную область.
        # np.conj(np.fft.fft2(x2)) берёт комплексное сопряжение преобразования x2, чтобы получить взаимную корреляцию
        # np.fft.ifft2(...) преобразует результат обратно в пространственную область.
        # np.fft.fftshift(...) центрирует результат корреляции, чтобы значения соответствовали физическим сдвигам.
        if self.our_fft:
            c = np.fft.fftshift(ifft_2d(fft_2d(x1) * np.conj(fft_2d(x2))))
        else:
            c = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(x1) * np.conj(np.fft.fft2(x2))))

        #  Вычисление евклидового расстояния
        d = (
                np.dot(np.conj(x1.flatten(order='K')), x1.flatten(order='K')) +
                np.dot(np.conj(x2.flatten(order='K')), x2.flatten(order='K')) -
                2 * c
        )
        # Преобразование расстояний в значения гауссового ядра
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / np.size(x1))
        # d нормализуется на количество элементов в ROI np.size(x1),
        # чтобы значения были независимы от размера фрагмента.
        # Применяется гауссова функция для перевода расстояний в значения ядра
        # Маленькие расстояния d дают большие значения k (близкие фрагменты).
        # Большие расстояния d дают маленькие значения k (дальние фрагменты).

        return k

    def training(self, x, y, sigma, lmbda):
        """
        Обучение модели корреляционного фильтра.
        Решение наименьших квадратов с регуляризацией ядра.

        :param x: Входной фрагмент.
        :param y: Карта отклика цели.
        :param sigma: Ширина ядра гауссовой функции.
        :param lmbda: Параметр регуляризации.
        :return: Преобразование Фурье обученной модели.
        """
        # находим k, которое описывает сходство каждого пикселя x с его циклическими сдвигами.
        k = self.dgk(x, x, sigma)

        # Карта отклика y преобразуется в частотную область через FFT
        # Матрица ядра k также преобразуется через FFT
        # Регуляризация добавляется к знаменателю для стабилизации обучения
        # Полученная модель описывает, как объект выглядит в частотной области
        if self.our_fft:
            alphaf = fft_2d(y) / (fft_2d(k) + lmbda)
        else:
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
        if self.our_fft:
            responses = np.real(ifft_2d(alphaf * fft_2d(k)))
        else:
            responses = np.real(np.fft.ifft2(alphaf * np.fft.fft2(k)))
        if self.debug:
            cv2.imshow('Gauss kernel', k)
            cv2.imshow('The reference fragment of the search', x)
            cv2.imshow('The current search fragment', z)
            cv2.imshow('Responses map', responses)
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

        # создаёт двумерные сетки индексов, где
        # J[i, j] - индекс j-ого столбца
        # I[i, j] - индекс i-ой строки
        J, I = np.meshgrid(j, i)

        # создание двумерного окна, которое применяет весы к пикселям, уменьшая влияние краёв.
        window = np.sin(np.pi * J / width) * np.sin(np.pi * I / height)
        # Перемножение создаёт двумерное окно, где значения близки к 0 по краям и достигают максимума в центре.
        # Наложение окна на нормализованное изображение
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

        #  Создаются массивы pad_y и pad_x для хранения значений отступов (вверх/вниз и влево/вправо),
        #  если объект выходит за пределы изображения.
        pad_y = [0, 0]
        pad_x = [0, 0]

        # Расчёт верхней границы области интереса
        # Если y1−height/2 (верхний край области) выходит за пределы кадра (<0), то граница устанавливается в 0
        if (y1 - height / 2) < 0:
            y_up = 0
            # Разница записывается в pad_y[0], чтобы дополнить вырезку отступом позже
            pad_y[0] = int(-(y1 - height / 2))
        else:
            y_up = int(y1 - height / 2)

        # Расчёт нижней границы области интереса, аналогичен предыдущему расчёту
        if (y1 + 3 * height / 2) > img.shape[0]:
            y_down = img.shape[0]
            pad_y[1] = int((y1 + 3 * height / 2) - img.shape[0])
        else:
            y_down = int(y1 + 3 * height / 2)

        # Определение границ вырезки по оси X
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

        # Вырезаем часть изображения
        cropped = img[y_up:y_down, x_left:x_right]
        # Приводим к оттенкам серого
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # Дополнить изображение, если объект выходит за границы
        # np.pad добавляет отступы: pad_y[0] и pad_y[1] добавляются сверху и снизу,
        # pad_x[0] и pad_x[1] добавляются слева и справа
        # Метод 'edge' повторяет значения пикселей на границе, чтобы не создавать резких переходов
        padded = np.pad(cropped, (pad_y, pad_x), 'edge')

        # Применение косинусного окна
        windowed = self.window(padded)
        return windowed

    def target(self, width: int, height: int) -> np.ndarray:
        """
        Генерация гауссовой формы карты отклика.

        :param width: Ширина области цели.
        :param height: Высота области цели.
        :return:
        """
        # Увеличиваем размеры карты отклика вдвое по ширине и высоте
        # Увеличенные размеры позволяют учитывать все возможные циклические сдвиги ROI в частотной области.
        # Это важно для работы алгоритма CSK, где используются свойства циркулянтных матриц.
        double_height = 2 * height
        double_width = 2 * width

        # Вычисление масштаба гауссового распределения
        s = np.sqrt(double_height * double_width) / 16

        # Создание сетки координат
        # np.arange(0, double_width) создаёт массив индексов по горизонтали j.
        j = np.arange(0, double_width)
        # np.arange(0, double_height) создаёт массив индексов по вертикали i.
        i = np.arange(0, double_height)
        # np.meshgrid(j, i) генерирует двумерные матрицы J и I
        # J[i, j] содержит координаты по оси X
        # I[i, j] содержит координаты по оси Y
        J, I = np.meshgrid(j, i)

        # Вычисление Гауссовой функции
        y = np.exp(-((J - width) ** 2 + (I - height) ** 2) / s ** 2)

        return y
