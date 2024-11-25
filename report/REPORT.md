# Индивидуалное задание №1

1. Реализован класс абстрактного трекера

```python
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
```

2. Реализован обработчик видео, 
принимаюший в том числе дальнейшую реализацию абстрактного класса **AbstractTracker** в качестве аргумента конструктора.

Использован паттерн стратегия, для переиспользования кода.

```python
    def __init__(
            self,
            tracker_factory: Callable[[], AbstractTracker],
            window_name: str = "Tracker Application",
            window_size: (int, int) = (1024, 576),
            tracker_name: str = "EMPTY"
    ):
        self._tracker_factory = tracker_factory
        self._window_name = window_name
        self._window_size = window_size
        self._tracker_name = tracker_name
        self._roi = None
```

3. Реализован класс-адаптер над классом cv2.TrackerCSRT

Обработано 5 видео

```python
class TrackerCSRTAdapter(AbstractTracker):
    def __init__(self):
        self.tracker = cv2.TrackerCSRT.create()

    def init(self, image, bounding_box):
        """
        Инициализация трекера CSRT.
        """
        return self.tracker.init(image, bounding_box)

    def update(self, image):
        """
        Обновление положения объекта с использованием кадра.
        """
        try:
            return self.tracker.update(image)
        except:
            return False, None

```

4. Реализован класс для работы алгоритма-детектора CSK и обработано 5 видео.
> Алгоритм CSK использует ядра корреляции для отслеживания объектов в видеопоследовательности. 
> Он применяет плотное выборочное покрытие фрагментов изображения и использует быстрое преобразование Фурье 
> для эффективного вычисления корреляции.

```python
class CSK(AbstractTracker):
        def init(
            self,
            frame: np.ndarray,
            bounding_box: bounding_box_type
    ) -> bool: ...

    def update(self, frame: np.ndarray) -> (bool, bounding_box_type): ...
```