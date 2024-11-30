from typing import Callable

import numpy as np

from AbstractTracker import AbstractTracker, bounding_box_type
import cv2


class TrackerApplication:
    _tracker_factory: Callable[[], AbstractTracker]
    _window_name: str
    _window_size: (int, int)
    _roi: bounding_box_type
    _tracker_name: str

    def __init__(
            self,
            tracker_factory: Callable[[], AbstractTracker],
            window_name: str = "Tracker Application",
            window_size: (int, int) = (1024, 576),
            tracker_name: str = "EMPTY",
            debug: bool = False,
    ):
        self._tracker_factory = tracker_factory
        self._window_name = window_name
        self._window_size = window_size
        self._tracker_name = tracker_name
        self._roi = None
        self._debug = debug

    def __get_video_config(self, ifstream: cv2.VideoCapture):
        return {
            "fourcc": int(ifstream.get(cv2.CAP_PROP_FOURCC)),
            "fps": int(ifstream.get(cv2.CAP_PROP_FPS)),
            "size": (int(ifstream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(ifstream.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "duration": int(ifstream.get(cv2.CAP_PROP_FRAME_COUNT) / int(ifstream.get(cv2.CAP_PROP_FPS)))
        }

    def __display_text(self, text: str, frame: np.ndarray) -> None:
        cv2.putText(
            frame,
            text,
            (50, 100),
            cv2.QT_FONT_NORMAL,
            1,
            (250, 0, 250),
            2,
            cv2.LINE_AA
        )

    def __draw_success(
            self,
            box: bounding_box_type,
            frame: np.ndarray,
            frame_time: float,
    ):
        x, y, w, h = [int(item) for item in box]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 2)

        self.__display_text(
            f'Algorythm: {self._tracker_name}. Frametime: {round(frame_time * 1000)} ms ({int(1 / frame_time)} FPS)',
            frame
        )

    def __process_frame(
            self,
            tracker: AbstractTracker,
            fps: int,
            ifstream: cv2.VideoCapture,
            ofstream: cv2.VideoWriter = None,
    ) -> bool:
        """
        Метод обработки конкретного фрейма

        :param tracker: трэкер
        :param fps:
        :param ifstream: поток ввода
        :param ofstream: поток вывода
        :return: успешность чтения кадра
        """

        ok, frame = ifstream.read()

        delay = int(1000 / fps)

        key = cv2.waitKey(delay) & 0xFF

        if not ok or key == 27:
            return False

        if self._roi is not None:
            timer = cv2.getTickCount()
            success, box = tracker.update(frame)
            frame_time = (cv2.getTickCount() - timer) / cv2.getTickFrequency()

            if success:
                self.__draw_success(box, frame, frame_time)

            else:
                self.__display_text(
                    'Tracking failed!',
                    frame
                )

                self._roi = None

                tracker = self._tracker_factory()
                tracker.debug = self._debug
        else:
            self.__display_text(
                'Press "s" to select object for tracking',
                frame
            )

        cv2.imshow(self._window_name, frame)

        if key == ord('s'):
            self._roi = cv2.selectROI(self._window_name, frame)
            if self._tracker_name == "CSK":
                self._roi = self.__expand_to_power_of_two(self._roi)
                print(f"ROI Shape: {self._roi}")
            tracker.init(frame, self._roi)
        elif key == ord('x'):
            self._roi = None

        if ofstream is not None:
            ofstream.write(frame)

        return True

    def __expand_to_power_of_two(self, roi: bounding_box_type) -> bounding_box_type:
        """
        Расширяет размеры ROI до ближайших степеней двойки.

        :param roi: Исходный ROI в формате (x, y, w, h)
        :return: Новый ROI, расширенный до ближайших степеней двойки
        """
        x, y, w, h = roi

        def nearest_power_of_two(value: int) -> int:
            return 1 << (value - 1).bit_length()

        new_w = nearest_power_of_two(w)
        new_h = nearest_power_of_two(h)

        # Центрируем расширенную область вокруг исходной
        new_x = max(0, x - (new_w - w) // 2)
        new_y = max(0, y - (new_h - h) // 2)

        return new_x, new_y, new_w, new_h

    def process(
            self,
            input_path: str,
            output_path: str = None,
    ) -> None:
        """
        Метод запуска процесса распознавания на конкретном видео

        :param input_path: путь к файлу для чтения
        :param output_path: путь к файлу выхода (не обязателен)
        """

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, *self._window_size)

        ifstream = cv2.VideoCapture(input_path)
        video_config = self.__get_video_config(ifstream)

        fourcc = video_config["fourcc"]
        fourcc = (chr(fourcc & 0xff)
                  + chr((fourcc >> 8) & 0xff)
                  + chr((fourcc >> 16) & 0xff)
                  + chr((fourcc >> 24) & 0xff)
                  )

        print(
            f'{input_path} | '
            f'{fourcc} | '
            f'{video_config["size"][0]}x{video_config["size"][1]} | '
            f'{video_config["fps"]} fps | '
            f'{video_config["duration"]} seconds'
        )

        ofstream = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter.fourcc(*'mp4v'),
            video_config['fps'],
            video_config['size']
        ) if output_path is not None else None

        tracker = self._tracker_factory()
        tracker.debug = self._debug

        while True:
            ok = self.__process_frame(tracker, video_config['fps'], ifstream, ofstream)

            if not ok:
                print("Video reading finished")
                break
