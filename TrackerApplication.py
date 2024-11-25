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
            tracker_name: str = "EMPTY"
    ):
        self._tracker_factory = tracker_factory
        self._window_name = window_name
        self._window_size = window_size
        self._tracker_name = tracker_name
        self._roi = None

    def __get_video_config(self, ifstream: cv2.VideoCapture):
        return {
            "fourcc": int(ifstream.get(cv2.CAP_PROP_FOURCC)),
            "fps": int(ifstream.get(cv2.CAP_PROP_FPS)),
            "size": (int(ifstream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(ifstream.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        }

    def __draw_success(
            self,
            box: bounding_box_type,
            frame: np.ndarray,
    ):
        x, y, w, h = [int(item) for item in box]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 2)

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
            success, box = tracker.update(frame)

            if success:
                self.__draw_success(box, frame)

            else:
                cv2.putText(
                    frame,
                    'Tracking failed!',
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (100, 100, 255),
                    3,
                    cv2.LINE_AA
                )

                self._roi = None

                tracker = self._tracker_factory()
        else:
            cv2.putText(
                frame,
                'Press "s" to select object for tracking',
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 100, 100),
                3,
                cv2.LINE_AA
            )

        cv2.imshow(self._window_name, frame)

        if key == ord('s'):
            self._roi = cv2.selectROI(self._window_name, frame)
            tracker.init(frame, self._roi)
        elif key == ord('x'):
            self._roi = None

        if ofstream is not None:
            ofstream.write(frame)

        return True

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

        ofstream = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter.fourcc(*'mp4v'),
            video_config['fps'],
            video_config['size']
        ) if output_path is not None else None

        tracker = self._tracker_factory()

        while True:
            ok = self.__process_frame(tracker, video_config['fps'], ifstream, ofstream)

            if not ok:
                print("Video reading finished")
                break
