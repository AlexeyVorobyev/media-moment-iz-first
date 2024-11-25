import cv2

from AbstractTracker import AbstractTracker
from TrackerApplication import TrackerApplication

class TrackerKCFAdapter(AbstractTracker):
    def __init__(self):
        self.tracker = cv2.TrackerKCF.create()

    def init(self, image, bounding_box):
        """
        Инициализация трекера KCF.
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



app = TrackerApplication(
    tracker_name="KCF",
    tracker_factory=lambda: TrackerKCFAdapter(),
)

video_to_process = [
    "resources/video1.mp4",
    "resources/video2.mp4",
    "resources/video3.mp4",
    "resources/video4.mp4",
    "resources/video5.mp4"
]

for item in video_to_process:
    app.process(item, f"{item.split("/")[0]}/kcf/{item.split("/")[1]}")
