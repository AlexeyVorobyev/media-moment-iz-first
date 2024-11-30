import csk
from TrackerApplication import TrackerApplication

app = TrackerApplication(
    tracker_name="CSK",
    tracker_factory=lambda: csk.CSK(),
)

video_to_process = [
    "resources/video1.mp4",
    "resources/video2.mp4",
    "resources/video3.mp4",
    "resources/video4.mp4",
    "resources/video5.mp4"
]

if __name__ == "__main__":
    for item in video_to_process:
        app.process(item, f"{item.split("/")[0]}/csk-l/{item.split("/")[1]}")
