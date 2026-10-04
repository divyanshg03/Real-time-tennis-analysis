from dataclasses import dataclass

import cv2


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def get_video_info(video_path, default_fps=24.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    info = VideoInfo(
        fps=float(fps) if fps and fps > 0 else float(default_fps),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def iter_video(video_path):
    """Yield frames one at a time (constant memory)."""
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def read_video(video_path):
    """Load every frame into memory. Fine for short clips; prefer ``iter_video``."""
    return list(iter_video(video_path))


class VideoSink:
    """Streaming video writer. Use as a context manager."""

    def __init__(self, path, fps, size, fourcc="MJPG"):
        self.path = path
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if not self.writer.isOpened():
            raise IOError(f"cannot open video writer for {path}")

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def save_video(output_video_frames, output_video_path, fps=24.0):
    if not output_video_frames:
        raise ValueError("no frames to save")
    h, w = output_video_frames[0].shape[:2]
    with VideoSink(output_video_path, fps, (w, h)) as sink:
        for frame in output_video_frames:
            sink.write(frame)


def read_frames_at(video_path, indices):
    """Random-access read of specific frame indices."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    finally:
        cap.release()
    return frames
