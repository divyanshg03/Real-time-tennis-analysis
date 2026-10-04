import cv2
import numpy as np

import live
from config import Config
from evaluation.metrics import event_metrics
from tests import synthetic as S
from trackers import BallTracker, PlayerTracker


class StreamPlayers(PlayerTracker):
    """Replays the synthetic detections one frame per call, like a detector would on a live stream."""

    def __init__(self, data, stride=1):
        super().__init__(None)
        self.data, self.i, self.stride = data, 0, stride   # stride: live mode calls it every Nth frame

    def detect_frame(self, frame):
        d = self.data[min(self.i * self.stride, len(self.data) - 1)]
        self.i += 1
        return d


class StreamBall(BallTracker):
    def __init__(self, data):
        super().__init__(None)
        self.data, self.i = data, 0

    def detect_frame(self, frame):
        d = self.data[min(self.i, len(self.data) - 1)]
        self.i += 1
        return d


def _write_video(path, n, fps=30.0, size=(S.W, S.H)):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    frame = np.full((size[1], size[0], 3), 90, np.uint8)
    for _ in range(n):
        w.write(frame)
    w.release()


def _run(tmp_path, output=None, **kw):
    stride = kw.get("player_every", 1)
    sc = S.build_scene(rallies=((30, 4),))
    video = str(tmp_path / "stream.avi")
    _write_video(video, sc["n_frames"])
    kp = S.keypoints_flat()
    lv = live.run_live(Config(), video, output=output,
                       player_tracker=StreamPlayers(sc["players"], stride),
                       ball_tracker=StreamBall(sc["ball"]),
                       court_predict=lambda img: kp, **kw)
    return sc, lv


def test_live_finds_the_hits_with_a_short_lag(tmp_path):
    sc, lv = _run(tmp_path)
    assert lv.frames_seen == sc["n_frames"]
    assert lv.analysis is not None
    found = lv.analysis["events"].hits
    truth = sc["truth"]["hits"]
    m = event_metrics(found, truth, 3)
    assert m["recall"] >= 0.75          # the last hit may still be waiting for later frames


def test_live_window_is_bounded(tmp_path):
    sc, lv = _run(tmp_path, window_seconds=3.0)
    assert len(lv.players) <= int(3.0 * 30)
    assert lv.analysis_len <= len(lv.players)


def test_live_writes_an_annotated_stream(tmp_path):
    out = str(tmp_path / "live.avi")
    sc, lv = _run(tmp_path, output=out, max_frames=90)
    cap = cv2.VideoCapture(out)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 90
    cap.release()


def test_live_fails_clearly_on_a_bad_source():
    try:
        live.open_source("does_not_exist.mp4")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_skipping_player_detection_keeps_the_hits_and_saves_calls(tmp_path):
    sc, every1 = _run(tmp_path)
    calls_full = every1.player_tracker.i
    sc, every3 = _run(tmp_path, player_every=3)
    assert every3.player_tracker.i < calls_full * 0.5
    m = event_metrics(every3.analysis["events"].hits, sc["truth"]["hits"], 3)
    assert m["recall"] >= 0.75
