import json
import os

import cv2
import numpy as np

import pipeline
from config import Config
from tests import synthetic as S
from tests.helpers import court_model_for
from trackers import BallTracker, PlayerTracker
from utils import get_video_info


class FakePlayers(PlayerTracker):
    def __init__(self, data):
        super().__init__(None)
        self.data, self.calls = data, 0

    def detect_frames(self, frames, cache_path=None, cache_meta=None, refresh=False):
        self.calls += 1
        for _ in frames:      # consume the stream like a real detector would
            pass
        return self.data


class FakeBall(BallTracker):
    def __init__(self, data):
        super().__init__(None)
        self.data = data

    def detect_frames(self, frames, cache_path=None, cache_meta=None, refresh=False):
        for _ in frames:
            pass
        return self.data


def _write_video(path, n, fps):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (S.W, S.H))
    frame = np.full((S.H, S.W, 3), 90, np.uint8)
    for _ in range(n):
        w.write(frame)
    w.release()


def test_full_pipeline_writes_video_and_tables(tmp_path):
    sc = S.build_scene(rallies=((30, 4),))
    n = sc["n_frames"]
    video = str(tmp_path / "clip.avi")
    _write_video(video, n, 30.0)
    assert get_video_info(video).fps == 30.0

    cfg = Config(input_video=video, output_video=str(tmp_path / "out" / "o.avi"),
                 output_dir=str(tmp_path / "out"), cache_dir=str(tmp_path / "cache"),
                 court_keypoints_json="")
    analysis = pipeline.run(cfg, player_tracker=FakePlayers(sc["players"]),
                            ball_tracker=FakeBall(sc["ball"]), court_model=court_model_for(n))

    out = tmp_path / "out"
    for name in ("o.avi", "shots.csv", "bounces.csv", "frame_stats.csv", "summary.json",
                 "heatmap_player_1.png", "heatmap_player_2.png"):
        assert (out / name).exists(), name

    cap = cv2.VideoCapture(str(out / "o.avi"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n
    assert cap.get(cv2.CAP_PROP_FPS) == 30.0      # fps comes from the input, not a constant
    cap.release()

    summary = json.loads((out / "summary.json").read_text())
    assert summary["fps"] == 30.0 and summary["rallies"] == 1
    assert len(analysis["shots"]) == 4


def test_rendered_frame_differs_from_input(tmp_path):
    sc = S.build_scene(rallies=((30, 3),))
    n = sc["n_frames"]
    video = str(tmp_path / "clip.avi")
    _write_video(video, n, 30.0)
    cfg = Config(input_video=video, output_video=str(tmp_path / "o.avi"), output_dir=str(tmp_path),
                 export_csv=False, export_heatmaps=False)
    pipeline.run(cfg, player_tracker=FakePlayers(sc["players"]), ball_tracker=FakeBall(sc["ball"]),
                 court_model=court_model_for(n))
    cap = cv2.VideoCapture(cfg.output_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 45)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (S.H, S.W)
    assert np.abs(frame.astype(int) - 90).sum() > 0     # overlays were drawn
    assert os.path.getsize(cfg.output_video) > 0
