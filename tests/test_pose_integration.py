import cv2
import numpy as np
import pandas as pd

from analytics.pose import add_shot_types
from tests.test_shot_type import pose


class FakePose:
    """Returns a fixed pose per hitter instead of running a model."""

    def __init__(self, by_bbox_x):
        self.by_bbox_x = by_bbox_x
        self.calls = 0

    def keypoints_for(self, frame, bbox):
        self.calls += 1
        return self.by_bbox_x[int(bbox[0])]


def _video(path, n=20):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for _ in range(n):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()


def test_shot_type_column_is_added_per_hitter(tmp_path):
    video = str(tmp_path / "v.avi")
    _video(video)
    shots = pd.DataFrame({"frame": [5, 12], "hitter": [2, 1], "is_serve": [False, False]})
    # role 2's box starts at x=10, role 1's at x=40; the fake pose depends on which box it is asked about
    players = [{2: [10, 0, 20, 40], 1: [40, 0, 50, 40]} for _ in range(20)]
    fake = FakePose({10: pose(580), 40: pose(420)})
    out = add_shot_types(shots, players, video, 20, fake)
    assert list(out["shot_type"]) == ["forehand", "backhand"]
    assert "shot_type" not in shots.columns          # the input table is not modified


def test_left_handed_role_is_flipped(tmp_path):
    video = str(tmp_path / "v.avi")
    _video(video)
    shots = pd.DataFrame({"frame": [5], "hitter": [2], "is_serve": [False]})
    players = [{2: [10, 0, 20, 40]} for _ in range(20)]
    k = pose(420, right_wrist=False)                 # left wrist, on the image left, seen from behind
    out = add_shot_types(shots, players, video, 20, FakePose({10: k}), right_handed=(True, False))
    assert list(out["shot_type"]) == ["forehand"]


def test_hits_near_the_clip_edges_do_not_fail(tmp_path):
    video = str(tmp_path / "v.avi")
    _video(video, n=6)
    shots = pd.DataFrame({"frame": [0, 5], "hitter": [2, 2], "is_serve": [False, False]})
    players = [{2: [10, 0, 20, 40]} for _ in range(6)]
    out = add_shot_types(shots, players, video, 6, FakePose({10: pose(580)}))
    assert len(out) == 2
