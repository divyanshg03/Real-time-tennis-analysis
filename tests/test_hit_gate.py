import cv2
import numpy as np
import pytest

import pipeline
from config import Config
from evaluation.metrics import event_metrics
from tests import synthetic as S
from tests.helpers import court_model_for, run_scene
from utils import get_foot_position


def _head_displacement():
    """Court-plane distance between the near player's feet and the projection of his head (frame 0)."""
    H = S.homography()
    b = S.player_bbox(2, 0)
    foot = np.array(H.pixel_to_court_point(get_foot_position(b)))
    head = np.array(H.pixel_to_court_point(((b[0] + b[2]) / 2, b[1])))
    return float(np.linalg.norm(head - foot))


def _gate_for_ball_at_ratio(ratio, max_ratio=1.1):
    """Gate for a ball on the ground ``ratio`` head-displacements in front of the near player (frame 0)."""
    foot = S.player_pos(2, 0)
    ball = np.full((1, 4), np.nan)
    ball[0] = S.ball_box((foot[0], foot[1] - ratio * _head_displacement()), 0.0)
    return pipeline._court_distance_gate(ball, [{2: S.player_bbox(2, 0)}], court_model_for(1), max_ratio)


def test_gate_accepts_a_ball_near_the_player_and_rejects_a_far_one():
    assert _gate_for_ball_at_ratio(0.6)(0) is True
    assert _gate_for_ball_at_ratio(1.8)(0) is False


def test_serve_window_allows_about_twice_the_distance():
    gate = _gate_for_ball_at_ratio(1.8)
    assert gate(0, serve_like=True) is True
    assert _gate_for_ball_at_ratio(3.0)(0, serve_like=True) is False


def test_gate_is_off_when_the_ratio_is_zero():
    ball = np.full((1, 4), np.nan)
    assert pipeline._court_distance_gate(ball, [{2: S.player_bbox(2, 0)}], court_model_for(1), 0) is None


def test_missing_ball_is_not_a_hit():
    ball = np.full((1, 4), np.nan)
    gate = pipeline._court_distance_gate(ball, [{2: S.player_bbox(2, 0)}], court_model_for(1), 1.1)
    assert gate(0) is False


def _camera(far_l, far_r, near_l, near_r):
    return cv2.getPerspectiveTransform(S._COURT, np.float32([far_l, far_r, near_l, near_r]))


@pytest.mark.parametrize("name,corners", [
    ("default", None),
    ("high camera", ((480, 150), (1440, 150), (330, 960), (1590, 960))),
    ("very high camera", ((380, 100), (1540, 100), (380, 1000), (1540, 1000))),
    ("close and wide", ((300, 260), (1620, 260), (-150, 1060), (2070, 1060))),
    ("low and flat", ((700, 330), (1220, 330), (200, 980), (1720, 980))),
])
def test_gate_costs_no_hits_across_camera_geometries(monkeypatch, name, corners):
    """A fixed distance in metres lost most hits on the close/wide and low cameras; the head-relative ratio must
    not lose any compared with running without the gate."""
    if corners is not None:
        monkeypatch.setattr(S, "H_COURT_TO_PIX", _camera(*corners))
    truth = None
    found = {}
    for label, ratio in (("gate", 1.1), ("no gate", 0)):
        sc, a, _, _ = run_scene(cfg=Config(max_hit_distance_ratio=ratio))
        truth = sc["truth"]["hits"]
        found[label] = event_metrics(a["events"].hits, truth, 3)
    assert found["gate"]["tp"] >= found["no gate"]["tp"], (name, found)
    assert found["gate"]["fp"] <= found["no gate"]["fp"], (name, found)
