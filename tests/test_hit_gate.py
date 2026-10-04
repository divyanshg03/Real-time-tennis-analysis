import numpy as np

import constants
import pipeline
from tests import synthetic as S
from tests.helpers import court_model_for

L = constants.COURT_LENGTH


def _gate_for_ball_at(distance_m):
    """Gate for a ball lying on the ground ``distance_m`` in front of the near player (frame 0)."""
    foot = S.player_pos(2, 0)
    ball = np.full((1, 4), np.nan)
    ball[0] = S.ball_box((foot[0], foot[1] - distance_m), 0.0)
    players = [{2: S.player_bbox(2, 0)}]
    return pipeline._court_distance_gate(ball, players, court_model_for(1), 6.5)


def test_gate_accepts_ball_near_player_and_rejects_far_ball():
    assert _gate_for_ball_at(4.0)(0) is True
    assert _gate_for_ball_at(9.0)(0) is False


def test_serve_like_window_allows_a_larger_distance():
    gate = _gate_for_ball_at(9.0)
    assert gate(0, serve_like=True) is True        # 9 m is within 6.5 * 1.6
    assert _gate_for_ball_at(12.0)(0, serve_like=True) is False


def test_gate_is_off_when_distance_is_zero():
    ball = np.full((1, 4), np.nan)
    assert pipeline._court_distance_gate(ball, [{2: S.player_bbox(2, 0)}], court_model_for(1), 0) is None


def test_missing_ball_is_not_a_hit():
    ball = np.full((1, 4), np.nan)
    gate = pipeline._court_distance_gate(ball, [{2: S.player_bbox(2, 0)}], court_model_for(1), 6.5)
    assert gate(0) is False
