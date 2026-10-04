import numpy as np
import pytest

from config import Config
from evaluation.metrics import event_metrics, speed_metrics
from tests import synthetic as S
from tests.helpers import run_scene, court_model_for
from utils.video_utils import VideoInfo
import pipeline


def test_clean_rally_events_and_speeds():
    sc, a, _, _ = run_scene()
    t = sc["truth"]
    assert event_metrics(a["events"].hits, t["hits"], 2)["f1"] == 1.0
    assert event_metrics(a["events"].bounces, t["bounces"], 1)["f1"] == 1.0
    shots = a["shots"]
    assert list(shots["hitter"]) == t["hitters"]
    assert shots["is_serve"].sum() == 2
    assert shots["rally_id"].nunique() == 2
    sp = speed_metrics(list(zip(shots["frame"], shots["ball_speed_kmh"])),
                       list(zip(t["hits"], t["speeds_kmh"])), 3)
    assert sp["n"] == 9
    assert sp["mape"] < 0.08          # +-1 frame of hit timing on a ~21 frame flight is ~5 %


def test_bounce_positions_and_in_out():
    _, a, _, _ = run_scene()
    b = a["bounces"]
    # the two final shots leave the court (x > 10.97) and must be flagged out
    assert (~b["inside_singles_lines"]).sum() == 2
    known = b[b["frame"] == 51].iloc[0]
    assert known["court_y_m"] == pytest.approx(6.3, abs=0.5)


def test_player_speed_and_distance():
    _, a, _, _ = run_scene()
    top = max(np.nanmax(a["speeds"][1]), np.nanmax(a["speeds"][2]))
    assert top == pytest.approx(11.3, abs=1.0)    # 2 m amplitude, 4 s period
    for p in ("player_1", "player_2"):
        assert a["summary"]["players"][p]["distance_covered_m"] > 30


def test_frame_stats_no_divide_by_zero():
    _, a, _, _ = run_scene()
    fs = a["frame_stats"]
    assert not np.isinf(fs.select_dtypes("number").to_numpy()).any()
    assert np.isnan(fs.loc[0, "player_1_avg_shot_speed"])        # nothing yet -> NaN, not inf
    assert fs["player_1_number_of_shots"].is_monotonic_increasing


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_robust_to_dropouts_false_positives_and_keypoint_noise(seed):
    sc, a, _, _ = run_scene(drop_prob=0.12, false_positives=6, seed=seed, kp_noise=1.5)
    t = sc["truth"]
    assert event_metrics(a["events"].hits, t["hits"], 3)["f1"] >= 0.85
    assert event_metrics(a["events"].bounces, t["bounces"], 2)["f1"] >= 0.85
    sp = speed_metrics(list(zip(a["shots"]["frame"], a["shots"]["ball_speed_kmh"])),
                       list(zip(t["hits"], t["speeds_kmh"])), 3)
    assert sp["mape"] < 0.12


def test_fps_scales_speed():
    """Same pixel motion at a different fps must give proportionally different speeds."""
    sc = S.build_scene()
    cm = court_model_for(sc["n_frames"])
    s30 = pipeline.analyse(Config(), VideoInfo(30.0, S.W, S.H, sc["n_frames"]), sc["players"], sc["ball"], cm)
    s24 = pipeline.analyse(Config(), VideoInfo(24.0, S.W, S.H, sc["n_frames"]), sc["players"], sc["ball"], cm)
    v30 = s30["shots"]["ball_speed_kmh"].median()
    v24 = s24["shots"]["ball_speed_kmh"].median()
    assert v24 == pytest.approx(v30 * 24 / 30, rel=0.05)


def test_empty_ball_track_does_not_crash():
    sc = S.build_scene()
    cm = court_model_for(sc["n_frames"])
    a = pipeline.analyse(Config(), VideoInfo(30.0, S.W, S.H, sc["n_frames"]), sc["players"],
                         [{}] * sc["n_frames"], cm)
    assert len(a["shots"]) == 0 and a["summary"]["rallies"] == 0
