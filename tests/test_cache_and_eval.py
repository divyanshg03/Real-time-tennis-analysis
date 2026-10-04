import numpy as np

from evaluation.metrics import (ball_detection_metrics, event_metrics, idf1, keypoint_metrics,
                                speed_metrics)
from tests import synthetic as S
from utils.cache import load_cache, save_cache


def test_cache_invalidated_by_meta_change(tmp_path):
    p = str(tmp_path / "c.json")
    save_cache(p, {"video": "a", "model": "m1"}, [1, 2, 3])
    assert load_cache(p, {"video": "a", "model": "m1"}) == [1, 2, 3]
    assert load_cache(p, {"video": "b", "model": "m1"}) is None     # other video
    assert load_cache(p, {"video": "a", "model": "m2"}) is None     # other model
    assert load_cache(str(tmp_path / "missing.json"), {}) is None


def test_event_metrics():
    m = event_metrics([10, 31, 99], [10, 30, 60], tol_frames=2)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)


def test_speed_metrics():
    m = speed_metrics([(10, 100.0), (30, 90.0)], [(10, 110.0), (31, 90.0)])
    assert m["n"] == 2 and m["mae_kmh"] == 5.0 and m["bias_kmh"] == -5.0


def test_ball_metrics():
    gt = np.array([[0, 0, 10, 10], [np.nan] * 4, [50, 50, 60, 60]], float)
    pred = np.array([[1, 1, 11, 11], [100, 100, 110, 110], [np.nan] * 4], float)
    m = ball_detection_metrics(pred, gt)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)


def test_keypoint_metrics_perfect_and_shifted():
    kp = S.keypoints_flat()
    perfect = keypoint_metrics(kp, kp)
    assert perfect["mean_px"] < 1e-9 and perfect["court_plane_mean_m"] < 1e-3
    off = keypoint_metrics(kp + 10, kp)
    assert off["pck@5px"] == 0.0 and off["pck@20px"] == 1.0


def test_idf1_penalises_id_switch():
    box = [0, 0, 10, 10]
    gt = [{1: box}] * 10
    same = idf1([{7: box}] * 10, gt)["idf1"]
    switched = idf1([{7: box}] * 5 + [{8: box}] * 5, gt)["idf1"]
    assert same == 1.0 and switched < 0.8
