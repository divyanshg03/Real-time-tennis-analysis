import numpy as np

from trackers.ball_tracker import (detections_to_array, interpolate_gaps, reject_outliers,
                                   clean_ball_positions, suppress_static_detections)


def _track(n=40):
    return [{1: [100 + 10 * i, 200, 110 + 10 * i, 210]} for i in range(n)]


def test_isolated_false_positive_removed():
    dets = _track()
    dets[20] = {1: [1500, 900, 1510, 910]}
    arr = reject_outliers(detections_to_array(dets), max_jump_px=150)
    assert np.isnan(arr[20, 0])
    assert not np.isnan(arr[19, 0]) and not np.isnan(arr[21, 0])


def test_short_gap_filled_long_gap_left():
    arr = detections_to_array(_track(60))
    arr[10:13] = np.nan     # 3 frames
    arr[30:50] = np.nan     # 20 frames
    out = interpolate_gaps(arr, max_gap=12)
    assert not np.isnan(out[10:13]).any()
    assert np.isnan(out[30:50]).all()
    assert out[11, 0] == 100 + 10 * 11   # linear


def test_leading_and_trailing_gaps_not_invented():
    arr = detections_to_array(_track(30))
    arr[:4] = np.nan
    arr[-3:] = np.nan
    out = interpolate_gaps(arr, max_gap=12)
    assert np.isnan(out[:4]).all() and np.isnan(out[-3:]).all()


def test_clean_pipeline_handles_empty():
    out = clean_ball_positions([{}] * 10, (1920, 1080))
    assert np.isnan(out).all()


def test_recurring_static_false_positive_removed_but_held_ball_kept():
    dets = _track(80)
    for i in (5, 20, 33, 47, 60, 71):               # logo detected now and then at the same pixel
        dets[i] = {1: [1339, 423, 1349, 433]}
    arr = suppress_static_detections(detections_to_array(dets))
    for i in (5, 20, 33, 47, 60, 71):
        assert np.isnan(arr[i, 0])
    assert not np.isnan(arr[6, 0]) and not np.isnan(arr[19, 0])

    held = [{1: [500, 500, 510, 510]} for _ in range(40)]     # ball held still, one consecutive run
    assert not np.isnan(suppress_static_detections(detections_to_array(held))).any()
