import numpy as np

from court_line_detector.court_line_detector import aggregate_keypoints, estimate_court_model
from tests import synthetic as S


def test_aggregate_discards_bad_sample():
    good = S.keypoints_flat()
    samples = [good + np.random.default_rng(i).normal(0, 0.5, 28) for i in range(6)]
    samples.append(good + 80)                       # one wildly wrong prediction
    kp, used = aggregate_keypoints(samples)
    assert used == 6
    assert np.abs(kp - good).max() < 2


def test_estimate_with_fake_predictor_and_moving_camera():
    base = S.keypoints_flat()
    shifted = base + np.tile([40.0, 0.0], 14)       # camera pans 40 px after frame 100

    def read(idxs):
        return list(idxs)                           # the "image" is just its frame index

    def predict(i):
        return base if i < 100 else shifted

    static = estimate_court_model(predict, read, 200, 30.0, n_samples=5)
    assert len(static.segments) == 1
    moving = estimate_court_model(predict, read, 200, 30.0, segment_seconds=100 / 30.0)
    assert len(moving.segments) == 2
    assert np.allclose(moving.keypoints_for_frame(10), base, atol=1e-6)
    assert np.allclose(moving.keypoints_for_frame(150), shifted, atol=1e-6)
    assert moving.quality["mean_reproj_error_m"] < 1e-3
