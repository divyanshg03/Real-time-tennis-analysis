import numpy as np
import pytest

from court_geometry import CourtHomography, is_inside_court
from tests import synthetic as S


def test_roundtrip_is_exact():
    H = S.homography()
    for xy in [(1.0, 2.0), (5.5, 11.88), (10.0, 23.0)]:
        got = H.pixel_to_court_point(tuple(S.to_pix(xy)))
        assert got == pytest.approx(xy, abs=1e-3)
    assert H.is_plausible()


def test_keypoint_noise_gives_small_court_error():
    H = CourtHomography(S.keypoints_flat(noise_px=2.0, seed=3))
    truth = (4.0, 15.0)
    got = H.pixel_to_court_point(tuple(S.to_pix(truth)))
    assert np.hypot(got[0] - truth[0], got[1] - truth[1]) < 0.4


def test_garbage_keypoints_flagged_implausible():
    bad = np.random.default_rng(0).uniform(0, 1000, 28)
    try:
        H = CourtHomography(bad)
    except ValueError:
        return
    assert not H.is_plausible()


def test_wrong_keypoint_count_rejected():
    with pytest.raises(ValueError):
        CourtHomography(np.zeros(20))


def test_inside_court():
    assert is_inside_court((5, 10))
    assert not is_inside_court((12, 10))
    assert not is_inside_court((0.5, 10), singles=True)
    assert is_inside_court((0.5, 10), singles=False)
