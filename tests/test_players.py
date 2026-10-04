import copy

from tests import synthetic as S
from trackers.player_tracker import PlayerTracker, interpolate_player_boxes


def _select(dets):
    H = S.homography()
    return PlayerTracker.select_court_players(dets, lambda i: H)


def test_ignores_umpire_and_ball_kid():
    sc = S.build_scene()
    out = _select(sc["players"])
    for i, f in enumerate(out):
        assert set(f) == {1, 2}
        assert abs(f[1][0] - sc["players"][i][101][0]) < 1e-6
        assert abs(f[2][0] - sc["players"][i][102][0]) < 1e-6


def test_survives_tracker_id_switch():
    sc = S.build_scene()
    dets = copy.deepcopy(sc["players"])
    for i in range(200, len(dets)):               # tracker relabels both players mid-video
        dets[i][501], dets[i][502] = dets[i].pop(101), dets[i].pop(102)
    out = _select(dets)
    assert all(set(f) == {1, 2} for f in out)


def test_short_dropout_interpolated_long_dropout_left_missing():
    sc = S.build_scene()
    dets = copy.deepcopy(sc["players"])
    for i in range(50, 55):
        dets[i].pop(101)
    for i in range(100, 160):
        dets[i].pop(102)
    out = _select(dets)
    assert all(1 in out[i] for i in range(50, 55))
    assert all(2 not in out[i] for i in range(105, 155))


def test_interpolate_boxes():
    boxes = [[0, 0, 10, 10], None, None, [30, 0, 40, 10]]
    out = interpolate_player_boxes(boxes, max_gap=5)
    assert out[1][0] == 10 and out[2][0] == 20
