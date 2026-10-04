import numpy as np

from analytics.shot_type import classify_shot, lateral_position


def pose(wrist_x, wrist_y=300.0, facing_away=True, right_wrist=True, nose_y=200.0, conf=0.9):
    """Fabricated COCO-17 pose. Shoulders at x=480/520; for a player seen from behind the right shoulder
    is on the image right, facing the camera it is on the image left."""
    k = np.zeros((17, 3))
    k[:, 2] = conf
    k[0] = (500, nose_y, conf)
    right_x, left_x = (520, 480) if facing_away else (480, 520)
    k[6] = (right_x, 250, conf)
    k[5] = (left_x, 250, conf)
    k[10 if right_wrist else 9] = (wrist_x, wrist_y, conf)
    return k


def test_forehand_wrist_on_dominant_side_seen_from_behind():
    # near player, back to camera: right side is image right
    assert classify_shot([pose(580)] * 3)[0] == "forehand"


def test_backhand_wrist_crossed_to_other_side_seen_from_behind():
    assert classify_shot([pose(420)] * 3)[0] == "backhand"


def test_side_flips_for_a_player_facing_the_camera():
    # far player: right shoulder is on the image left, so a wrist on the image left is a forehand
    assert classify_shot([pose(420, facing_away=False)] * 3)[0] == "forehand"
    assert classify_shot([pose(580, facing_away=False)] * 3)[0] == "backhand"


def test_left_hander_uses_the_left_wrist_and_flips_sides():
    k = pose(420, right_wrist=False)     # left wrist on the image left, seen from behind
    assert classify_shot([k] * 3, right_handed=False)[0] == "forehand"
    k = pose(580, right_wrist=False)
    assert classify_shot([k] * 3, right_handed=False)[0] == "backhand"


def test_serve_needs_the_flag_and_a_raised_wrist():
    assert classify_shot([pose(520, wrist_y=120)] * 3, is_serve=True)[0] == "serve"
    assert classify_shot([pose(520, wrist_y=120)] * 3)[0] == "overhead"       # no flag: a smash


def test_first_hit_of_a_clip_that_starts_mid_rally_is_not_a_serve():
    # flagged as a serve (first hit near the baseline) but the wrist is low at contact: a groundstroke
    assert classify_shot([pose(580)] * 3, is_serve=True)[0] == "forehand"
    assert classify_shot([pose(420)] * 3, is_serve=True)[0] == "backhand"


def test_serve_flag_alone_is_trusted_when_the_pose_is_unreadable():
    assert classify_shot([None, None], is_serve=True)[0] == "serve"


def test_median_ignores_one_bad_frame():
    seq = [pose(580), pose(420), pose(585), pose(575)]
    assert classify_shot(seq)[0] == "forehand"


def test_missing_or_low_confidence_pose_is_unknown_not_a_guess():
    assert classify_shot([None, None])[0] == "unknown"
    assert classify_shot([pose(580, conf=0.1)] * 3)[0] == "unknown"
    near_midline = classify_shot([pose(505)] * 3)
    assert near_midline[0] == "unknown"


def test_shoulders_edge_on_are_rejected():
    k = pose(580)
    k[6][0], k[5][0] = 502, 498      # 4 px apart: body turned edge-on
    assert lateral_position(k) is None
