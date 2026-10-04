"""Shot type from body pose around the moment of contact.

Keypoints follow the COCO-17 layout used by YOLO pose models: ``(x, y, confidence)`` per joint, with
left/right meaning the player's own left and right (a pose model infers this from body orientation, so
it holds for a player seen from behind as well as from the front).

Rule of thumb used here, for a right-handed player: at contact the racket wrist is on the right of the
body midline for a forehand and has crossed to the left for a backhand (a two-handed backhand included,
because the dominant wrist crosses over). Serves are taken from the serve flag; a racket wrist above the
nose is an overhead.

Limits: handedness has to be supplied (right-handed by default); volleys, slices and drop shots are not
separated from groundstrokes; and the result is only as good as the hit frame and the pose estimate,
which is poor for a small far-side player. Validate on your own labelled clips before relying on it.
"""
import numpy as np

NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 0, 5, 6, 9, 10


def lateral_position(kps, right_handed=True, min_conf=0.3, min_shoulder_px=8.0):
    """Racket wrist position along the shoulder line, in shoulder widths from the body midline.

    Positive means on the player's dominant (forehand) side. Returns ``(lateral, wrist_above_nose)`` or
    ``None`` when the joints needed are missing or the shoulders are too close together to be reliable
    (the body is turned edge-on to the camera).
    """
    if kps is None:
        return None
    r_sh, l_sh = kps[R_SHOULDER], kps[L_SHOULDER]
    wrist = kps[R_WRIST] if right_handed else kps[L_WRIST]
    if min(r_sh[2], l_sh[2], wrist[2]) < min_conf:
        return None
    axis = r_sh[:2] - l_sh[:2]
    width = float(np.linalg.norm(axis))
    if width < min_shoulder_px:
        return None
    mid = (r_sh[:2] + l_sh[:2]) / 2
    lateral = float(np.dot(wrist[:2] - mid, axis) / (width * width))
    if not right_handed:
        lateral = -lateral   # a left-hander's forehand side is the left shoulder side
    nose = kps[NOSE]
    above = bool(wrist[1] < nose[1]) if nose[2] >= min_conf else None
    return lateral, above


def classify_shot(keypoint_sequence, is_serve=False, right_handed=True, margin=0.15):
    """Classify one shot from the keypoints of the hitter over a few frames around contact.

    Returns ``(shot_type, detail)`` with shot_type one of serve, overhead, forehand, backhand, unknown.
    The median over the usable frames is used, so one bad pose estimate does not decide the result.
    """
    laterals, aboves = [], []
    for kps in keypoint_sequence:
        res = lateral_position(kps, right_handed)
        if res is None:
            continue
        laterals.append(res[0])
        if res[1] is not None:
            aboves.append(res[1])
    detail = {"frames_used": len(laterals), "frames_total": len(keypoint_sequence)}
    if is_serve:
        return "serve", detail
    if not laterals:
        return "unknown", detail
    lat = float(np.median(laterals))
    detail["lateral"] = round(lat, 2)
    if aboves and float(np.mean(aboves)) > 0.5:
        return "overhead", detail
    if lat > margin:
        return "forehand", detail
    if lat < -margin:
        return "backhand", detail
    return "unknown", detail
