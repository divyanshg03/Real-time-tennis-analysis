"""Hit and bounce detection from a ball trajectory.

The old detector counted sign flips of a 5-frame rolling mean over a hardcoded
25-frame window (24 fps assumed), could not tell hits from bounces and never looked at
the last ~30 frames. This version:

* works on contiguous valid segments (no invented data across long gaps),
* smooths with Savitzky-Golay and compares the mean vertical velocity before/after,
* scales every window with the real FPS,
* requires a hit to happen near a player (when player boxes are available), which is
  what separates a racket contact from a bounce,
* treats the remaining "ball falls, then rises" reversals as bounces,
* handles the start and end of the clip.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.signal import savgol_filter

from utils import point_to_bbox_distance


@dataclass
class Events:
    hits: List[int] = field(default_factory=list)
    bounces: List[int] = field(default_factory=list)


def ball_centers(ball_arr):
    """(N, 4) xyxy -> (N, 2) centers (NaN where missing)."""
    return np.column_stack([(ball_arr[:, 0] + ball_arr[:, 2]) / 2,
                            (ball_arr[:, 1] + ball_arr[:, 3]) / 2])


def valid_segments(mask, min_len):
    """Yield (start, end_exclusive) of runs of True in ``mask`` with length >= min_len."""
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_len:
            yield i, j
        i = j


def _velocity(y, window):
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = max(5, window)
    if window > len(y):
        window = len(y) if len(y) % 2 == 1 else len(y) - 1
    if window < 5:
        return y.copy(), np.gradient(y)
    smooth = savgol_filter(y, window, 2)
    vel = savgol_filter(y, window, 2, deriv=1)
    return smooth, vel


def _hit_changes(vel_xy, half_window, v_min):
    """Candidate racket contacts: a sudden change of the mean 2-D velocity vector.

    Compares the mean velocity over ``half_window`` frames before and after each frame.
    Any strong change qualifies (direction reversal, speed-up from a toss, glancing
    cross-court change); whether it is a *hit* or a *bounce* is decided by where the ball
    is relative to the players. Returns list of (index, score).
    """
    n = len(vel_xy)
    out = []
    edge = max(2, half_window // 2)
    for i in range(1, n - 1):
        lo = max(0, i - half_window)
        hi = min(n, i + 1 + half_window)
        if i - lo < edge or hi - (i + 1) < edge:
            continue
        change = float(np.hypot(*(vel_xy[i + 1:hi].mean(axis=0) - vel_xy[lo:i].mean(axis=0))))
        if change >= 3.0 * v_min:
            out.append((i, change))
    return out


def _bounce_kinks(vy, half_window, v_min):
    """Candidate bounces: sudden *upward* kick in the image-space vertical velocity.

    Ground motion dominates vy, so a bounce rarely flips its sign; what it always does is
    make vy drop abruptly (falling -> rising adds a large negative step) while gravity only
    adds a small positive drift. Returns list of (index, score).
    """
    n = len(vy)
    edge = max(2, half_window)
    delta = np.full(n, np.nan)
    for i in range(edge, n - edge):
        delta[i] = vy[i + 1:i + 1 + half_window].mean() - vy[max(0, i - half_window):i].mean()
    finite = delta[np.isfinite(delta)]
    if len(finite) < 5:
        return []
    # Perspective + gravity give delta a positive baseline; measure the drop against it.
    base = float(np.median(finite))
    mad = 1.4826 * float(np.median(np.abs(finite - base)))
    thr = max(2.0 * v_min, 3.0 * mad)
    drop = base - delta
    out = []
    for i in range(edge, n - edge):
        d = drop[i]
        if d > thr and d >= np.nanmax(drop[max(0, i - half_window):i + half_window + 1]):
            out.append((i, float(d)))
    return out


def _non_max_suppress(cands, min_sep):
    """cands: list of (frame, score). Keep the strongest within ``min_sep`` frames of each other."""
    kept = []
    for frame, score in sorted(cands, key=lambda c: -c[1]):
        if all(abs(frame - k) >= min_sep for k, _ in kept):
            kept.append((frame, score))
    return sorted(k for k, _ in kept)


def detect_events(ball_arr, player_detections, fps, frame_height,
                  hit_window_seconds=0.25, min_hit_separation_seconds=0.5, hit_gate=None):
    """Detect racket hits and ground bounces.

    ball_arr: (N, 4) ball boxes with NaN holes.
    player_detections: per-frame ``{role: bbox}`` (or None to skip the contact-zone test).
    hit_gate: optional ``(frame, serve_like) -> bool``; a hit candidate is dropped when it returns False.
    The pipeline uses it to require the ball to be near a player on the court plane, which rejects a bounce
    that happens in front of a player (inside his image-space contact zone but metres away from him).
    ``serve_like`` is True within the first ~1.2 s of a continuous ball track, where a serve is struck
    high above the head and the ball projects further from the server.
    """
    n = len(ball_arr)
    centers = ball_centers(ball_arr)
    valid = ~np.isnan(centers[:, 1])

    w_hit = max(3, int(round(hit_window_seconds * fps)))
    w_bounce = max(2, int(round(0.1 * fps)))
    sep_hit = max(2, int(round(min_hit_separation_seconds * fps)))
    # a threshold in px/frame that scales with resolution and (inversely) with fps
    v_min = max(0.8, 0.003 * frame_height * 30.0 / fps)
    sg_window = max(5, int(round(0.15 * fps)) | 1)

    hit_cands, bounce_cands = [], []
    serve_window = int(round(1.2 * fps))   # a serve is struck within ~1 s of the toss appearing
    for start, end in valid_segments(valid, min_len=2 * w_bounce + 3):
        seg = centers[start:end]
        _, vy = _velocity(seg[:, 1], sg_window)
        _, vx = _velocity(seg[:, 0], sg_window)

        for i, score in _hit_changes(np.column_stack([vx, vy]), w_hit, v_min):
            frame = start + i
            if player_detections is not None and not _in_contact_zone(centers[frame], player_detections[frame]):
                continue
            if hit_gate is not None and not hit_gate(frame, (frame - start) <= serve_window):
                continue
            hit_cands.append((frame, score))

        for i, score in _bounce_kinks(vy, w_bounce, v_min):
            bounce_cands.append((start + i, score))

    hits = _non_max_suppress(hit_cands, sep_hit)
    bounces = _non_max_suppress(bounce_cands, max(3, int(round(0.25 * fps))))
    # a velocity kink right at a racket contact is the hit, not a bounce
    bounces = [b for b in bounces if all(abs(b - h) > w_bounce + 1 for h in hits)]
    return Events(hits=hits, bounces=bounces)


def _in_contact_zone(ball_center, players):
    """Is the ball where a racket could be: around the upper body/arm reach of a player?

    The zone is the player's box, widened sideways by 0.6x its height (racket reach), and
    spanning from 0.6x height above the head (serve contact is ~2.7 m up) down to 85 % of the way to the feet. A ball
    at foot level is bouncing, not being struck.
    """
    if not players:
        return False
    bx, by = ball_center
    for bbox in players.values():
        h = bbox[3] - bbox[1]
        if h <= 0:
            continue
        if (bbox[0] - 0.6 * h <= bx <= bbox[2] + 0.6 * h) and (bbox[1] - 0.6 * h <= by <= bbox[1] + 0.85 * h):
            return True
    return False


def refine_kink(points, frame, k=4, iters=3):
    """Sub-frame time of a *sharp* trajectory kink such as a bounce.

    Fits a line to the ``k`` points up to and including ``frame`` and another to the ``k``
    points from ``frame`` onward, and returns the time where the two lines are closest plus
    the point there: ``(t, x, y)``. Re-centres and repeats so a 1-frame initial error does
    not contaminate the fit. At 30 fps a fast ball moves ~1 m per frame, so snapping a bounce
    to a whole frame alone would cost up to ~1 m of position error.

    Only used for bounces. Racket contacts (especially on the far side, where perspective
    squeezes the image motion) are smooth in the image and are not refined this way.
    Falls back to ``frame`` when there is not enough clean data.
    """
    n = len(points)
    f = int(frame)
    best = None
    for _ in range(iters):
        pre = np.arange(f - k, f + 1)
        post = np.arange(f, f + k + 1)
        if pre[0] < 0 or post[-1] >= n or np.isnan(points[np.r_[pre, post]]).any():
            break
        Apre = np.polyfit(pre.astype(float), points[pre], 1)    # rows: slope, intercept; cols: x, y
        Apost = np.polyfit(post.astype(float), points[post], 1)
        da, dc = Apre[0] - Apost[0], Apre[1] - Apost[1]
        denom = float(da @ da)
        if denom < 1e-9:
            break
        t = float(-(da @ dc) / denom)
        if not (f - 3 <= t <= f + 3):
            break
        best = (t, float(np.polyval(Apre[:, 0], t)), float(np.polyval(Apre[:, 1], t)))
        f = int(round(t))
    if best is None:
        return (float(frame), float(points[frame, 0]), float(points[frame, 1]))
    return best


def refine_bounce(ball_arr, frame, k=4):
    """Bounce location: the kink of the ball's bottom-centre, the point that touches the ground."""
    bottom = np.column_stack([(ball_arr[:, 0] + ball_arr[:, 2]) / 2, ball_arr[:, 3]])
    return refine_kink(bottom, frame, k)
