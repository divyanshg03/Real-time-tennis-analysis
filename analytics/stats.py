"""Physical stats (shot speed, player speed, serves, rallies, bounces) on the court plane.

Everything is computed through the court homography (meters), not through
height-based pixel scaling.

Ball speed model
----------------
The ball is airborne, so the ground-plane homography is only exact for points that
touch the ground. We therefore measure what we can anchor to the ground:

* ``hit_to_bounce``: hitter's feet at contact (the ball is ~1 m above them) to the
  bounce point (the ball is on the ground). The 3-D path length adds the assumed
  contact height. This is the *average* speed over the first part of the flight; the
  true speed off the racket is higher (drag, bounce slowdown), typically 10-20 %.
* ``hit_to_hit`` (fallback when no bounce was found): hitter's feet to the next
  hitter's feet over the time between hits. This underestimates speed because it
  includes the slow descent and the time spent after the bounce.

Treat speeds as estimates and validate them against radar data (see evaluation/).
"""
import numpy as np
import pandas as pd

import constants
from court_geometry import NET_Y, is_inside_court
from utils import get_foot_position, point_to_bbox_distance
from analytics.events import ball_centers, refine_bounce

ROLES = (1, 2)


def player_court_tracks(players, homography_for_frame, smooth_window=5):
    """Per-role (N, 2) court positions in meters (NaN where the player is missing), smoothed."""
    n = len(players)
    tracks = {r: np.full((n, 2), np.nan) for r in ROLES}
    for i, frame in enumerate(players):
        H = homography_for_frame(i)
        for r in ROLES:
            if r in frame:
                tracks[r][i] = H.pixel_to_court_point(get_foot_position(frame[r]))
    if smooth_window and smooth_window > 1:
        for r in ROLES:
            df = pd.DataFrame(tracks[r])
            tracks[r] = df.rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
    return tracks


def player_speeds_kmh(track, fps, max_speed_kmh):
    """Per-frame speed (km/h) from a smoothed court track; implausible values -> NaN."""
    step = np.linalg.norm(np.diff(track, axis=0), axis=1)  # meters per frame
    speed = np.concatenate([[np.nan], step * fps * 3.6])
    speed[speed > max_speed_kmh] = np.nan
    return speed


def _nearest_role(ball_center, frame_players):
    best, best_d = None, float("inf")
    for r, bbox in frame_players.items():
        d = point_to_bbox_distance(ball_center, bbox)
        if d < best_d:
            best, best_d = r, d
    return best


def compute_shots(events, ball_arr, players, homography_for_frame, fps, cfg):
    """Build the shots table, bounces table and per-frame player speed arrays."""
    tracks = player_court_tracks(players, homography_for_frame, cfg.player_smooth_window)
    speeds = {r: player_speeds_kmh(tracks[r], fps, cfg.max_player_speed_kmh) for r in ROLES}
    centers = ball_centers(ball_arr)
    n = len(ball_arr)

    # ---- bounces on the ground plane --------------------------------------
    bounce_rows = []
    for b in events.bounces:
        if np.isnan(ball_arr[b, 0]):
            continue
        t_exact, px, py = refine_bounce(ball_arr, b)
        # frame index used for the homography is the nearest frame; the point itself is sub-frame
        x, y = homography_for_frame(int(round(t_exact))).pixel_to_court_point((px, py))
        bounce_rows.append({
            "frame": b, "t_exact": t_exact, "time_s": t_exact / fps, "court_x_m": x, "court_y_m": y,
            "side": "far" if y < NET_Y else "near",
            # indicative only: +-10 cm covers the ball radius but not detection error
            "inside_singles_lines": is_inside_court((x, y), margin=0.1, singles=True),
            "inside_doubles_lines": is_inside_court((x, y), margin=0.1, singles=False),
        })
    bounces_df = pd.DataFrame(bounce_rows, columns=[
        "frame", "t_exact", "time_s", "court_x_m", "court_y_m", "side", "inside_singles_lines",
        "inside_doubles_lines"])

    # ---- shots ------------------------------------------------------------
    hits = list(events.hits)
    hitters = []
    for h in hits:
        c = centers[h]
        hitters.append(None if np.isnan(c[0]) else _nearest_role(c, players[h]))

    rows = []
    rally_id, rally_shot = 0, 0
    last_hit_frame = None
    for k, h in enumerate(hits):
        hitter = hitters[k]
        if hitter is None:
            continue
        next_hit = hits[k + 1] if k + 1 < len(hits) else None
        window_end = min(next_hit if next_hit is not None else n, h + int(cfg.max_flight_seconds * fps))
        p0 = tracks[hitter][h]
        t_hit = float(h)  # whole-frame: racket contacts are not reliably sub-frame localisable

        # serve / rally bookkeeping
        gap_s = (h - last_hit_frame) / fps if last_hit_frame is not None else None
        near_baseline = (not np.isnan(p0[0])) and (
            p0[1] <= 2.5 or p0[1] >= constants.COURT_LENGTH - 2.5)
        new_point = gap_s is None or gap_s > cfg.serve_gap_seconds
        is_serve = bool(new_point and near_baseline)
        if new_point:
            rally_id += 1
            rally_shot = 0
        rally_shot += 1
        last_hit_frame = h

        # ball speed
        speed, method, flight_s = np.nan, "none", np.nan
        bounce = next((b for b in bounce_rows if h + 2 <= b["frame"] < window_end), None)
        if bounce is not None and not np.isnan(p0[0]):
            d = float(np.hypot(bounce["court_x_m"] - p0[0], bounce["court_y_m"] - p0[1]))
            d3 = float(np.hypot(d, cfg.assumed_contact_height_m))
            flight_s = (bounce["t_exact"] - t_hit) / fps
            speed, method = d3 / flight_s * 3.6, "hit_to_bounce"
        elif next_hit is not None and hitters[k + 1] is not None:
            p1 = tracks[hitters[k + 1]][next_hit]
            if not np.isnan(p0[0]) and not np.isnan(p1[0]) and (next_hit - h) >= 2:
                d = float(np.hypot(*(p1 - p0)))
                flight_s = (next_hit - h) / fps
                speed, method = d / flight_s * 3.6, "hit_to_hit"
        rejected = False
        if not np.isnan(speed) and not (5.0 <= speed <= cfg.max_ball_speed_kmh):
            speed, method, rejected = np.nan, method + "_rejected", True

        # opponent running speed while the ball is in flight
        opponent = 2 if hitter == 1 else 1
        seg = speeds[opponent][h:window_end]
        opp_speed = float(np.nanmean(seg)) if np.isfinite(seg).any() else np.nan

        rows.append({
            "shot_index": len(rows), "frame": h, "t_exact": t_hit, "time_s": t_hit / fps, "hitter": hitter,
            "is_serve": is_serve, "rally_id": rally_id, "rally_shot_number": rally_shot,
            "hitter_court_x_m": p0[0], "hitter_court_y_m": p0[1],
            "ball_speed_kmh": speed, "speed_method": method, "flight_time_s": flight_s,
            "speed_rejected": rejected, "opponent_speed_kmh": opp_speed,
        })
    shots_df = pd.DataFrame(rows, columns=[
        "shot_index", "frame", "t_exact", "time_s", "hitter", "is_serve", "rally_id", "rally_shot_number",
        "hitter_court_x_m", "hitter_court_y_m", "ball_speed_kmh", "speed_method", "flight_time_s",
        "speed_rejected", "opponent_speed_kmh"])

    return {"shots": shots_df, "bounces": bounces_df, "tracks": tracks, "speeds": speeds}


def build_frame_stats(shots_df, n_frames):
    """Per-frame running stats for the HUD. NaN means "not available yet" (no divide-by-zero)."""
    count = {r: 0 for r in ROLES}
    shot_speeds = {r: [] for r in ROLES}
    run_speeds = {r: [] for r in ROLES}
    last_shot = {r: np.nan for r in ROLES}
    last_run = {r: np.nan for r in ROLES}

    def snapshot():
        out = {}
        for r in ROLES:
            out[f"player_{r}_number_of_shots"] = count[r]
            out[f"player_{r}_last_shot_speed"] = last_shot[r]
            out[f"player_{r}_avg_shot_speed"] = float(np.mean(shot_speeds[r])) if shot_speeds[r] else np.nan
            out[f"player_{r}_last_player_speed"] = last_run[r]
            out[f"player_{r}_avg_player_speed"] = float(np.mean(run_speeds[r])) if run_speeds[r] else np.nan
        return out

    snapshots = [(-1, snapshot())]  # state before any shot
    for _, s in shots_df.sort_values("frame").iterrows():
        h = int(s["hitter"])
        o = 2 if h == 1 else 1
        count[h] += 1
        if not np.isnan(s["ball_speed_kmh"]):
            shot_speeds[h].append(s["ball_speed_kmh"])
            last_shot[h] = s["ball_speed_kmh"]
        if not np.isnan(s["opponent_speed_kmh"]):
            run_speeds[o].append(s["opponent_speed_kmh"])
            last_run[o] = s["opponent_speed_kmh"]
        snapshots.append((int(s["frame"]), snapshot()))

    columns = list(snapshots[0][1].keys())
    data = {c: np.empty(n_frames) for c in columns}
    idx = 0
    for f in range(n_frames):
        while idx + 1 < len(snapshots) and snapshots[idx + 1][0] <= f:
            idx += 1
        for c in columns:
            data[c][f] = snapshots[idx][1][c]
    df = pd.DataFrame(data)
    df.insert(0, "frame_num", np.arange(n_frames))
    return df


def summarize(shots_df, speeds, tracks, fps, quality=None):
    """Per-player summary (JSON friendly)."""
    summary = {"fps": fps, "players": {}, "court_quality": quality or {}}
    ok = shots_df[shots_df["ball_speed_kmh"].notna()] if len(shots_df) else shots_df
    for r in ROLES:
        mine = shots_df[shots_df["hitter"] == r] if len(shots_df) else shots_df
        mine_ok = ok[ok["hitter"] == r] if len(ok) else ok
        serves = mine_ok[mine_ok["is_serve"]] if len(mine_ok) else mine_ok
        track = tracks[r]
        step = np.linalg.norm(np.diff(track, axis=0), axis=1)
        sp = speeds[r]
        valid_step = np.isfinite(step) & np.isfinite(sp[1:])
        summary["players"][f"player_{r}"] = {
            "side": "far" if r == 1 else "near",
            "shots": int(len(mine)),
            "serves": int(mine["is_serve"].sum()) if len(mine) else 0,
            "avg_shot_speed_kmh": _nanround(mine_ok["ball_speed_kmh"].mean()) if len(mine_ok) else None,
            "max_shot_speed_kmh": _nanround(mine_ok["ball_speed_kmh"].max()) if len(mine_ok) else None,
            "avg_serve_speed_kmh": _nanround(serves["ball_speed_kmh"].mean()) if len(serves) else None,
            "distance_covered_m": round(float(step[valid_step].sum()), 1),
            "top_running_speed_kmh": _nanround(np.nanmax(sp)) if np.isfinite(sp).any() else None,
        }
    summary["rallies"] = int(shots_df["rally_id"].nunique()) if len(shots_df) else 0
    summary["longest_rally_shots"] = int(shots_df.groupby("rally_id").size().max()) if len(shots_df) else 0
    return summary


def _nanround(v, nd=1):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)
