"""Synthetic rally with known ground truth, used to test the analytics without models/video."""
import cv2
import numpy as np

import constants
from court_geometry import COURT_KEYPOINTS_METERS, CourtHomography

W, H = 1920, 1080
FPS = 30.0
L, CW = constants.COURT_LENGTH, constants.DOUBLE_LINE_WIDTH

# camera: the 4 doubles corners land on these pixels (perspective, far court is smaller)
_PIX = np.float32([[560, 220], [1360, 220], [260, 960], [1660, 960]])
_COURT = np.float32([[0, 0], [CW, 0], [0, L], [CW, L]])
H_COURT_TO_PIX = cv2.getPerspectiveTransform(_COURT, _PIX)


def to_pix(xy):
    pt = np.float32([[xy]])
    return cv2.perspectiveTransform(pt, H_COURT_TO_PIX)[0, 0]


def scale_px_per_m(xy):
    a, b = to_pix(xy), to_pix((xy[0] + 1.0, xy[1]))
    return float(np.linalg.norm(b - a))


def keypoints_flat(noise_px=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.array([to_pix(p) for p in COURT_KEYPOINTS_METERS], dtype=np.float64)
    pts += rng.normal(0, noise_px, pts.shape) if noise_px else 0
    return pts.ravel()


def player_pos(role, frame, amp=2.0, period_s=4.0):
    """Far player (role 1) behind the far baseline, near player (role 2) behind the near one."""
    base_y = -1.5 if role == 1 else L + 1.5
    x = CW / 2 + amp * np.sin(2 * np.pi * (frame / FPS) / period_s + (0 if role == 1 else 1.0))
    return np.array([x, base_y])


def player_bbox(role, frame):
    foot = player_pos(role, frame)
    px = to_pix(tuple(foot))
    s = scale_px_per_m(tuple(foot))
    return [px[0] - 0.3 * s, px[1] - 1.85 * s, px[0] + 0.3 * s, px[1]]


def ball_box(ground_xy, height_m):
    px = to_pix(tuple(ground_xy))
    s = scale_px_per_m(tuple(ground_xy))
    bottom_y = px[1] - height_m * s
    r = 6
    return [px[0] - r, bottom_y - 2 * r, px[0] + r, bottom_y]


def build_scene(rallies=((30, 5), (400, 4)), flight_frames=30, bounce_frac=0.7,
                contact_h=1.0, drop_prob=0.0, false_positives=0, seed=1):
    """Returns dict with per-frame players/ball detections and ground truth.

    Each rally is ``(first_hit_frame, n_hits)``; hitters alternate starting with the near
    player; the ball flies hitter -> bounce -> next hitter, and after the final hit it flies
    to the other side and bounces once.
    """
    rng = np.random.default_rng(seed)
    n_frames = max(r[0] + r[1] * flight_frames + flight_frames for r in rallies) + 40
    ball = [None] * n_frames
    truth = {"hits": [], "bounces": [], "hitters": [], "speeds_kmh": [], "serves": []}

    for first, n_hits in rallies:
        hit_frames = [first + k * flight_frames for k in range(n_hits)]
        hitters = [2 if k % 2 == 0 else 1 for k in range(n_hits)]
        # toss before the serve: released at 1.6 m, apex 2.8 m after 15 frames (ballistic), contact at 2.6 m
        server = player_pos(hitters[0], first)
        for t in range(15):
            f = first - 15 + t
            u = t / 14.0
            ball[f] = ball_box(server, 1.6 + 1.2 * (1 - (1 - u) ** 2))
        for k in range(n_hits + 1):
            f0 = hit_frames[k] if k < n_hits else hit_frames[-1] + flight_frames
            if k == n_hits:
                break
            hitter_role = hitters[k]
            p0 = player_pos(hitter_role, f0)
            if k + 1 < n_hits:
                f1, p1 = hit_frames[k + 1], player_pos(hitters[k + 1], hit_frames[k + 1])
            else:  # final shot lands on the other side and runs out of the picture, nobody hits it
                f1 = f0 + flight_frames
                p1 = np.array([CW + 5.0, -8.0 if hitter_role == 2 else L + 8.0])
            fb = f0 + int(round(bounce_frac * (f1 - f0)))
            h0 = 2.6 if k == 0 else contact_h
            f_last = f1 if k + 1 < n_hits else fb + 8
            for f in range(f0, f_last + 1):
                s = (f - f0) / (f1 - f0)
                ground = p0 + s * (p1 - p0)
                if f <= fb:
                    u = (f - f0) / (fb - f0)
                    hgt = h0 * (1 - u) + 1.2 * 4 * u * (1 - u)
                else:
                    u = (f - fb) / (f1 - fb)
                    hgt = contact_h * u + 0.8 * 4 * u * (1 - u)
                ball[f] = ball_box(ground, hgt)
            truth["hits"].append(f0)
            truth["hitters"].append(hitter_role)
            truth["bounces"].append(fb)
            truth["serves"].append(k == 0)
            bounce_ground = p0 + (fb - f0) / (f1 - f0) * (p1 - p0)
            d = float(np.linalg.norm(bounce_ground - p0))
            truth["speeds_kmh"].append(np.hypot(d, h0) / ((fb - f0) / FPS) * 3.6)

    ball_dets = []
    for f in range(n_frames):
        if ball[f] is None or rng.random() < drop_prob:
            ball_dets.append({})
        else:
            ball_dets.append({1: list(ball[f]) + [0.9]})
    for _ in range(false_positives):
        f = int(rng.integers(0, n_frames))
        x, y = float(rng.uniform(100, W - 100)), float(rng.uniform(100, H - 100))
        ball_dets[f] = {1: [x, y, x + 12, y + 12, 0.3]}

    players = []
    for f in range(n_frames):
        d = {101: player_bbox(1, f), 102: player_bbox(2, f)}
        # an umpire standing beside the net post and a ball kid behind the court, both outside the lines
        d[900] = [to_pix((-3.0, L / 2))[0] - 25, to_pix((-3.0, L / 2))[1] - 130,
                  to_pix((-3.0, L / 2))[0] + 25, to_pix((-3.0, L / 2))[1]]
        d[901] = [to_pix((CW + 1.0, L + 9.0))[0] - 20, to_pix((CW + 1.0, L + 9.0))[1] - 110,
                  to_pix((CW + 1.0, L + 9.0))[0] + 20, to_pix((CW + 1.0, L + 9.0))[1]]
        players.append(d)

    return {"n_frames": n_frames, "ball": ball_dets, "players": players, "truth": truth}


def homography():
    return CourtHomography(keypoints_flat())
