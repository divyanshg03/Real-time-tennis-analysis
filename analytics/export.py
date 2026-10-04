"""CSV / JSON / heatmap export so results don't only live inside the rendered video."""
import json
import os

import cv2
import numpy as np

from court_geometry import COURT_KEYPOINTS_METERS

_W = float(COURT_KEYPOINTS_METERS[:, 0].max())
_L = float(COURT_KEYPOINTS_METERS[:, 1].max())


def export_tables(out_dir, shots_df, bounces_df, frame_stats_df, summary, config_dict=None):
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "shots": os.path.join(out_dir, "shots.csv"),
        "bounces": os.path.join(out_dir, "bounces.csv"),
        "frame_stats": os.path.join(out_dir, "frame_stats.csv"),
        "summary": os.path.join(out_dir, "summary.json"),
    }
    shots_df.to_csv(paths["shots"], index=False)
    bounces_df.to_csv(paths["bounces"], index=False)
    frame_stats_df.to_csv(paths["frame_stats"], index=False)
    payload = dict(summary)
    if config_dict is not None:
        payload["config"] = config_dict
    with open(paths["summary"], "w") as f:
        json.dump(payload, f, indent=2, default=float)
    return paths


def court_heatmap(track, px_per_m=30, blur_m=0.6):
    """Top-down occupancy heatmap image (BGR) of a (N, 2) court-meter track."""
    h, w = int(_L * px_per_m) + 1, int(_W * px_per_m) + 1
    grid = np.zeros((h, w), np.float32)
    pts = track[np.isfinite(track).all(axis=1)]
    for x, y in pts:
        gx, gy = int(round(x * px_per_m)), int(round(y * px_per_m))
        if 0 <= gx < w and 0 <= gy < h:
            grid[gy, gx] += 1
    k = int(blur_m * px_per_m) | 1
    grid = cv2.GaussianBlur(grid, (k, k), 0)
    if grid.max() > 0:
        grid = grid / grid.max()
    img = cv2.applyColorMap((grid * 255).astype(np.uint8), cv2.COLORMAP_JET)
    for a, b in [(0, 2), (4, 5), (6, 7), (1, 3), (0, 1), (2, 3), (8, 9), (10, 11), (12, 13)]:
        p, q = COURT_KEYPOINTS_METERS[a] * px_per_m, COURT_KEYPOINTS_METERS[b] * px_per_m
        cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), (255, 255, 255), 1)
    net_y = int(_L / 2 * px_per_m)
    cv2.line(img, (0, net_y), (w - 1, net_y), (255, 255, 255), 2)
    return img


def export_heatmaps(out_dir, tracks):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for role, track in tracks.items():
        path = os.path.join(out_dir, f"heatmap_player_{role}.png")
        cv2.imwrite(path, court_heatmap(track))
        paths.append(path)
    return paths
