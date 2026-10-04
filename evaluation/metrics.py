"""Accuracy metrics. Without ground truth "better accuracy" cannot be measured, so every
component of the pipeline has a metric here:

* ball detection      -> precision / recall / mean pixel error
* hit & bounce events -> F1 with a frame tolerance (+ mean frame error)
* shot speeds         -> MAE / MAPE / bias against radar or known speeds
* court keypoints     -> pixel error, PCK, and error on the court plane in meters
* player tracking     -> IDF1 (identity-aware) with IoU matching
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from court_geometry import COURT_KEYPOINTS_METERS, CourtHomography, keypoints_to_array


# ---- ball detection ------------------------------------------------------
def ball_detection_metrics(pred, gt, tol_px=10.0):
    """pred, gt: (N, 4) xyxy arrays with NaN where there is no ball.

    A prediction is a true positive if its center is within ``tol_px`` of the ground truth.
    """
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    has_p, has_g = ~np.isnan(pred[:, 0]), ~np.isnan(gt[:, 0])
    pc = np.column_stack([(pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2])
    gc = np.column_stack([(gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2])
    both = has_p & has_g
    dist = np.full(len(pred), np.nan)
    dist[both] = np.linalg.norm(pc[both] - gc[both], axis=1)
    tp = int((dist <= tol_px).sum())
    fp = int(has_p.sum()) - tp
    fn = int(has_g.sum()) - tp
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "mean_error_px": float(np.nanmean(dist[dist <= tol_px])) if tp else float("nan")}


# ---- events --------------------------------------------------------------
def event_metrics(pred_frames, gt_frames, tol_frames=3):
    """One-to-one greedy matching of predicted to true event frames within ``tol_frames``."""
    pred, gt = sorted(pred_frames), sorted(gt_frames)
    used, errors = set(), []
    for g in gt:
        best, best_d = None, None
        for j, p in enumerate(pred):
            d = abs(p - g)
            if j not in used and d <= tol_frames and (best is None or d < best_d):
                best, best_d = j, d
        if best is not None:
            used.add(best)
            errors.append(pred[best] - g)
    tp = len(errors)
    fp, fn = len(pred) - tp, len(gt) - tp
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "mean_frame_error": float(np.mean(errors)) if errors else float("nan"),
            "mean_abs_frame_error": float(np.mean(np.abs(errors))) if errors else float("nan")}


# ---- speeds --------------------------------------------------------------
def speed_metrics(pred, gt, tol_frames=5):
    """pred, gt: lists of (frame, km/h). Matches by nearest frame within ``tol_frames``."""
    pairs = []
    used = set()
    for gf, gv in gt:
        cands = [(abs(pf - gf), j) for j, (pf, pv) in enumerate(pred)
                 if j not in used and abs(pf - gf) <= tol_frames and pv == pv]
        if cands:
            _, j = min(cands)
            used.add(j)
            pairs.append((pred[j][1], gv))
    if not pairs:
        return {"n": 0, "mae_kmh": float("nan"), "mape": float("nan"), "bias_kmh": float("nan")}
    p, g = np.array(pairs).T
    return {"n": len(pairs), "mae_kmh": float(np.mean(np.abs(p - g))),
            "mape": float(np.mean(np.abs(p - g) / g)), "bias_kmh": float(np.mean(p - g))}


# ---- court keypoints -----------------------------------------------------
def keypoint_metrics(pred, gt, pck_px=(5, 10, 20)):
    """pred, gt: flat 28-number keypoint arrays (pixels).

    Returns pixel errors, PCK at several thresholds, and the error on the court plane in
    meters: the *predicted* homography is applied to the *true* keypoints and compared
    with the known court coordinates (this is what actually affects speeds and positions).
    """
    p, g = keypoints_to_array(pred), keypoints_to_array(gt)
    err = np.linalg.norm(p - g, axis=1)
    out = {"mean_px": float(err.mean()), "max_px": float(err.max())}
    for t in pck_px:
        out[f"pck@{t}px"] = float((err <= t).mean())
    H = CourtHomography(p)
    plane = np.linalg.norm(H.pixel_to_court(g) - COURT_KEYPOINTS_METERS, axis=1)
    out["court_plane_mean_m"] = float(plane.mean())
    out["court_plane_max_m"] = float(plane.max())
    return out


# ---- player tracking -----------------------------------------------------
def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def idf1(pred_frames, gt_frames, iou_thr=0.5):
    """Identity F1. Each argument is a per-frame list of ``{id: bbox}``."""
    gt_ids = sorted({i for f in gt_frames for i in f})
    pr_ids = sorted({i for f in pred_frames for i in f})
    if not gt_ids or not pr_ids:
        return {"idf1": 0.0, "idp": 0.0, "idr": 0.0}
    gi = {k: n for n, k in enumerate(gt_ids)}
    pi = {k: n for n, k in enumerate(pr_ids)}
    overlap = np.zeros((len(gt_ids), len(pr_ids)))
    for pf, gf in zip(pred_frames, gt_frames):
        for g, gb in gf.items():
            for p, pb in pf.items():
                if _iou(gb, pb) >= iou_thr:
                    overlap[gi[g], pi[p]] += 1
    total_gt = sum(len(f) for f in gt_frames)
    total_pr = sum(len(f) for f in pred_frames)
    rows, cols = linear_sum_assignment(-overlap)
    idtp = float(overlap[rows, cols].sum())
    idfn, idfp = total_gt - idtp, total_pr - idtp
    return {"idf1": 2 * idtp / (2 * idtp + idfn + idfp) if idtp else 0.0,
            "idp": idtp / total_pr if total_pr else 0.0,
            "idr": idtp / total_gt if total_gt else 0.0}
