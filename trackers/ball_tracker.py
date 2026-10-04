import cv2
import numpy as np

from utils import load_cache, save_cache


def detections_to_array(ball_detections):
    """List of {1: bbox} / {} -> (N, 4) float array, NaN where the ball is missing."""
    arr = np.full((len(ball_detections), 4), np.nan)
    for i, det in enumerate(ball_detections):
        box = det.get(1)
        if box is not None and len(box) >= 4:
            arr[i] = box[:4]
    return arr


def array_to_detections(arr):
    return [{} if np.isnan(row).any() else {1: row.tolist()} for row in arr]


def reject_outliers(arr, max_jump_px, max_passes=3):
    """Drop isolated false positives.

    A detection is rejected when it is further from *both* its previous and next
    valid detection than the ball could plausibly travel (``max_jump_px`` per
    frame, scaled by the frame gap). Endpoints are checked against their single
    neighbour. Returns a copy with rejected rows set to NaN.
    """
    arr = arr.copy()
    for _ in range(max_passes):
        valid = np.where(~np.isnan(arr[:, 0]))[0]
        if len(valid) < 3:
            break
        centers = np.column_stack([(arr[valid, 0] + arr[valid, 2]) / 2,
                                   (arr[valid, 1] + arr[valid, 3]) / 2])
        gaps = np.diff(valid)
        speeds = np.linalg.norm(np.diff(centers, axis=0), axis=1) / gaps  # px/frame
        too_fast = speeds > max_jump_px
        reject = []
        for k in range(len(valid)):
            prev_bad = too_fast[k - 1] if k > 0 else None
            next_bad = too_fast[k] if k < len(valid) - 1 else None
            flags = [f for f in (prev_bad, next_bad) if f is not None]
            if flags and all(flags):
                reject.append(valid[k])
        if not reject:
            break
        arr[reject] = np.nan
    return arr


def suppress_static_detections(arr, radius_px=6.0, min_runs=3, run_gap=5):
    """Remove detections that keep reappearing at the same pixel (a logo, a net post, a sign).

    A real ball cannot return to exactly the same spot in several separate stretches of the
    video, whereas a static false positive does. A ball held still in *one* consecutive run
    (before a serve) is kept. Returns a copy with those rows set to NaN.
    """
    from scipy.spatial import cKDTree

    arr = arr.copy()
    valid = np.where(~np.isnan(arr[:, 0]))[0]
    if len(valid) < min_runs:
        return arr
    centers = np.column_stack([(arr[valid, 0] + arr[valid, 2]) / 2, (arr[valid, 1] + arr[valid, 3]) / 2])
    tree = cKDTree(centers)
    flagged = set()
    for k, neigh in enumerate(tree.query_ball_point(centers, r=radius_px)):
        if k in flagged or len(neigh) < min_runs:
            continue
        frames = np.sort(valid[neigh])
        runs = 1 + int((np.diff(frames) > run_gap).sum())
        if runs >= min_runs:
            flagged.update(neigh)
    if flagged:
        arr[valid[sorted(flagged)]] = np.nan
    return arr


def interpolate_gaps(arr, max_gap):
    """Linearly fill NaN runs no longer than ``max_gap`` that have valid data on both sides.

    Longer gaps and leading/trailing gaps stay NaN: we don't invent trajectories.
    """
    arr = arr.copy()
    n = len(arr)
    isnan = np.isnan(arr[:, 0])
    i = 0
    while i < n:
        if not isnan[i]:
            i += 1
            continue
        j = i
        while j < n and isnan[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            for c in range(4):
                arr[i:j, c] = np.interp(np.arange(i, j), [i - 1, j], [arr[i - 1, c], arr[j, c]])
        i = j
    return arr


def clean_ball_positions(ball_detections, frame_size, max_jump_frac=0.12, max_gap_frames=12):
    """Outlier rejection followed by bounded interpolation. Returns an (N, 4) array with NaN holes."""
    w, h = frame_size
    arr = detections_to_array(ball_detections)
    arr = suppress_static_detections(arr)
    arr = reject_outliers(arr, max_jump_px=max_jump_frac * float(np.hypot(w, h)))
    return interpolate_gaps(arr, max_gap_frames)


class BallTracker:
    def __init__(self, model_path=None, conf=0.15, imgsz=1280, tile_fallback=False):
        self.conf = conf
        self.imgsz = imgsz
        self.tile_fallback = tile_fallback
        self.model = None
        if model_path:
            from ultralytics import YOLO
            self.model = YOLO(model_path)

    # ---- post-processing -------------------------------------------------
    def clean_positions(self, ball_detections, frame_size, max_jump_frac=0.12, max_gap_frames=12):
        return clean_ball_positions(ball_detections, frame_size, max_jump_frac, max_gap_frames)

    # ---- detection -------------------------------------------------------
    def detect_frames(self, frames, cache_path=None, cache_meta=None, refresh=False):
        """Detect the ball in every frame (``frames`` may be a generator)."""
        if cache_path and not refresh:
            cached = load_cache(cache_path, cache_meta)
            if cached is not None:
                return [{} if b is None else {1: b} for b in cached]

        raw = [self.detect_frame(frame) for frame in frames]

        if cache_path:
            save_cache(cache_path, cache_meta, [None if not d else d[1] for d in raw])
        return raw

    def _predict_best(self, image, offset=(0, 0)):
        results = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        best = None
        for box in results.boxes:
            conf = float(box.conf.tolist()[0])
            if best is None or conf > best[4]:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                best = [x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1], conf]
        return best

    def detect_frame(self, frame):
        """Return {1: [x1, y1, x2, y2, conf]} for the highest-confidence ball, or {}."""
        best = self._predict_best(frame)
        if best is None and self.tile_fallback:
            h, w = frame.shape[:2]
            th, tw = int(h * 0.6), int(w * 0.6)
            for oy in (0, h - th):
                for ox in (0, w - tw):
                    cand = self._predict_best(frame[oy:oy + th, ox:ox + tw], offset=(ox, oy))
                    if cand is not None and (best is None or cand[4] > best[4]):
                        best = cand
        return {} if best is None else {1: best}

    # ---- drawing ---------------------------------------------------------
    def draw_frame(self, frame, ball_dict):
        for bbox in ball_dict.values():
            x1, y1, x2, y2 = bbox[:4]
            cv2.putText(frame, "Ball", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        return frame
