"""Pose estimation around each hit, and attaching a shot type to the shots table.

The pose model only runs on a few frames per shot, so this is cheap compared with detection. It is
optional: the default pipeline does not need the pose weights.
"""
import numpy as np

from analytics.shot_type import classify_shot
from utils import read_frames_at

CONTACT_OFFSETS = (-2, -1, 0, 1)   # frames around the detected hit; the hit frame is only good to about +-1


def map_keypoints_to_frame(kps, x0, y0, scale):
    """Keypoints found on a crop (origin ``x0, y0``, resized by ``scale``) back to full-frame pixels."""
    out = np.array(kps, dtype=np.float64, copy=True)
    out[:, 0] = out[:, 0] / scale + x0
    out[:, 1] = out[:, 1] / scale + y0
    return out


def crop_window(bbox, frame_shape, margin=0.25, min_height=512):
    """Crop box around a player and the factor to enlarge it by so a small far-side player is big enough
    for the pose model. Returns ``(x0, y0, x1, y1, scale)`` in full-frame pixels."""
    h_img, w_img = frame_shape[:2]
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x0 = int(max(0, bbox[0] - margin * bw)); x1 = int(min(w_img, bbox[2] + margin * bw))
    y0 = int(max(0, bbox[1] - margin * bh)); y1 = int(min(h_img, bbox[3] + margin * bh))
    scale = max(1.0, min_height / max(y1 - y0, 1))
    return x0, y0, x1, y1, scale


class PoseEstimator:
    """Wraps an Ultralytics pose model. ``keypoints_for(frame, bbox)`` crops around the player, enlarges the
    crop (far-side players are only ~100 px tall) and returns the (17, 3) pose in full-frame pixels, or None."""

    def __init__(self, model_path="yolov8m-pose.pt", min_crop_height=512):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.min_crop_height = min_crop_height

    def keypoints_for(self, frame, bbox):
        import cv2
        x0, y0, x1, y1, scale = crop_window(bbox, frame.shape, min_height=self.min_crop_height)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        res = self.model.predict(crop, verbose=False)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return None
        kps = res.keypoints.data.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        # the player is the person nearest the middle of the crop
        mid = np.array([crop.shape[1] / 2, crop.shape[0] / 2])
        centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])
        best = int(np.argmin(np.linalg.norm(centers - mid, axis=1)))
        return map_keypoints_to_frame(kps[best], x0, y0, scale)


def add_shot_types(shots_df, players, video_path, n_frames, pose, right_handed=(True, True)):
    """Add a ``shot_type`` column. ``players[frame][role]`` is the selected bbox; ``right_handed`` is
    indexed by role - 1."""
    types = []
    for _, shot in shots_df.iterrows():
        hit, role = int(shot["frame"]), int(shot["hitter"])
        frames_idx = [f for f in (hit + o for o in CONTACT_OFFSETS) if 0 <= f < n_frames and players[f].get(role)]
        images = read_frames_at(video_path, frames_idx)
        seq = [pose.keypoints_for(img, players[f][role]) for f, img in zip(frames_idx, images)]
        label, _ = classify_shot(seq, is_serve=bool(shot["is_serve"]), right_handed=right_handed[role - 1])
        types.append(label)
    out = shots_df.copy()
    out["shot_type"] = types
    return out
