"""Pose estimation around each hit, and attaching a shot type to the shots table.

The pose model only runs on a few frames per shot, so this is cheap compared with detection. It is
optional: the default pipeline does not need the pose weights.
"""
import numpy as np

from analytics.shot_type import classify_shot
from utils import read_frames_at

CONTACT_OFFSETS = (-2, -1, 0, 1)   # frames around the detected hit; the hit frame is only good to about +-1


class PoseEstimator:
    """Wraps an Ultralytics pose model; ``keypoints_for(frame, bbox)`` returns the (17, 3) pose of the person
    closest to ``bbox`` or None."""

    def __init__(self, model_path="yolov8m-pose.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def keypoints_for(self, frame, bbox):
        res = self.model.predict(frame, verbose=False)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return None
        kps = res.keypoints.data.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        target = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])
        d = np.linalg.norm(centers - target, axis=1)
        best = int(np.argmin(d))
        height = max(bbox[3] - bbox[1], 1.0)
        return kps[best] if d[best] <= height else None   # nobody close enough to be the player


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
