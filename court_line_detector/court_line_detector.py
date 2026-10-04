import json
import logging

import cv2
import numpy as np

from court_geometry import CourtHomography

log = logging.getLogger(__name__)


class CourtLineDetector:
    """ResNet50 court keypoint regressor (14 points -> 28 numbers)."""

    def __init__(self, model_path):
        import torch
        import torchvision.transforms as transforms
        from torchvision import models

        self._torch = torch
        # weights=None: the checkpoint overwrites everything, no need to download ImageNet weights
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        # BatchNorm must use running statistics at inference; without eval() a batch of 1
        # normalises with its own statistics and the keypoints degrade.
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with self._torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0
        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.putText(image, str(i // 2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image


def aggregate_keypoints(samples, outlier_px=5.0):
    """Median of several keypoint predictions, discarding samples that disagree with the consensus.

    ``samples`` is a list of flat 28-number arrays. Returns (keypoints, n_used).
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("samples must be a list of flat keypoint arrays")
    median = np.median(arr, axis=0)
    if len(arr) >= 3:
        dev = np.linalg.norm((arr - median).reshape(len(arr), -1, 2), axis=2).mean(axis=1)
        keep = dev <= max(3 * np.median(dev), outlier_px)
        if keep.sum() >= 1:
            arr = arr[keep]
            median = np.median(arr, axis=0)
    return median, len(arr)


class CourtModel:
    """Court keypoints/homography per time segment (one segment for a static camera)."""

    def __init__(self, segments):
        # segments: list of dicts(start, end, keypoints, homography); end is exclusive
        self.segments = segments

    def _segment(self, frame_idx):
        for seg in self.segments:
            if seg["start"] <= frame_idx < seg["end"]:
                return seg
        return self.segments[-1] if frame_idx >= self.segments[-1]["end"] else self.segments[0]

    def homography_for_frame(self, frame_idx):
        return self._segment(frame_idx)["homography"]

    def keypoints_for_frame(self, frame_idx):
        return self._segment(frame_idx)["keypoints"]

    @property
    def quality(self):
        errs = [float(s["homography"].reprojection_error_m.mean()) for s in self.segments]
        return {"segments": len(self.segments), "mean_reproj_error_m": float(np.mean(errs)),
                "worst_reproj_error_m": float(np.max(errs))}


def manual_court_model(keypoints_json_path, frame_count):
    """Build a static CourtModel from user-supplied keypoints (28 numbers in a JSON list)."""
    with open(keypoints_json_path) as f:
        kps = np.asarray(json.load(f), dtype=np.float64).ravel()
    return CourtModel([{"start": 0, "end": max(frame_count, 1), "keypoints": kps,
                        "homography": CourtHomography(kps)}])


def estimate_court_model(predict_fn, read_frames_at, frame_count, fps,
                         n_samples=7, segment_seconds=0.0, max_error_m=0.5):
    """Estimate court geometry robustly.

    ``predict_fn(image) -> 28 keypoints`` and ``read_frames_at(indices) -> list of images``
    are injected so this can be tested without a model or a video. Keypoints are
    predicted on several frames and aggregated with a median; with
    ``segment_seconds > 0`` that is repeated per segment so a camera that moves/zooms
    between shots gets a fresh homography.
    """
    if segment_seconds and segment_seconds > 0:
        seg_len = max(int(round(segment_seconds * fps)), 1)
        bounds = [(s, min(s + seg_len, frame_count)) for s in range(0, frame_count, seg_len)]
    else:
        bounds = [(0, frame_count)]

    per_seg_samples = max(3, n_samples if len(bounds) == 1 else min(n_samples, 3))
    segments = []
    for start, end in bounds:
        idxs = np.unique(np.linspace(start, end - 1, per_seg_samples).round().astype(int)).tolist()
        samples = [predict_fn(img) for img in read_frames_at(idxs)]
        kps, used = aggregate_keypoints(samples)
        H = CourtHomography(kps)
        if not H.is_plausible(max_mean_error_m=max_error_m):
            log.warning("court homography for frames %d-%d looks unreliable "
                        "(mean reprojection error %.2f m, inliers %.0f%%). Check the keypoint model "
                        "or supply --court-keypoints-json.", start, end,
                        H.reprojection_error_m.mean(), 100 * H.inlier_mask.mean())
        segments.append({"start": start, "end": end, "keypoints": kps, "homography": H})
        log.info("court segment %d-%d: %d/%d samples used", start, end, used, len(samples))
    return CourtModel(segments)
