"""Real-world court geometry and pixel -> court-plane homography.

Court coordinates are in meters. Origin is the far-left doubles corner
(keypoint 0), +x runs to the right, +y runs toward the camera. The net sits
at y = COURT_LENGTH / 2.

Keypoint order matches the court keypoint model (14 points) and the layout
that ``MiniCourt`` draws.
"""
import cv2
import numpy as np

import constants

_W = constants.DOUBLE_LINE_WIDTH
_L = constants.COURT_LENGTH
_ALLEY = constants.DOUBLE_ALLY_DIFFERENCE
_SINGLES_W = constants.SINGLE_LINE_WIDTH
_SERVICE = constants.NO_MANS_LAND_HEIGHT

COURT_KEYPOINTS_METERS = np.array([
    (0.0, 0.0),                         # 0  far-left doubles corner
    (_W, 0.0),                          # 1  far-right doubles corner
    (0.0, _L),                          # 2  near-left doubles corner
    (_W, _L),                           # 3  near-right doubles corner
    (_ALLEY, 0.0),                      # 4  far baseline, left singles line
    (_ALLEY, _L),                       # 5  near baseline, left singles line
    (_W - _ALLEY, 0.0),                 # 6  far baseline, right singles line
    (_W - _ALLEY, _L),                  # 7  near baseline, right singles line
    (_ALLEY, _SERVICE),                 # 8  far service line, left
    (_ALLEY + _SINGLES_W, _SERVICE),    # 9  far service line, right
    (_ALLEY, _L - _SERVICE),            # 10 near service line, left
    (_ALLEY + _SINGLES_W, _L - _SERVICE),  # 11 near service line, right
    (_W / 2, _SERVICE),                 # 12 far T
    (_W / 2, _L - _SERVICE),            # 13 near T
], dtype=np.float64)

NET_Y = _L / 2


def keypoints_to_array(keypoints):
    """Flat [x0, y0, x1, y1, ...] -> (14, 2) float array."""
    return np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)


class CourtHomography:
    """Maps image pixels onto the court plane (meters) and back.

    Only valid for points that lie on the ground plane (feet, bounces).
    """

    def __init__(self, image_keypoints, ransac_threshold_px=8.0):
        pts = keypoints_to_array(image_keypoints)
        if pts.shape != COURT_KEYPOINTS_METERS.shape:
            raise ValueError(f"expected 14 keypoints, got shape {pts.shape}")
        self.image_keypoints = pts
        H, mask = cv2.findHomography(pts, COURT_KEYPOINTS_METERS, cv2.RANSAC, ransac_threshold_px)
        if H is None:
            raise ValueError("homography could not be estimated from the court keypoints")
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.inlier_mask = mask.ravel().astype(bool)

    @property
    def reprojection_error_m(self):
        """Per-keypoint error (meters) after mapping to the court plane."""
        proj = self.pixel_to_court(self.image_keypoints)
        return np.linalg.norm(proj - COURT_KEYPOINTS_METERS, axis=1)

    @property
    def reprojection_error_px(self):
        """Per-keypoint error (pixels) when projecting the ideal court back to the image."""
        back = self.court_to_pixel(COURT_KEYPOINTS_METERS)
        return np.linalg.norm(back - self.image_keypoints, axis=1)

    def is_plausible(self, max_mean_error_m=0.5, min_inlier_fraction=0.7):
        return (
            float(self.reprojection_error_m.mean()) <= max_mean_error_m
            and float(self.inlier_mask.mean()) >= min_inlier_fraction
        )

    def _apply(self, M, points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, M).reshape(-1, 2)

    def pixel_to_court(self, points):
        return self._apply(self.H, points)

    def court_to_pixel(self, points):
        return self._apply(self.H_inv, points)

    def pixel_to_court_point(self, point):
        return tuple(self.pixel_to_court([point])[0])


def is_inside_court(court_xy, margin=0.0, singles=False):
    """True if a court-plane point lies inside the court (plus ``margin`` meters)."""
    x, y = court_xy
    x_min = (_ALLEY if singles else 0.0) - margin
    x_max = (_W - _ALLEY if singles else _W) + margin
    return x_min <= x <= x_max and -margin <= y <= _L + margin
