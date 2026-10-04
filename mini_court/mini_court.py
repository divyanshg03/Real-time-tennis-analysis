import cv2
import numpy as np

import constants
from court_geometry import COURT_KEYPOINTS_METERS, NET_Y


class MiniCourt:
    """Top-down court overlay. Positions come in as court-plane meters (from the homography)."""

    LINES = [
        (0, 2), (4, 5), (6, 7), (1, 3),   # sidelines (doubles + singles)
        (0, 1), (2, 3),                   # baselines
        (8, 9), (10, 11),                 # service lines
        (12, 13),                         # center service line
    ]

    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()

    # ---- layout ----------------------------------------------------------
    def set_canvas_background_box_position(self, frame):
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        self.pixels_per_meter = self.court_drawing_width / constants.DOUBLE_LINE_WIDTH

    def court_to_mini(self, court_xy):
        """Court meters -> mini-court pixel position."""
        x, y = court_xy
        return (self.court_start_x + x * self.pixels_per_meter,
                self.court_start_y + y * self.pixels_per_meter)

    def set_court_drawing_key_points(self):
        pts = [self.court_to_mini(p) for p in COURT_KEYPOINTS_METERS]
        self.drawing_key_points = [int(round(v)) for p in pts for v in p]

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    # ---- drawing ---------------------------------------------------------
    def _keypoint(self, idx):
        return (self.drawing_key_points[idx * 2], self.drawing_key_points[idx * 2 + 1])

    def draw_court(self, frame):
        for i in range(0, len(self.drawing_key_points), 2):
            cv2.circle(frame, (self.drawing_key_points[i], self.drawing_key_points[i + 1]), 5, (0, 0, 255), -1)
        for a, b in self.LINES:
            cv2.line(frame, self._keypoint(a), self._keypoint(b), (0, 0, 0), 2)
        net_y = int(round(self.court_to_mini((0, NET_Y))[1]))
        cv2.line(frame, (self._keypoint(0)[0], net_y), (self._keypoint(1)[0], net_y), (255, 0, 0), 2)
        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, 0.5, shapes, 0.5, 0)[mask]
        return out

    def draw_mini_court(self, frame):
        return self.draw_court(self.draw_background_rectangle(frame))

    def draw_points(self, frame, court_points, color=(0, 255, 0), radius=5):
        """Draw court-plane points (meters) given as ``{label: (x, y)}``; NaNs are skipped."""
        for _, (x, y) in court_points.items():
            if np.isnan(x) or np.isnan(y):
                continue
            px, py = self.court_to_mini((x, y))
            cv2.circle(frame, (int(px), int(py)), radius, color, -1)
        return frame
