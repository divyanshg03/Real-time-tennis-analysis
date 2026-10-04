import cv2
import numpy as np

from court_geometry import NET_Y, COURT_KEYPOINTS_METERS
from utils import get_foot_position, load_cache, save_cache

COURT_W = float(COURT_KEYPOINTS_METERS[:, 0].max())
COURT_L = float(COURT_KEYPOINTS_METERS[:, 1].max())


def distance_to_court_rect(xy, margin_x=0.0, margin_y=0.0):
    """0 if the point is inside the (expanded) court rectangle, otherwise meters outside it."""
    x, y = xy
    dx = max(-margin_x - x, 0, x - (COURT_W + margin_x))
    dy = max(-margin_y - y, 0, y - (COURT_L + margin_y))
    return float(np.hypot(dx, dy))


def interpolate_player_boxes(boxes, max_gap):
    """Fill short gaps in a per-frame list of Optional[bbox]."""
    n = len(boxes)
    out = list(boxes)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < n and out[j] is None:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            a, b = np.asarray(out[i - 1][:4]), np.asarray(out[j][:4])
            for k in range(i, j):
                t = (k - i + 1) / (j - i + 1)
                out[k] = ((1 - t) * a + t * b).tolist()
        i = j
    return out


class PlayerTracker:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            from ultralytics import YOLO
            self.model = YOLO(model_path)

    # ---- player selection ------------------------------------------------
    @staticmethod
    def select_court_players(player_detections, homography_for_frame,
                             margin_x=1.5, margin_y=4.0, max_jump_m=8.0, max_gap=30):
        """Pick the two players per frame using court geometry instead of track IDs.

        Player 1 is the far-side player, player 2 the near-side player. Because the
        choice is made per frame from where people stand on the court (people outside
        the court - umpire, ball kids, spectators - are ignored) it survives tracker
        ID switches. Returns a list of ``{1: bbox, 2: bbox}`` dicts; a role missing
        for a frame longer than ``max_gap`` is simply absent.
        """
        n = len(player_detections)
        per_role = {1: [None] * n, 2: [None] * n}
        last_court = {1: None, 2: None}

        for i, dets in enumerate(player_detections):
            H = homography_for_frame(i)
            cands = []
            for bbox in dets.values():
                court_xy = H.pixel_to_court_point(get_foot_position(bbox))
                if distance_to_court_rect(court_xy, margin_x, margin_y) > 0:
                    continue  # outside the allowed area around the court
                role = 1 if court_xy[1] < NET_Y else 2
                cands.append((role, court_xy, bbox))

            for role in (1, 2):
                in_role = [c for c in cands if c[0] == role]
                if not in_role:
                    continue
                best = None
                prev = last_court[role]
                if prev is not None:
                    d, cand = min(((np.hypot(c[1][0] - prev[0], c[1][1] - prev[1]), c) for c in in_role),
                                  key=lambda t: t[0])
                    if d <= max_jump_m:
                        best = cand
                if best is None:
                    # (Re)initialise: most "on court" = closest to the court rectangle, then largest box
                    best = min(in_role, key=lambda c: (distance_to_court_rect(c[1]),
                                                       -(c[2][3] - c[2][1])))
                per_role[role][i] = list(best[2][:4])
                last_court[role] = best[1]

        for role in (1, 2):
            per_role[role] = interpolate_player_boxes(per_role[role], max_gap)

        return [{role: per_role[role][i] for role in (1, 2) if per_role[role][i] is not None}
                for i in range(n)]

    # ---- detection -------------------------------------------------------
    def detect_frames(self, frames, cache_path=None, cache_meta=None, refresh=False):
        if cache_path and not refresh:
            cached = load_cache(cache_path, cache_meta)
            if cached is not None:
                return [{int(k): v for k, v in frame.items()} for frame in cached]

        detections = [self.detect_frame(frame) for frame in frames]

        if cache_path:
            save_cache(cache_path, cache_meta, [{str(k): v for k, v in d.items()} for d in detections])
        return detections

    def detect_frame(self, frame):
        """Return {track_id: bbox} for every person (class 0) in the frame.

        When the tracker has not assigned an ID yet (``box.id is None``) a negative
        per-frame ID is used, so the detection is kept and the run does not crash.
        """
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)[0]
        player_dict = {}
        for k, box in enumerate(results.boxes):
            track_id = int(box.id.tolist()[0]) if box.id is not None else -(k + 1)
            player_dict[track_id] = box.xyxy.tolist()[0]
        return player_dict

    # ---- drawing ---------------------------------------------------------
    def draw_frame(self, frame, player_dict):
        for player_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox[:4]
            cv2.putText(frame, f"Player {player_id}", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        return frame
