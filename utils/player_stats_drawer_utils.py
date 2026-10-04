import cv2
import numpy as np


def _fmt(v):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"


def draw_player_stats_frame(frame, row, top=None):
    """Draw the stats panel on one frame. ``row`` is one row of the frame-stats DataFrame."""
    width, height = 380, 230
    start_x = frame.shape[1] - 400
    start_y = top if top is not None else max(10, min(frame.shape[0] - height - 10, frame.shape[0] - 500))
    start_x = max(0, start_x)

    overlay = frame.copy()
    cv2.rectangle(overlay, (start_x, start_y), (start_x + width, start_y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    def put(text, dx, dy, scale, thick):
        cv2.putText(frame, text, (start_x + dx, start_y + dy), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thick)

    put("Player 1 (far)  Player 2 (near)", 10, 30, 0.55, 2)
    lines = [
        ("Shots", "number_of_shots", 70, "{:.0f}"),
        ("Shot Speed", "last_shot_speed", 105, None),
        ("avg. S. Speed", "avg_shot_speed", 140, None),
        ("Player Speed", "last_player_speed", 175, None),
        ("avg. P. Speed", "avg_player_speed", 210, None),
    ]
    for label, key, dy, fmt in lines:
        v1, v2 = row[f"player_1_{key}"], row[f"player_2_{key}"]
        if fmt:
            t1, t2 = fmt.format(v1), fmt.format(v2)
            unit = ""
        else:
            t1, t2, unit = _fmt(v1), _fmt(v2), " km/h"
        put(label, 10, dy, 0.45, 1)
        put(f"{t1}{unit}", 130, dy, 0.5, 2)
        put(f"{t2}{unit}", 255, dy, 0.5, 2)
    return frame
