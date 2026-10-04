"""Central configuration. Every tunable that used to be hardcoded lives here."""
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # I/O
    input_video: str = "input_videos/input_video.mp4"
    output_video: str = "output_videos/output_video.avi"
    output_dir: str = "output_videos"
    cache_dir: str = "tracker_stubs"
    use_cache: bool = True
    refresh_cache: bool = False

    # Models
    player_model: str = "yolov8x"
    ball_model: str = "models/yolo5_last.pt"
    court_model: str = "models/keypoints_model.pth"
    court_keypoints_json: str = ""   # optional manual override: JSON list of 28 numbers

    # Ball detection
    ball_conf: float = 0.15
    ball_imgsz: int = 1280
    ball_tile_fallback: bool = False  # retry on 2x2 overlapping tiles when the full frame misses
    ball_max_jump_frac: float = 0.12  # max plausible jump per frame as a fraction of frame diagonal
    ball_max_gap_frames: int = 12     # never interpolate across longer gaps

    # Court
    court_samples: int = 7            # frames sampled to estimate keypoints (median)
    court_segment_seconds: float = 0.0  # >0 re-estimates the court every N seconds (moving camera)
    court_max_reproj_error_m: float = 0.5

    # Events / physics
    hit_window_seconds: float = 0.25
    min_hit_separation_seconds: float = 0.5
    # A hit needs the ball near a player on the court plane. The ball is airborne, so its projection is displaced
    # by parallax; the player's own head (about 1.8 m up) shows how much for this camera, so the limit is a ratio
    # of ball distance to head distance and needs no per-camera tuning. Real hits on the sample clip were 0.4-0.94,
    # bounces in front of a player 1.2-1.7; synthetic hits are about 0.5 in every camera tried. 0 disables it.
    max_hit_distance_ratio: float = 1.1
    serve_gap_seconds: float = 4.0
    max_flight_seconds: float = 3.0   # a shot's bounce must occur within this time of the hit
    assumed_contact_height_m: float = 1.0
    max_ball_speed_kmh: float = 260.0
    max_player_speed_kmh: float = 45.0
    player_smooth_window: int = 5

    # Shot type (optional: needs pose weights, downloaded on first use)
    shot_types: bool = False
    pose_model: str = "yolov8m-pose.pt"
    left_handed: str = ""            # roles that are left-handed, e.g. "2" or "1,2" (1 = far, 2 = near)

    # Output
    export_csv: bool = True
    export_heatmaps: bool = True
    draw_court_keypoints: bool = True

    def to_dict(self):
        return asdict(self)
