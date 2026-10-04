"""End-to-end pipeline.

Pass 1 streams the video through the detectors (cached, constant memory).
Analysis then runs on the detections only.
Pass 2 streams the video again, draws the overlays and writes the output.

Trackers and the court model can be injected, which is how the tests run the full
pipeline without model weights.
"""
import logging
import os

import cv2
import numpy as np

from analytics.events import ball_centers
from analytics import (detect_events, compute_shots, build_frame_stats, summarize,
                       export_tables, export_heatmaps)
from config import Config
from court_line_detector.court_line_detector import (CourtLineDetector, CourtModel,
                                                     estimate_court_model, manual_court_model)
from mini_court import MiniCourt
from trackers import PlayerTracker, BallTracker, clean_ball_positions
from utils import (get_video_info, iter_video, read_frames_at, VideoSink, make_meta,
                   draw_player_stats_frame, get_foot_position)

log = logging.getLogger(__name__)

BOUNCE_MARKER_FRAMES = 15
SERVE_DISTANCE_FACTOR = 2.0   # serves are struck higher (about 1.5-1.7 head distances, versus ~0.5 for a rally hit)
MIN_HEAD_DISPLACEMENT_M = 1.0  # guards against a degenerate head/foot projection


def build_court_model(cfg: Config, info):
    if cfg.court_keypoints_json:
        log.info("using manual court keypoints from %s", cfg.court_keypoints_json)
        return manual_court_model(cfg.court_keypoints_json, info.frame_count)
    detector = CourtLineDetector(cfg.court_model)
    return estimate_court_model(
        detector.predict,
        lambda idxs: read_frames_at(cfg.input_video, idxs),
        info.frame_count, info.fps,
        n_samples=cfg.court_samples,
        segment_seconds=cfg.court_segment_seconds,
        max_error_m=cfg.court_max_reproj_error_m,
    )


def _court_distance_gate(ball_arr, players, court_model, max_ratio):
    """frame -> True when the ball is near a player on the court plane.

    "Near" is relative to that player's own head: both the ball and the head are airborne points, so both are
    displaced by the same camera parallax. The ball must be within ``max_ratio`` times the head's displacement
    from the player's feet (``SERVE_DISTANCE_FACTOR`` times more in the serve window).
    """
    if not max_ratio:
        return None
    centers = ball_centers(ball_arr)

    def gate(frame, serve_like=False):
        limit = max_ratio * (SERVE_DISTANCE_FACTOR if serve_like else 1.0)
        c = centers[frame]
        if np.isnan(c[0]):
            return False
        H = court_model.homography_for_frame(frame)
        bx, by = H.pixel_to_court_point((c[0], c[1]))
        for bbox in players[frame].values():
            fx, fy = H.pixel_to_court_point(get_foot_position(bbox))
            hx, hy = H.pixel_to_court_point(((bbox[0] + bbox[2]) / 2, bbox[1]))
            head = max(np.hypot(hx - fx, hy - fy), MIN_HEAD_DISPLACEMENT_M)
            if np.hypot(bx - fx, by - fy) <= limit * head:
                return True
        return False

    return gate


def analyse(cfg: Config, info, player_detections, ball_detections, court_model: CourtModel):
    """Pure analysis: detections in, tables out (no video or models involved)."""
    players = PlayerTracker.select_court_players(
        player_detections, court_model.homography_for_frame)
    ball_arr = clean_ball_positions(ball_detections, (info.width, info.height),
                                    cfg.ball_max_jump_frac, cfg.ball_max_gap_frames)
    gate = _court_distance_gate(ball_arr, players, court_model, cfg.max_hit_distance_ratio)
    events = detect_events(ball_arr, players, info.fps, info.height,
                           cfg.hit_window_seconds, cfg.min_hit_separation_seconds, hit_gate=gate)
    result = compute_shots(events, ball_arr, players, court_model.homography_for_frame, info.fps, cfg)
    frame_stats = build_frame_stats(result["shots"], len(ball_arr))
    summary = summarize(result["shots"], result["speeds"], result["tracks"], info.fps,
                        quality=court_model.quality)
    return {"players": players, "ball": ball_arr, "events": events, "frame_stats": frame_stats,
            "summary": summary, **result}


def render(cfg: Config, info, analysis, court_model: CourtModel, player_tracker, ball_tracker):
    os.makedirs(os.path.dirname(cfg.output_video) or ".", exist_ok=True)
    players, ball_arr = analysis["players"], analysis["ball"]
    tracks, frame_stats = analysis["tracks"], analysis["frame_stats"]
    bounce_rows = analysis["bounces"].to_dict("records")
    mini_court = None
    stats_rows = frame_stats.to_dict("records")

    with VideoSink(cfg.output_video, info.fps, (info.width, info.height)) as sink:
        for i, frame in enumerate(iter_video(cfg.input_video)):
            if i >= len(players):
                break
            if mini_court is None:
                mini_court = MiniCourt(frame)

            frame = player_tracker.draw_frame(frame, players[i])
            row = ball_arr[i]
            if not np.isnan(row[0]):
                frame = ball_tracker.draw_frame(frame, {1: row.tolist()})
            if cfg.draw_court_keypoints:
                kps = court_model.keypoints_for_frame(i)
                for k in range(0, len(kps), 2):
                    cv2.circle(frame, (int(kps[k]), int(kps[k + 1])), 5, (0, 0, 255), -1)

            frame = mini_court.draw_mini_court(frame)
            frame = mini_court.draw_points(
                frame, {r: tuple(tracks[r][i]) for r in tracks}, color=(0, 255, 0))
            recent = {b["frame"]: (b["court_x_m"], b["court_y_m"]) for b in bounce_rows
                      if 0 <= i - b["frame"] < BOUNCE_MARKER_FRAMES}
            frame = mini_court.draw_points(frame, recent, color=(0, 255, 255), radius=4)

            frame = draw_player_stats_frame(frame, stats_rows[i], top=mini_court.end_y + 20)
            cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            sink.write(frame)


def run(cfg: Config, player_tracker=None, ball_tracker=None, court_model=None, pose=None):
    info = get_video_info(cfg.input_video)
    log.info("video: %dx%d @ %.3f fps, %d frames", info.width, info.height, info.fps, info.frame_count)

    if court_model is None:
        court_model = build_court_model(cfg, info)
    log.info("court quality: %s", court_model.quality)

    player_tracker = player_tracker or PlayerTracker(cfg.player_model)
    ball_tracker = ball_tracker or BallTracker(cfg.ball_model, cfg.ball_conf, cfg.ball_imgsz,
                                               cfg.ball_tile_fallback)

    stem = os.path.splitext(os.path.basename(cfg.input_video))[0]
    use_cache = cfg.use_cache
    p_path = os.path.join(cfg.cache_dir, f"{stem}_players.json") if use_cache else None
    b_path = os.path.join(cfg.cache_dir, f"{stem}_ball.json") if use_cache else None
    p_meta = make_meta(cfg.input_video, cfg.player_model, "players")
    b_meta = make_meta(cfg.input_video, f"{cfg.ball_model}|conf={cfg.ball_conf}|imgsz={cfg.ball_imgsz}"
                                        f"|tile={cfg.ball_tile_fallback}", "ball")

    player_detections = player_tracker.detect_frames(iter_video(cfg.input_video), p_path, p_meta,
                                                     cfg.refresh_cache)
    ball_detections = ball_tracker.detect_frames(iter_video(cfg.input_video), b_path, b_meta,
                                                 cfg.refresh_cache)

    analysis = analyse(cfg, info, player_detections, ball_detections, court_model)
    if cfg.shot_types:
        from analytics.pose import PoseEstimator, add_shot_types
        lefties = {int(r) for r in cfg.left_handed.split(",") if r.strip()}
        analysis["shots"] = add_shot_types(
            analysis["shots"], analysis["players"], cfg.input_video, len(analysis["ball"]),
            pose or PoseEstimator(cfg.pose_model), right_handed=(1 not in lefties, 2 not in lefties))
    log.info("detected %d hits, %d bounces", len(analysis["events"].hits), len(analysis["events"].bounces))

    if cfg.export_csv:
        paths = export_tables(cfg.output_dir, analysis["shots"], analysis["bounces"],
                              analysis["frame_stats"], analysis["summary"], cfg.to_dict())
        log.info("wrote %s", ", ".join(paths.values()))
    if cfg.export_heatmaps:
        export_heatmaps(cfg.output_dir, analysis["tracks"])

    render(cfg, info, analysis, court_model, player_tracker, ball_tracker)
    log.info("wrote %s", cfg.output_video)
    return analysis
