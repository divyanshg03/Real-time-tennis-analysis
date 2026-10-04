"""Near-real-time analysis of a camera, stream or video file.

    python live.py --source 0 --display                       # webcam
    python live.py --source rtsp://host/stream --output out.avi
    python live.py --source input_videos/input_video.mp4 --display

Frames are detected one at a time; every ``update_every`` frames the full analysis is re-run over the last
``window_seconds`` of detections and the overlay shows its latest result. Two things to know:

* **Latency.** A hit can only be confirmed once about half a second of later frames exists, and the analysis
  runs in steps, so the statistics trail the picture by roughly a second.
* **Static camera.** The court is located once from the first ``warmup_seconds`` of frames and not updated.

The detectors and the court predictor can be injected, which is how the tests run it without model weights.
"""
import argparse
import logging
import time
from collections import deque

import cv2
import numpy as np

import pipeline
from config import Config
from court_line_detector.court_line_detector import (CourtLineDetector, estimate_court_model,
                                                     manual_court_model)
from mini_court import MiniCourt
from trackers import BallTracker, PlayerTracker
from utils import VideoInfo, VideoSink, draw_player_stats_frame

log = logging.getLogger(__name__)


class LiveAnalyzer:
    """Feed it frames with ``push``; read the latest result from ``self.analysis``."""

    def __init__(self, cfg, info, player_tracker, ball_tracker, court_model,
                 window_seconds=12.0, update_every=15, player_every=1):
        self.player_every = max(1, int(player_every))
        self.cfg, self.info = cfg, info
        self.player_tracker, self.ball_tracker, self.court_model = player_tracker, ball_tracker, court_model
        self.window = max(int(window_seconds * info.fps), 30)
        self.update_every = update_every
        self.players = deque(maxlen=self.window)
        self.balls = deque(maxlen=self.window)
        self.frames_seen = 0
        self.analysis = None
        self.analysis_len = 0        # number of frames the latest analysis covered

    def push(self, frame):
        # The player model is the expensive one. Skipped frames get an empty detection and the analysis
        # interpolates short gaps in the chosen player boxes, so the ball (which needs every frame) is unaffected.
        run_players = self.frames_seen % self.player_every == 0
        self.players.append(self.player_tracker.detect_frame(frame) if run_players else {})
        self.balls.append(self.ball_tracker.detect_frame(frame))
        self.frames_seen += 1
        enough = len(self.players) >= int(2 * self.info.fps)
        if enough and self.frames_seen % self.update_every == 0:
            self._update()

    def _update(self):
        n = len(self.players)
        info = VideoInfo(self.info.fps, self.info.width, self.info.height, n)
        # Court geometry is static, so any frame index maps to the same homography.
        self.analysis = pipeline.analyse(self.cfg, info, list(self.players), list(self.balls), self.court_model)
        self.analysis_len = n

    # ---- drawing -----------------------------------------------------------------------------------------
    def render(self, frame, mini_court):
        out = frame.copy()
        chosen = PlayerTracker.select_court_players([self.players[-1]], self.court_model.homography_for_frame)[0]
        for role, bbox in chosen.items():
            self.player_tracker.draw_frame(out, {role: bbox})
        ball = self.balls[-1]
        if ball:
            self.ball_tracker.draw_frame(out, ball)
        out = mini_court.draw_mini_court(out)
        a = self.analysis
        if a is not None:
            last = a["frame_stats"].iloc[-1]
            tracks = {r: tuple(a["tracks"][r][-1]) for r in a["tracks"]}
            out = mini_court.draw_points(out, tracks)
            out = draw_player_stats_frame(out, last, top=mini_court.end_y + 20)
        cv2.putText(out, f"LIVE  frame {self.frames_seen}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return out


def open_source(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    info = VideoInfo(float(fps) if fps and fps > 1 else 30.0,
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 0)
    return cap, info


def run_live(cfg, source, display=False, output=None, max_frames=None, warmup_seconds=2.0,
             player_tracker=None, ball_tracker=None, court_predict=None, window_seconds=12.0, update_every=15, player_every=1):
    cap, info = open_source(source)
    log.info("source %s: %dx%d @ %.1f fps", source, info.width, info.height, info.fps)
    player_tracker = player_tracker or PlayerTracker(cfg.player_model)
    ball_tracker = ball_tracker or BallTracker(cfg.ball_model, cfg.ball_conf, cfg.ball_imgsz, cfg.ball_tile_fallback)

    # warm-up: buffer a couple of seconds to locate the court, then replay them through the analyzer
    warm = []
    while len(warm) < int(warmup_seconds * info.fps):
        ok, frame = cap.read()
        if not ok:
            break
        warm.append(frame)
    if not warm:
        raise RuntimeError("no frames received from the source")
    if cfg.court_keypoints_json:
        court_model = manual_court_model(cfg.court_keypoints_json, 10 ** 9)
    else:
        predict = court_predict or CourtLineDetector(cfg.court_model).predict
        court_model = estimate_court_model(predict, lambda idxs: [warm[i] for i in idxs], len(warm), info.fps,
                                           n_samples=min(cfg.court_samples, len(warm)),
                                           max_error_m=cfg.court_max_reproj_error_m)
        court_model.segments[-1]["end"] = 10 ** 9      # static camera: one homography for every frame
    log.info("court located: %s", court_model.quality)

    live = LiveAnalyzer(cfg, info, player_tracker, ball_tracker, court_model, window_seconds, update_every,
                        player_every)
    mini = MiniCourt(warm[0])
    sink = VideoSink(output, info.fps, (info.width, info.height)) if output else None

    def handle(frame):
        live.push(frame)
        if sink or display:
            shown = live.render(frame, mini)
            if sink:
                sink.write(shown)
            if display:
                cv2.imshow("tennis live", shown)
                return cv2.waitKey(1) & 0xFF != ord("q")
        return True

    started, processed, keep_going = time.time(), 0, True
    try:
        for frame in warm:
            keep_going = handle(frame)
            processed += 1
            if not keep_going:
                break
        while keep_going and (max_frames is None or processed < max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            keep_going = handle(frame)
            processed += 1
    finally:
        cap.release()
        if sink:
            sink.close()
        if display:
            cv2.destroyAllWindows()
    if live.analysis is None and len(live.players) >= 2:
        live._update()          # short clip: make sure there is a final result
    elapsed = max(time.time() - started, 1e-6)
    log.info("processed %d frames in %.1f s (%.1f fps)", processed, elapsed, processed / elapsed)
    return live


def main(argv=None):
    d = Config()
    p = argparse.ArgumentParser(description="Near-real-time tennis analysis")
    p.add_argument("--source", default="0", help="camera index, stream URL or video file")
    p.add_argument("--display", action="store_true", help="show a window (press q to quit)")
    p.add_argument("--output", default="", help="also write the annotated stream to this .avi")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--window-seconds", type=float, default=12.0)
    p.add_argument("--update-every", type=int, default=15, help="re-run the analysis every N frames")
    p.add_argument("--player-every", type=int, default=1,
                   help="detect players on every Nth frame (2-3 roughly halves the cost; gaps are interpolated)")
    p.add_argument("--player-model", default=d.player_model)
    p.add_argument("--ball-model", default=d.ball_model)
    p.add_argument("--court-model", default=d.court_model)
    p.add_argument("--court-keypoints-json", default=d.court_keypoints_json)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(player_model=a.player_model, ball_model=a.ball_model, court_model=a.court_model,
                 court_keypoints_json=a.court_keypoints_json)
    live = run_live(cfg, a.source, a.display, a.output or None, a.max_frames,
                    window_seconds=a.window_seconds, update_every=a.update_every,
                    player_every=a.player_every)
    if live.analysis is not None:
        print(live.analysis["summary"])


if __name__ == "__main__":
    main()
