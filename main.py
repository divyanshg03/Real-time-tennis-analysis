import argparse
import logging
import os
import sys

from config import Config


def parse_args(argv=None):
    d = Config()
    p = argparse.ArgumentParser(description="Tennis match video analysis")
    p.add_argument("--input", default=d.input_video, help="input video")
    p.add_argument("--output", default=d.output_video, help="annotated output video (.avi)")
    p.add_argument("--output-dir", default=d.output_dir, help="where CSV/JSON/heatmaps are written")
    p.add_argument("--player-model", default=d.player_model)
    p.add_argument("--ball-model", default=d.ball_model)
    p.add_argument("--court-model", default=d.court_model)
    p.add_argument("--court-keypoints-json", default=d.court_keypoints_json,
                   help="manual court keypoints (JSON list of 28 numbers); skips the keypoint model")
    p.add_argument("--court-segment-seconds", type=float, default=d.court_segment_seconds,
                   help="re-estimate the court every N seconds (for moving cameras); 0 = static camera")
    p.add_argument("--ball-conf", type=float, default=d.ball_conf)
    p.add_argument("--ball-imgsz", type=int, default=d.ball_imgsz)
    p.add_argument("--ball-tile-fallback", action="store_true",
                   help="retry missed ball detections on overlapping tiles (slower, finds small balls)")
    p.add_argument("--cache-dir", default=d.cache_dir)
    p.add_argument("--no-cache", action="store_true", help="never read or write the detection cache")
    p.add_argument("--refresh-cache", action="store_true", help="recompute detections and overwrite the cache")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--no-heatmaps", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def config_from_args(a) -> Config:
    return Config(
        input_video=a.input, output_video=a.output, output_dir=a.output_dir,
        cache_dir=a.cache_dir, use_cache=not a.no_cache, refresh_cache=a.refresh_cache,
        player_model=a.player_model, ball_model=a.ball_model, court_model=a.court_model,
        court_keypoints_json=a.court_keypoints_json, court_segment_seconds=a.court_segment_seconds,
        ball_conf=a.ball_conf, ball_imgsz=a.ball_imgsz, ball_tile_fallback=a.ball_tile_fallback,
        export_csv=not a.no_csv, export_heatmaps=not a.no_heatmaps,
    )


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = config_from_args(args)

    required = [cfg.input_video]
    if not cfg.court_keypoints_json:
        required.append(cfg.court_model)
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        sys.exit("missing file(s): " + ", ".join(missing) +
                 "\nSee models/put_models_here.txt and the README for the model download links.")

    from pipeline import run
    run(cfg)


if __name__ == "__main__":
    main()
