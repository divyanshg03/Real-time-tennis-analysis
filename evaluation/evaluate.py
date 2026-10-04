"""Score a pipeline run against hand-labelled ground truth.

    python -m evaluation.evaluate --gt labels/clip1.json --pred-dir output_videos

See evaluation/README.md for the ground-truth format. Only the sections present in the
ground-truth file are evaluated.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from evaluation.metrics import (ball_detection_metrics, event_metrics, speed_metrics,
                                keypoint_metrics, idf1)


def evaluate_run(gt, shots_df=None, bounces_df=None, ball_arr=None, keypoints=None, players=None,
                 event_tol_frames=3):
    """``gt`` is the parsed ground-truth dict; the other arguments are pipeline outputs."""
    report = {}
    if gt.get("hits") and shots_df is not None:
        report["hits"] = event_metrics(shots_df["frame"].tolist(), gt["hits"], event_tol_frames)
    if gt.get("bounces") and bounces_df is not None:
        report["bounces"] = event_metrics(bounces_df["frame"].tolist(), gt["bounces"], event_tol_frames)
    if gt.get("shot_speeds") and shots_df is not None:
        pred = list(zip(shots_df["frame"], shots_df["ball_speed_kmh"]))
        truth = [(s["frame"], s["kmh"]) for s in gt["shot_speeds"]]
        report["shot_speeds"] = speed_metrics(pred, truth, tol_frames=max(event_tol_frames, 5))
    if "ball" in gt and ball_arr is not None:
        g = np.array([[np.nan] * 4 if b is None else b for b in gt["ball"]], float)
        report["ball_detection"] = ball_detection_metrics(ball_arr[:len(g)], g)
    if "keypoints" in gt and keypoints is not None:
        report["court_keypoints"] = keypoint_metrics(keypoints, gt["keypoints"])
    if "players" in gt and players is not None:
        gt_players = [{int(k): v for k, v in f.items()} for f in gt["players"]]
        report["player_tracking"] = idf1(players[:len(gt_players)], gt_players)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred-dir", default="output_videos")
    ap.add_argument("--tol-frames", type=int, default=3)
    args = ap.parse_args(argv)

    with open(args.gt) as f:
        gt = json.load(f)
    read = lambda name: (pd.read_csv(os.path.join(args.pred_dir, name))
                         if os.path.exists(os.path.join(args.pred_dir, name)) else None)
    report = evaluate_run(gt, shots_df=read("shots.csv"), bounces_df=read("bounces.csv"),
                          event_tol_frames=args.tol_frames)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
