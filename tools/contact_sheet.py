"""Render a grid of numbered frames to one image, for scanning a clip quickly.

    python tools/contact_sheet.py --video input_videos/input_video.mp4 --start 0 --end 60 --step 2

Use it to find roughly where hits and bounces are, then refine with tools/label_events.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from utils import get_video_info, read_frames_at  # noqa: E402


def make_sheet(frames, indices, cols=4, tile_w=480):
    tiles = []
    for img, idx in zip(frames, indices):
        h, w = img.shape[:2]
        tile = cv2.resize(img, (tile_w, int(h * tile_w / w)))
        cv2.putText(tile, str(idx), (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input_videos/input_video.mp4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", default="labels/contact_sheet.png")
    args = ap.parse_args(argv)

    info = get_video_info(args.video)
    end = info.frame_count if args.end is None else min(args.end, info.frame_count)
    indices = list(range(args.start, end, args.step))
    frames = read_frames_at(args.video, indices)
    sheet = make_sheet(frames, indices[:len(frames)], args.cols)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, sheet)
    print(f"wrote {args.out} ({len(frames)} frames, {sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
