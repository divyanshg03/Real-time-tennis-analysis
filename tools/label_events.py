"""Interactive labeller for hits and bounces (ground truth for evaluation/).

    python tools/label_events.py --video input_videos/input_video.mp4 --out labels/input_video.json

Keys
    d / Right    next frame          a / Left   previous frame
    w / Up       +10 frames          s / Down   -10 frames
    h            toggle HIT at this frame (racket contact)
    b            toggle BOUNCE at this frame (ball touches the ground)
    n / p        jump to next / previous labelled event
    q / Esc      save and quit        (progress is also saved after every label)

How to label well
  * A hit is the first frame where the ball is visibly touching the racket strings.
  * A bounce is the frame where the ball is lowest, touching the court.
  * Step with a/d around the moment; at 30 fps the exact frame matters (+-1 frame ~ 5 % speed).
  * Existing labels are loaded and kept, so you can stop and resume.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from utils import get_video_info  # noqa: E402


def load_labels(path, video, info):
    if os.path.exists(path):
        with open(path) as f:
            labels = json.load(f)
    else:
        labels = {}
    labels.setdefault("video", video)
    labels.setdefault("fps", info.fps)
    labels.setdefault("frame_count", info.frame_count)
    labels.setdefault("hits", [])
    labels.setdefault("bounces", [])
    labels.setdefault("shot_speeds", [])
    return labels


def toggle(labels, key, frame):
    """Add ``frame`` to labels[key] or remove it if present. Returns True if now labelled."""
    frames = labels[key]
    if frame in frames:
        frames.remove(frame)
        return False
    frames.append(frame)
    frames.sort()
    return True


def save_labels(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(labels, f, indent=2)
    os.replace(tmp, path)


def next_event(labels, frame, direction):
    events = sorted(set(labels["hits"]) | set(labels["bounces"]))
    if direction > 0:
        later = [e for e in events if e > frame]
        return later[0] if later else frame
    earlier = [e for e in events if e < frame]
    return earlier[-1] if earlier else frame


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input_videos/input_video.mp4")
    ap.add_argument("--out", default="labels/input_video.json")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args(argv)

    info = get_video_info(args.video)
    labels = load_labels(args.out, args.video, info)
    cap = cv2.VideoCapture(args.video)
    last = info.frame_count - 1
    frame_idx = max(0, min(args.start, last))
    win = "label events  (h=hit  b=bounce  a/d=step  w/s=+-10  n/p=next/prev label  q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        is_hit, is_bounce = frame_idx in labels["hits"], frame_idx in labels["bounces"]
        banner = f"frame {frame_idx}/{last}  t={frame_idx / info.fps:.2f}s   hits={len(labels['hits'])} bounces={len(labels['bounces'])}"
        cv2.putText(frame, banner, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        if is_hit:
            cv2.putText(frame, "HIT", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        if is_bounce:
            cv2.putText(frame, "BOUNCE", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 255), 4)
        cv2.imshow(win, frame)

        k = cv2.waitKeyEx(0)
        ch = chr(k & 0xFF) if 0 <= (k & 0xFF) < 128 else ""
        if ch in ("q",) or k == 27:
            break
        elif ch == "d" or k == 2555904:
            frame_idx = min(last, frame_idx + 1)
        elif ch == "a" or k == 2424832:
            frame_idx = max(0, frame_idx - 1)
        elif ch == "w" or k == 2490368:
            frame_idx = min(last, frame_idx + 10)
        elif ch == "s" or k == 2621440:
            frame_idx = max(0, frame_idx - 10)
        elif ch == "h":
            toggle(labels, "hits", frame_idx)
            save_labels(args.out, labels)
        elif ch == "b":
            toggle(labels, "bounces", frame_idx)
            save_labels(args.out, labels)
        elif ch == "n":
            frame_idx = next_event(labels, frame_idx, +1)
        elif ch == "p":
            frame_idx = next_event(labels, frame_idx, -1)

    save_labels(args.out, labels)
    cap.release()
    cv2.destroyAllWindows()
    print(f"saved {args.out}: {len(labels['hits'])} hits, {len(labels['bounces'])} bounces")


if __name__ == "__main__":
    main()
