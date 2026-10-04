# Tennis Analysis

Analyses a tennis match video: tracks the two players and the ball, locates the court, and
reports shot speed, player speed, distance covered, hits, serves, rallies and bounces
(with in/out indication), both as an annotated video and as CSV/JSON/heatmaps.

## How it works

1. **Court**: a ResNet50 keypoint model predicts 14 court points on several frames; the median
   is used and a **homography** maps image pixels to court meters (`court_geometry.py`).
   With `--court-segment-seconds N` it is re-estimated every N seconds for moving cameras.
2. **Players**: YOLOv8 tracking; the two players are chosen *per frame* from where people stand
   on the court (umpire / ball kids are ignored; tracker ID switches do not matter).
   Player 1 = far side, Player 2 = near side.
3. **Ball**: fine-tuned YOLO (highest-confidence box per frame, optional tiled retry), then
   outlier rejection and interpolation only across short gaps (never invents long trajectories).
4. **Events** (`analytics/events.py`): hits (velocity change with the ball in a player's
   contact zone), bounces (upward kink in vertical velocity, refined to sub-frame), serves and
   rallies (time gaps), all scaled by the real FPS.
5. **Stats** (`analytics/stats.py`): speeds on the court plane via the homography.

## Install & run

```bash
pip install -r requirements.txt
# put the models in models/ (see models/put_models_here.txt)
python main.py --input input_videos/input_video.mp4
```

Useful options: `--court-keypoints-json`, `--court-segment-seconds`, `--ball-imgsz`,
`--ball-tile-fallback`, `--no-cache`, `--refresh-cache`, `--output-dir`. See `python main.py -h`.

Outputs in `--output-dir`: `shots.csv`, `bounces.csv`, `frame_stats.csv`, `summary.json`,
`heatmap_player_{1,2}.png`, plus the annotated video.

Detections are cached in `tracker_stubs/*.json`, keyed on the video content and the model, so
changing the video or model recomputes instead of silently reusing stale results.

## Accuracy: what to expect and known limits

* **Ball speed is an estimate.** It is the average speed from the hitter's contact point to the
  bounce point (ground-anchored, plus an assumed 1 m contact height). Off-the-racket speed is
  typically 10-20 % higher. Validate against radar data before trusting absolute numbers.
* **Hit timing is about +-1 frame.** Far-side contacts look smooth in the image, so on a ~20 frame
  flight this is roughly +-5 % in speed. Footage at 50-60 fps improves it directly.
* **Bounce in/out is indicative only.** At 30 fps a fast ball moves ~1 m per frame; the sub-frame
  refinement helps but this is not line calling.
* Static or slowly moving cameras only unless `--court-segment-seconds` is used; broadcast footage
  with cuts to other angles is not handled (segment re-estimation does not detect cuts).
* A hit must have the ball within 6.5 m of a player on the court plane (`max_hit_distance_m`), which stops a bounce in front
  of a player being read as a hit. That limit was tuned on one clip; check it on yours.
* Bounces are only found where the ball's vertical velocity visibly kinks; on the sample clip 1 of 3 labelled bounces was found.
* Volleys, slices and drop shots are not separated from groundstrokes (see shot types below).

## Shot types and live mode

* `--shot-types` adds a `shot_type` column (forehand, backhand, serve, overhead) from pose at contact. It downloads a pose
  model on first use. Use `--left-handed 1,2` for left-handers (1 = far side, 2 = near side). On the sample clip it got the two
  shots I could verify right (a forehand and a two-handed backhand) and returned `unknown` for the far player, whose pose is too
  small to read.
* `python live.py --source 0 --display` (or a stream URL or file) runs the analysis in near real time on a rolling window. Stats
  trail the picture by about a second and a static camera is assumed. With the default heavy models it processed only 1.8 fps on
  the development machine (heavily loaded at the time), so use lighter models (`--player-model yolov8s`) and a GPU for real use.

## Tests and evaluation

```bash
python -m pytest tests -q
```

The tests run the whole analysis on a synthetic rally with known ground truth (hits, bounces,
speeds, serves, running speed), including dropouts, false positives and keypoint noise. They do
**not** prove accuracy on real footage. For that, label some clips and use `evaluation/`
(see `evaluation/README.md`): ball P/R, event F1, speed MAE, keypoint error in meters, IDF1.

## Training

* Ball detector: `training/tennis_ball_detector_training.ipynb` (or `training/train_ball_detector.py`)
* Court keypoints: `training/tennis_court_keypoints_training.ipynb`

Pretrained weights: [ball detector](https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view?usp=sharing),
[court keypoints](https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing)

## Layout

```
main.py / pipeline.py      CLI and orchestration (two streaming passes, constant memory)
config.py                  every tunable in one place
court_geometry.py          court model in meters + pixel->court homography
trackers/                  player + ball detection, cleaning, player selection
court_line_detector/       keypoint model + robust multi-frame / per-segment estimation
analytics/                 events, stats, CSV/JSON/heatmap export
mini_court/                top-down overlay
evaluation/                metrics against ground truth
tests/                     unit + synthetic end-to-end tests
```
