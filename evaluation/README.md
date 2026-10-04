# Evaluation

Accuracy can only be improved if it can be measured. This folder scores each component
against hand-labelled ground truth.

## Ground-truth file (JSON, all sections optional)

```json
{
  "fps": 30,
  "hits":   [30, 60, 90],
  "bounces": [51, 81],
  "shot_speeds": [{"frame": 30, "kmh": 148.0}],
  "ball": [[x1, y1, x2, y2], null, ...],
  "keypoints": [x0, y0, x1, y1, ... 28 numbers],
  "players": [{"1": [x1, y1, x2, y2], "2": [...]}, ...]
}
```

* `hits` / `bounces`: frame numbers where a racket contact / ground bounce happens.
* `shot_speeds`: radar or broadcast speeds, matched to the nearest detected hit.
* `ball`: per-frame ball box or `null`.
* `keypoints`: the 14 court keypoints in pixels, in the order used by `court_geometry.py`.
* `players`: per-frame `{id: bbox}` for the two players (ids are arbitrary but consistent).

## Running

```bash
python main.py --input clip1.mp4 --output-dir runs/clip1
python -m evaluation.evaluate --gt labels/clip1.json --pred-dir runs/clip1
```

or from Python, `evaluation.evaluate.evaluate_run(...)` accepts the in-memory outputs
(ball array, keypoints, players) so ball / keypoint / tracking metrics can be scored too.

## Metrics

| Component | Metric |
|---|---|
| Ball detection | precision, recall, F1 (center within 10 px), mean px error |
| Hits, bounces | F1 with a frame tolerance, mean (abs) frame error |
| Shot speed | MAE km/h, MAPE, bias |
| Court keypoints | mean px error, PCK@5/10/20 px, **court-plane error in meters** |
| Player tracking | IDF1 / IDP / IDR |

Court-plane error is the one that matters most: it is what the predicted homography does to
real positions, and so to every speed and position the pipeline reports.

## Hold-out protocol

Label clips from matches **not** used for training (other surface, other camera angle). Track
the metrics per clip and keep a table in the PR for any change to detection, geometry or event
logic, so regressions are visible.
