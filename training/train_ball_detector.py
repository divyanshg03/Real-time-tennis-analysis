"""Fine-tune the ball detector with augmentation suited to a tiny, fast, blurry object.

NOT RUN in this repo's audit (needs a GPU and the dataset); treat as a starting point.

    python training/train_ball_detector.py --data training/tennis-ball-detection-6/data.yaml

Keep a hold-out set from *different matches/surfaces* than the training data and score it with
evaluation/ - the bundled Roboflow set (clay, one player, synthetic frames) is small and narrow.
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280, help="the ball is a few pixels: train large")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        # photometric: lighting / court colour / broadcast grading
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.4,
        # geometric: keep scale/translate modest so the tiny ball is not cropped away
        scale=0.3, translate=0.1, degrees=0.0, fliplr=0.5, flipud=0.0,
        # motion blur is the main failure mode; mosaic helps small-object recall
        mosaic=1.0, close_mosaic=10, mixup=0.0,
        patience=30,
    )
    metrics = model.val()
    print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
