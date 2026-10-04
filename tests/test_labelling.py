import json

from tools.label_events import load_labels, next_event, save_labels, toggle
from utils.video_utils import VideoInfo


def test_toggle_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "l.json")
    info = VideoInfo(30.0, 1920, 1080, 214)
    labels = load_labels(path, "v.mp4", info)
    assert labels["hits"] == [] and labels["fps"] == 30.0

    assert toggle(labels, "hits", 50) is True
    assert toggle(labels, "hits", 10) is True
    assert labels["hits"] == [10, 50]                 # kept sorted
    assert toggle(labels, "hits", 10) is False        # second press removes it
    toggle(labels, "bounces", 30)
    save_labels(path, labels)

    again = load_labels(path, "v.mp4", info)
    assert again["hits"] == [50] and again["bounces"] == [30]
    assert json.load(open(path))["frame_count"] == 214


def test_next_event_navigation():
    labels = {"hits": [10, 50], "bounces": [30]}
    assert next_event(labels, 10, +1) == 30
    assert next_event(labels, 30, +1) == 50
    assert next_event(labels, 50, +1) == 50           # nothing later: stay
    assert next_event(labels, 30, -1) == 10
