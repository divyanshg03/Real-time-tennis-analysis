"""JSON detection cache keyed on the video + model identity.

Replaces the old pickle "stubs": a stale cache for a different video/model is
detected and recomputed instead of being silently reused, and JSON can't
execute code on load.
"""
import hashlib
import json
import os


def video_fingerprint(video_path):
    st = os.stat(video_path)
    h = hashlib.sha1()
    h.update(f"{os.path.basename(video_path)}|{st.st_size}|{int(st.st_mtime)}".encode())
    with open(video_path, "rb") as f:
        h.update(f.read(1 << 20))  # first MB guards against same-size replacements
    return h.hexdigest()


def load_cache(path, meta):
    """Return the cached payload if ``meta`` matches what was stored, else None."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return None
    if blob.get("meta") != meta:
        return None
    return blob.get("data")


def save_cache(path, meta, data):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"meta": meta, "data": data}, f)


def make_meta(video_path, model, kind, version=1):
    """Identity of a cached result: which video, which model, which kind of detections."""
    return {"video": video_fingerprint(video_path), "model": str(model), "kind": kind, "version": version}
