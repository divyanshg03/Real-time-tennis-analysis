from utils.cache import load_cache, save_cache


def test_cache_invalidated_by_meta_change(tmp_path):
    p = str(tmp_path / "c.json")
    save_cache(p, {"video": "a", "model": "m1"}, [1, 2, 3])
    assert load_cache(p, {"video": "a", "model": "m1"}) == [1, 2, 3]
    assert load_cache(p, {"video": "b", "model": "m1"}) is None     # other video
    assert load_cache(p, {"video": "a", "model": "m2"}) is None     # other model
    assert load_cache(str(tmp_path / "missing.json"), {}) is None
