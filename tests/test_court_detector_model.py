import numpy as np
import torch
from torchvision import models

from court_line_detector.court_line_detector import CourtLineDetector


def test_detector_loads_checkpoint_in_eval_mode_and_predicts(tmp_path):
    net = models.resnet50(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, 28)
    path = str(tmp_path / "kp.pth")
    torch.save(net.state_dict(), path)

    det = CourtLineDetector(path)
    assert not det.model.training          # BatchNorm must use running stats at inference
    img = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    a = det.predict(img)
    b = det.predict(img)
    assert a.shape == (28,)
    assert np.allclose(a, b)               # deterministic: no batch-stat dependence
