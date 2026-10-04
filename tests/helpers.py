from config import Config
from court_line_detector.court_line_detector import CourtModel
from tests import synthetic as S
from utils.video_utils import VideoInfo
import pipeline


def court_model_for(n_frames, kp_noise=0.0):
    H = S.CourtHomography(S.keypoints_flat(noise_px=kp_noise))
    return CourtModel([{"start": 0, "end": n_frames, "keypoints": S.keypoints_flat(), "homography": H}])


def run_scene(cfg=None, kp_noise=0.0, fps=S.FPS, **scene_kw):
    sc = S.build_scene(**scene_kw)
    cm = court_model_for(sc["n_frames"], kp_noise)
    info = VideoInfo(fps, S.W, S.H, sc["n_frames"])
    return sc, pipeline.analyse(cfg or Config(), info, sc["players"], sc["ball"], cm), cm, info
