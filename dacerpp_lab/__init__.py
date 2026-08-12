"""dacerpp_lab : Isaac Lab 연동 계층.

isaaclab 가 없는 환경(예: 트랙 미리보기/단위테스트)에서도 순수 파이썬 모듈
(tracks, vectorized_track, track_field)은 import 되도록, 무거운 env 모듈은 지연 로드.
"""
from .tracks import (TrackParams, NUM_LAYOUTS, make_centerline, make_width_profile,
                     build_track_mesh)
from .track_field import TrackField, TrackFieldCfg

try:  # isaaclab 필요 모듈
    from .env_cfg import RacingEnvCfg, RacingCfg
    from .racing_env import RacingEnv
except Exception:  # pragma: no cover
    RacingEnvCfg = RacingCfg = RacingEnv = None