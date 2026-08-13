#!/usr/bin/env python3
"""대회 코스 변형본 1024개 생성 (competition_tracks/<name>/<name>_centerline.csv).

■ 배경 (2026-07-30 실차 분석 rl_controller_analysis_20260730.txt 원인 A)
  구 생성기는 hangeong 중심선을 '규정 도면 치수'(22 x 8 m)로 리스케일했다. 그러나
  실제 설치·SLAM 된 코스(연습 맵 lobby_0723/0728/0730)는 16.4~16.9 x 5.6~5.9 m 로,
  구 생성물은 실제보다 1.4배 크고 곡률이 절반(|κ|max 중앙 0.90 vs 실제 1.85)이었다.
  = 실전 헤어핀(V자)이 학습에 사실상 없었고, 그래서 실차가 V자에서 벽에 박았다.

■ 2026-08-13 확장 — 끝단 노치가 V자만이 아닐 수 있다 (U자 / ㄷ자)
  대회 코스 계열은 전부 '긴 루프 + 장축 한쪽 끝의 재진입 노치' 구조이고, 그 노치가
  바로 실차가 박던 V자 부분이다. 신규 연습 맵 0813 은 그 노치가 **ㄷ자**(안쪽 섬 끝이
  직사각형 -> 중심선이 손가락을 감아 도는 180도 반환점)로 바뀌었다.
    측정(로드 후 카운터턴 = 누적회전 Ψ 의 낙차):
      0723~0806 48~69도 / 0811 107도 / 0812 87도  <- 전부 V자
      0813                                  160도  <- U/ㄷ 급 반환점
    그런데 기존 절차 생성본 504종의 카운터턴은 최대 107도, >=150도가 **0%** 였다.
    = 2026-07-30 사고와 같은 구조의 노출 공백이 U/ㄷ 에 그대로 남아 있었다.
  그래서 V 밴드는 그대로 두고(검증된 분포), 노치만 해석적으로 교체하는 경로를 추가해
  U자/ㄷ자 밴드를 신설했다(morph_notch). 배분 V 50% / U 25% / ㄷ 25%.

■ 이 생성기의 원칙
  (1) 시드 형상 = 실제 코스 10종(hangeong + 연습 맵 9개).
  (2) 크기 = 연습 맵 실측(장변 16.4~17.3 / 단변 5.3~5.9 m)을 넉넉히 브래킷
      (장변 13~24 / 단변 4.1~11 m). 곡률은 크기에 반비례하므로 실측 크기로
      되돌리는 것만으로 곡률이 실제 수준으로 회복된다.
  (3) 날카로움 스펙트럼 = '대형/완만'(|κ|max 1.18)부터 '초급함'(2.85, 실측
      최대 2.78을 넘어섬)까지 밴드 배분. V 는 5밴드(급함+초급함이 지배),
      U/ㄷ 는 각 4밴드(대형/완만/중간/급함)로 1.1~2.6 을 덮는다.
  (4) ★검증은 '로드 후'(process_measured_centerline)로 한다. 학습 로더가 CSV 를
      읽을 때 11점 평활을 또 걸어 곡률이 줄므로, 생성 CSV 의 곡률이 아니라 '학습이
      실제 보는 곡률'로 밴드 판정해야 목표 분포가 학습에 그대로 들어간다.
      실측 맵은 좌/우 폭을 반드시 따로 넣는다(post_load_stats 주석 참조).
  (5) 연습 맵 전부는 competition 1024 안에 '그대로'(실측 치수·폭) 포함한다.
  (6) ★생성 분포가 실측 연습 맵을 '감싸는지'를 매 실행 말미에 지표별로 찍는다.
      2026-08-13 기준 9종 x 8지표(장/단/랩/|κ|max/|κ|p99/폭min/R+hw/카운터턴).
      카운터턴은 '같은 형상 계열'끼리 대조한다(V 맵 8종 vs V 밴드, 0813 vs ㄷ 밴드).

생성물은 f1tenth CSV 포맷(x_m, y_m, w_tr_right_m, w_tr_left_m)이라 기존 로더가
그대로 읽고, TrackField 가 축소 없이 실측 맵과 동급으로 취급한다.

■ 연습 맵 탐색 경로 (LOBBY_DIRS)
  generated/map_lobby/map_lobby_*/  (0723~0806, 0813) 와 generated/maps/lobby_*/ (0811~)
  두 곳을 모두 훑는다. 폴더명의 "map_" 접두사는 떼고 "comp_" 를 붙여 이름을 만든다
  (map_lobby_0813 -> comp_lobby_0813, lobby_0811 -> comp_lobby_0811).

실행:
  python scripts/make_competition_tracks.py --clean          # 1024개(기본) 전체 재생성
  python scripts/make_competition_tracks.py --add_ud         # ★기존 comp### 보존 +
       연습 맵 갱신 + 부족분을 U/ㄷ 로만 채워 --n 까지 증설(2026-08-13 확장에 사용).
  python scripts/make_competition_tracks.py --n 60 --seed 7
  python scripts/make_competition_tracks.py --reals_only     # 연습 맵만 갱신/추가
       (절차 생성본 comp### 는 건드리지 않는다 — 새 연습 맵을 기존 세트에
        끼워 넣을 때 사용. 총 개수를 유지하려면 comp### 를 그만큼 지울 것.)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dacerpp_lab.tracks import (TrackParams, _resample_equal_arclength,
                                _periodic_catmull_rom, centerline_features,
                                _polyline_self_intersects,
                                process_measured_centerline)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "competition_tracks")
HANGEONG = os.path.join(ROOT, "f1tenth_racetracks", "hangeong", "hangeong_centerline.csv")
# 연습 맵(global_waypoints.json) 탐색 루트 — 두 곳 모두 하위 1단계를 훑는다.
LOBBY_DIRS = [os.path.join(ROOT, "generated", "map_lobby"),   # 0723~0806
              os.path.join(ROOT, "generated", "maps")]        # 0811~ (신규 반입 위치)

# 크기(m): 연습 맵 실측 장변 16.4~16.9 / 단변 5.6~5.9 를 넉넉히 브래킷.
# 장변 하한 13 = 실측보다 작게(더 급한 코스), 상한 20 = 실측보다 크게(더 완만).
SPEC_LONG_RANGE = (13.0, 20.0)
SPEC_SHORT_RANGE = (4.5, 7.5)
# 폭(m): 실측 코리도 0.95~2.62 를 포함. 좁은 V apex 는 로더의 곡률 클립이 자동으로
# 좁히므로(실측 맵도 동일 처리), 여기서 규정 1 m 하한을 강제하지 않는다.
# ★2026-08-12 하한 0.90 -> 0.80: 구 값으론 '로드 후' 최소 폭이 0.878 m 까지만 내려가
#   실측 연습 맵(0806=0.825, 0811=0.854, 0812=0.870)의 최협부를 절차 생성이 못 덮었다.
#   실측 6~8종이 0.825~1.075 이므로 0.80~1.35 가 실측을 제대로 브래킷한다.
WIDTH_LO_RANGE = (0.80, 1.35)        # 전체 최소 폭 하한
WIDTH_GAP_RANGE = (0.60, 1.60)       # 최대 = 최소 + 이 값
WIDTH_CLIP_LO = 0.78                 # make_width 절대 하한(로더가 여기서 더 깎는다)

# 통과 가능성 여유(m): 바깥 라인 반경 R_curv + hw >= wp_veh_r_min(0.76) + 이 값.
# ★2026-08-12 P.wp_outer_margin(0.10) -> 0.03. 실측 연습 맵 0811 의 최악 지점이
#   R+hw = 0.801 m 라, 0.10 여유를 강제하면 '실제로 설치·주행된 코스'가 생성 기준에서
#   탈락한다(구 절차 하한 0.950 vs 실측 0.801). 0.03 = 하한 0.79 로 실측 0.801 을
#   분포 안쪽에 둔다. 차량 기하 최소회전반경 0.76 m 자체는 여전히 지키므로
#   기하학적 통과는 보장되고, 여유만 실측 수준으로 좁히는 것이다.
#   ※ 이 값을 더 낮추면 R+hw < 0.76 = 물리적으로 못 도는 코스가 나온다. 하한 사수.
PASS_MARGIN = 0.03

# V자 날카로움 밴드: (이름, '로드 후' |κ|max 목표, 저주파 지터진폭, bbox 장변 범위(m))
# ★2026-07-31 원해상도 방식으로 교체: 구 방식(제어점 다운샘플→Catmull-Rom)은 급한 V
#   apex 를 뭉개 |κ|max 가 실측(1.77~1.89)에 못 미쳤다(평활 0 에서도 1.63). 실측 중심선을
#   원해상도로 직접 변형(저주파 지터+bbox)하면 apex 가 보존돼 실측을 재현/초과한다(측정:
#   장변 16m→~1.9, 18m→~1.8, 14m→~2.1). 날카로움은 주로 bbox 장변으로 조절(작을수록 급함).
# 급함 상한 1.98 = 통과 한계(R+hw>=0.86 & hw<=0.7R → 1.7R>=0.86 → R>=0.51 → |κ|<=1.98).
#   실측 1.89 가 이 바로 아래이므로 급함 밴드는 실측과 동급이다.
# (이름, '로드 후' |κ|max 목표, 저주파 지터진폭, bbox 장변 범위(m), 종횡비(장/단) 범위,
#  시드 풀 = 이 밴드가 쓸 시드 형상 이름들(None = 완만~대형용 기본 7종))
# ★크기-곡률 결합(측정): bbox 가 클수록 곡률이 낮다 — 실측 practice 16m→|κ|max 1.85,
#   규정 22x8→1.52, 여유상한 24x11→1.25. 즉 규정 크기 코스 자체가 practice(16m)보다 완만.
#   크기 상한을 24x11 로 넓히면 곡률 범위가 아래로 확장된다(완만한 대형 맵 추가) — 물리적
#   불가피. sharp V(소형 practice)는 유지하고 대형/완만을 tail 로 더해 race-day 큰 코스 대비.
#
# ★2026-08-12 '초급함' 밴드 신설 — 신규 연습 맵 0811/0812 대응.
#   0811/0812 는 '로드 후' |κ|max 2.78 / 2.43, |κ|p99 2.15 / 1.99, min(R+hw) 0.80 / 0.86
#   으로, 구 4밴드(상한 1.98)가 단 한 종도 덮지 못했다. 헤어핀 apex 가 기존 연습 맵보다
#   한 단계 더 날카로워진 것이며, 이 노출이 0% 면 2026-07-30 실차 사고와 같은 구조의
#   갭이 남는다.
#   ■ 시드 풀 분리가 필수인 이유(측정): perturb 는 원해상도 apex 를 보존하므로 시드의
#     날카로움이 그대로 따라온다. 같은 bbox 장변 16.5m 에서 |κ|max 중앙값이
#       hangeong 1.93 / 0723~0806 1.97~2.14 / 0811 3.05 / 0812 2.60
#     로 갈린다. 즉 (a) 기존 7종 시드로는 bbox 를 13m 아래로 줄이지 않는 한 1.98 을
#     넘기지 못하고(그러면 랩이 34m 로 실측 44m 에서 벗어난다), (b) 반대로 0811/0812
#     시드를 완만/대형 밴드에 섞으면 그 밴드의 목표 곡률에 영원히 도달하지 못해
#     생성이 relax 경로로 새 버린다. 그래서 밴드마다 시드 풀을 고정한다.
SEEDS_BASE = ("hangeong", "lobby_0723", "lobby_0728", "lobby_0730",
              "lobby_0731", "lobby_0804", "lobby_0806")
SEEDS_ULTRA = ("lobby_0811", "lobby_0812")
# U/ㄷ 밴드는 노치를 통째로 해석적 캡으로 갈아끼우므로 시드의 노치 형상이 무의미하다
# -> 실측 10종 전부를 시드 풀로 쓴다(나머지 구간의 다양성만 시드에서 얻는다).
SEEDS_ALL = SEEDS_BASE + SEEDS_ULTRA + ("lobby_0813",)

# ---- 노치(V자 부분) 형상 ----
# kind: "V" = 시드 노치를 그대로 변형(기존 경로), "U"/"D" = 노치를 해석적 캡으로 교체.
# 노치 파라미터 rn = 카운터턴(반환점) 반경. |κ|max 를 사실상 이 값이 정한다.
#   U: 반경 rn 의 단일 180도 호(둥근 손가락 끝)
#   D(ㄷ): 반경 rn 의 90도 코너 2개 + 그 사이 직선 cross (직사각 손가락 끝)
# r_side = 노치 양옆 코너 반경(손가락을 감아 도는 바깥 회전).
NOTCH_V, NOTCH_U, NOTCH_D = "V", "U", "D"

# (이름, '로드 후' |κ|max 목표, 저주파 지터진폭, bbox 장변 범위(m), 종횡비(장/단) 범위,
#  시드 풀, kind, 노치 캡 파라미터(kind=V 면 None))
BANDS = [
    ("완만", (1.40, 1.68), (0.010, 0.045), (18.0, 21.0), (2.6, 3.2), SEEDS_BASE, NOTCH_V, None),
    ("중간", (1.68, 1.85), (0.010, 0.050), (16.0, 18.2), (2.7, 3.2), SEEDS_BASE, NOTCH_V, None),   # 실측(1.77~1.89) 대역
    ("급함", (1.85, 1.98), (0.015, 0.060), (14.0, 16.4), (2.8, 3.3), SEEDS_BASE, NOTCH_V, None),   # 실측 practice~구 통과 한계
    ("대형", (1.18, 1.52), (0.008, 0.035), (21.0, 24.0), (2.15, 2.75), SEEDS_BASE, NOTCH_V, None),  # 규정 22x8~여유 24x11
    # 초급함: 실측 0811(2.78)·0812(2.43) 를 안쪽에 두도록 1.98~2.85. bbox 는 실측과
    # 같은 15~19m 로 두어 랩 길이(≈40~48m)와 코스 규모는 실전 그대로 유지하고,
    # 날카로움만 시드 apex 에서 얻는다(= 실측 맵이 실제로 그런 형상이다).
    ("초급함", (1.98, 2.85), (0.015, 0.060), (15.0, 19.0), (2.5, 3.3), SEEDS_ULTRA, NOTCH_V, None),
    # ---- U자(둥근 반환점) 4밴드 : |κ|max 1.15~2.60 ----
    ("U대형", (1.15, 1.50), (0.008, 0.035), (20.0, 23.5), (2.15, 2.75), SEEDS_ALL, NOTCH_U,
     dict(rn=(0.85, 1.20), r_side=(0.85, 1.50))),
    ("U완만", (1.40, 1.75), (0.010, 0.045), (17.5, 20.5), (2.6, 3.2), SEEDS_ALL, NOTCH_U,
     dict(rn=(0.78, 1.10), r_side=(0.78, 1.40))),
    ("U중간", (1.75, 2.05), (0.010, 0.050), (15.5, 18.5), (2.6, 3.2), SEEDS_ALL, NOTCH_U,
     dict(rn=(0.66, 0.90), r_side=(0.70, 1.25))),
    ("U급함", (2.05, 2.60), (0.015, 0.060), (13.5, 16.0), (2.7, 3.3), SEEDS_ALL, NOTCH_U,
     dict(rn=(0.52, 0.75), r_side=(0.62, 1.10))),
    # ---- ㄷ자(직사각 반환점) 4밴드 : 실측 0813(|κ|max 1.86, 카운터턴 160도)이 'ㄷ중간' 안쪽 ----
    ("ㄷ대형", (1.10, 1.45), (0.008, 0.035), (20.0, 23.5), (2.15, 2.75), SEEDS_ALL, NOTCH_D,
     dict(rn=(0.85, 1.20), r_side=(0.85, 1.50), cross=(0.70, 1.80))),
    ("ㄷ완만", (1.40, 1.72), (0.010, 0.045), (17.5, 20.5), (2.6, 3.2), SEEDS_ALL, NOTCH_D,
     dict(rn=(0.75, 1.05), r_side=(0.78, 1.40), cross=(0.60, 1.70))),
    ("ㄷ중간", (1.72, 2.00), (0.010, 0.050), (15.5, 18.5), (2.6, 3.2), SEEDS_ALL, NOTCH_D,
     dict(rn=(0.62, 0.88), r_side=(0.70, 1.25), cross=(0.55, 1.60))),
    ("ㄷ급함", (2.00, 2.55), (0.015, 0.060), (13.5, 16.0), (2.7, 3.3), SEEDS_ALL, NOTCH_D,
     dict(rn=(0.50, 0.72), r_side=(0.62, 1.10), cross=(0.45, 1.40))),
]
# 전체 재생성(--clean) 배분: 16슬롯 = V 8 + U 4 + ㄷ 4 => V 50% / U 25% / ㄷ 25%.
# V 8슬롯 안의 비율은 기존과 동일(급함 3, 초급함 2, 완만/중간/대형 각 1) — 검증된 분포를
# 그대로 유지한다. 형상이 고루 섞이도록 V/U/ㄷ 를 교대로 배치한다.
BAND_PATTERN = (0, 6, 2, 10, 1, 7, 2, 11, 3, 8, 2, 12, 4, 5, 4, 9)
# --add_ud(증설) 배분: 기존 comp### 504종(전부 V)과 연습 맵 8종(V)을 보존한 채
# 부족분을 U/ㄷ 로만 채운다 => 512(기존 V) + 512(신규 U/ㄷ+0813) = 1024, 정확히 50/50.
BAND_PATTERN_UD = (5, 9, 6, 10, 7, 11, 8, 12)
SHORT_MAX = 11.0                 # 단변 상한(m)

# ---- U/ㄷ 전용 수용 기준 (V 밴드는 기존 기준 그대로 — 검증된 분포를 흔들지 않는다) ----
COUNTER_UD_MIN = 145.0   # '로드 후' 카운터턴(도) 하한. 실측 0813 = 160도 가 분포 안쪽.
                         # 이 밑은 로더 평활에 노치가 뭉개져 U/ㄷ 로 안 읽히는 것들이다.
WIDTH_MIN_UD = 0.75      # '로드 후' 최소 폭(m). 기존 절차 생성본 최소 0.73 envelope 안.
LAP_MAX_UD = 70.0        # 랩 상한(m). 기존 절차 생성본 최대 69.7 envelope 안.

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1024, help="총 트랙 수(연습 맵 포함)")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--out", type=str, default=OUT_DIR)
parser.add_argument("--clean", action="store_true", help="기존 생성물 삭제 후 재생성")
parser.add_argument("--reals_only", action="store_true",
                    help="연습 맵(comp_lobby_*)만 쓰고 종료 — 절차 생성본은 그대로 둔다")
parser.add_argument("--add_ud", action="store_true",
                    help="기존 comp### 를 보존한 채 부족분을 U/ㄷ 밴드로만 증설")
args = parser.parse_args()

P = TrackParams()
DS = P.ds


def _smooth_closed(xy: np.ndarray, win: int = 11) -> np.ndarray:
    h = win // 2
    k = np.ones(win) / win
    pad = np.vstack([xy[-h:], xy, xy[:h]])
    return np.stack([np.convolve(pad[:, d], k, mode="valid") for d in (0, 1)], 1)[:len(xy)]


def _pca_align(xy: np.ndarray) -> np.ndarray:
    """장축을 x 로 정렬(이후 bbox 종횡비 조작을 규정 치수 기준으로)."""
    c = xy - xy.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    al = c @ vt.T
    if al[:, 0].ptp() < al[:, 1].ptp():
        al = al[:, ::-1]
    return al


def find_lobby_maps() -> list[str]:
    """LOBBY_DIRS 전체에서 <맵>/global_waypoints.json 을 찾아 맵 이름순으로 반환."""
    found = []
    for d in LOBBY_DIRS:
        found += glob.glob(os.path.join(d, "*", "global_waypoints.json"))
    return sorted(found, key=lambda p: lobby_name(p))


def lobby_name(jp: str) -> str:
    """<...>/map_lobby_0806/global_waypoints.json -> 'lobby_0806'."""
    return os.path.basename(os.path.dirname(jp)).replace("map_", "")


def load_lobby_map(jp: str):
    """연습 맵 global_waypoints.json -> (중심선 xy, w_right, w_left)."""
    w = json.load(open(jp))["centerline_waypoints"]["wpnts"]
    xy = np.array([[p["x_m"], p["y_m"]] for p in w], float)
    w_r = np.array([p["d_right"] for p in w], float)
    w_l = np.array([p["d_left"] for p in w], float)
    if np.linalg.norm(xy[0] - xy[-1]) < 1e-6:
        xy, w_r, w_l = xy[:-1], w_r[:-1], w_l[:-1]
    return xy, w_r, w_l


def _resample_full(xy):
    """원해상도(ds 간격) 등호장 리샘플 — 다운샘플 없이 apex 보존."""
    tot = np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1).sum()
    return _resample_equal_arclength(xy, max(64, int(round(tot / DS))))


def load_seed_shapes():
    """시드 형상 4종(원해상도 PCA 정렬 xy). ★평활하지 않는다 — 급한 V apex 를 그대로
    남겨야 실측(1.77~1.89) 곡률이 재현된다. SLAM 노이즈는 로더(process_measured_centerline)
    의 11점 평활이 처리(실측 맵과 동일 경로)."""
    shapes = []
    import csv as _csv
    rows = np.array([[float(v) for v in r[:2]] for r in _csv.reader(open(HANGEONG))
                     if r and not r[0].lstrip().startswith("#")])
    xy = rows if np.linalg.norm(rows[0] - rows[-1]) > 1e-6 else rows[:-1]
    shapes.append(("hangeong", _pca_align(_resample_full(xy))))
    for jp in find_lobby_maps():
        nm = lobby_name(jp)
        xy, _, _ = load_lobby_map(jp)
        shapes.append((nm, _pca_align(_resample_full(xy))))
    return shapes


def perturb(base: np.ndarray, rng, band):
    """실측 중심선을 '원해상도로 직접' 변형 — 저주파 법선 지터 + bbox 스케일 + 미러.
    구 방식의 제어점 다운샘플/Catmull-Rom(=apex 뭉갬)을 제거해 급한 V 곡률을 보존한다.

    반환 (중심선, 이번에 쓴 장변 long_m) — U/ㄷ 는 캡 교체 뒤 rescale_long 으로 그 값에
    되맞춰야 하므로 호출자가 알아야 한다."""
    amp_r, long_r, asp_r = band[2], band[3], band[4]
    cl = _pca_align(_resample_full(base))              # 원해상도 유지
    n = len(cl)
    # 저주파 법선 지터: 큰 창으로 스무딩한 노이즈 -> 전체 형상만 완만히 바꾸고 apex 는 보존
    seg = np.roll(cl, -1, 0) - cl
    psi = np.arctan2(seg[:, 1], seg[:, 0])
    nrm = np.stack([-np.sin(psi), np.cos(psi)], 1)
    win = max(15, n // 10) | 1                          # 홀수 큰 창(저주파화)
    h = win // 2
    noise = np.convolve(np.concatenate([(nz := rng.normal(0, 1, n))[-h:], nz, nz[:h]]),
                        np.ones(win) / win, mode="valid")[:n]
    ext = 0.5 * (cl[:, 0].ptp() + cl[:, 1].ptp())
    cl = cl + nrm * (rng.uniform(*amp_r) * noise * ext)[:, None]
    # bbox: 장변 = long_m(밴드), 종횡비(밴드) — 크기와 종횡비가 곡률을 결정. 단변 상한 SHORT_MAX.
    al = _pca_align(cl)
    long_m = rng.uniform(*long_r)
    short_m = min(long_m / rng.uniform(*asp_r), SHORT_MAX)
    cl = al * np.array([long_m / max(al[:, 0].ptp(), 1e-6),
                        short_m / max(al[:, 1].ptp(), 1e-6)])
    if rng.uniform() < 0.5:
        cl[:, 1] *= -1.0
    if rng.uniform() < 0.5:
        cl = cl[::-1].copy()
    return _resample_full(cl), long_m


# ---------------------------------------------------------------------------
# 노치(V자 부분) 형상 교체 — V -> U / ㄷ
#
# 이 코스 계열의 '장축 한쪽 끝'에는 안쪽 섬의 끝이 중심선을 되밀어 만든 재진입 노치가
# 있고, 그게 실차가 박던 V자 부분이다. 여기서는 그 노치 구간만 잘라내고 해석적으로
# 만든 캡(직선+원호 조합)으로 갈아끼운다. 나머지 구간은 실측 시드 그대로 남는다
# = "실측 형상 기반" 이라는 이 생성기의 1원칙을 유지한 채 노치 형상만 바꾼다.
#
#   캡 구성:  [a] arc(+φ1,r_side) [b1] {카운터턴} [b2] arc(+φ2,r_side) [a']
#     카운터턴  U : arc(-θ, rn)                      (둥근 반환점)
#               ㄷ: arc(-θ/2, rn) [cross] arc(-θ/2, rn)   (직사각 반환점)
#     φ1 + φ2 - θ = 원 경로가 그 구간에서 돌던 각(dψ)  -> 접선 연속
# ---------------------------------------------------------------------------
STRAIGHT_K = 0.25        # |κ| 이하 = 직선(창 경계를 직선 위에 두어 접선을 깨끗하게)
STRAIGHT_MIN = 2.0       # 직선 런 최소 길이(m)
COUNTER_DET_MIN = np.radians(35.0)   # 노치 후보 최소 회전량
# 노치 캡 공통 파라미터. 밴드별 rn/r_side/cross 만 BANDS 에서 덮어쓴다.
NOTCH_CFG = dict(
    tries=140,
    leg=(0.45, 1.70),          # 카운터턴 앞뒤 직선(=손가락 길이). ★짧으면 로더 평활이
                               #   카운터턴을 양옆 코너와 뭉개 U/ㄷ 로 안 읽힌다.
    split=(0.36, 0.64),        # φ1 / (φ1+φ2)
    theta_U=(155.0, 200.0),    # 카운터턴 각(도) — 180도 = 완전한 반환점
    theta_D=(160.0, 205.0),
    rn=(0.60, 1.00), r_side=(0.70, 1.30), cross=(0.55, 1.60),
)
# 캡 직선별 (해 탐색 가중치, 하한, 상한). 가중치가 클수록 그 직선이 닫힘 오차를 흡수한다.
_STRAIGHT_BOUND = dict(a=(4.0, -6.0, 8.0), ap=(4.0, 0.05, 8.0),
                       b1=(1.6, 0.25, 3.0), b2=(1.6, 0.25, 3.0), cross=(0.6, 0.20, 2.4))


def _orient_ccw(cl: np.ndarray) -> np.ndarray:
    """총 회전이 +가 되도록 진행 방향 정규화(미러/역주행에도 노치 부호가 고정된다)."""
    f = centerline_features(cl)
    return cl if (f["kappa"] * f["seglen"]).sum() > 0 else cl[::-1].copy()


def _true_runs(flag: np.ndarray):
    """주기 불리언 배열의 True 런 [(시작, 끝)] 과 롤 오프셋."""
    n = len(flag)
    if flag.all() or not flag.any():
        return [], 0
    sh = int(np.argmax(~flag))
    fr = np.roll(flag, -sh)
    out, i = [], 0
    while i < n:
        if fr[i]:
            j = i
            while j < n and fr[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out, sh


def _long_straight_mask(kap: np.ndarray, sl: np.ndarray) -> np.ndarray:
    rs, sh = _true_runs(np.abs(kap) < STRAIGHT_K)
    out = np.zeros(len(kap), bool)
    srl = np.roll(sl, -sh)
    for i, j in rs:
        if srl[i:j].sum() >= STRAIGHT_MIN:
            out[i:j] = True
    return np.roll(out, sh)


def counter_turns(cl: np.ndarray):
    """누적 회전 Ψ(s) 의 하강 구간 = 카운터턴(재진입) 목록 [(하강량rad, i시작, i끝)].

    ★'κ < -임계' 런의 적분으로 재면 안 된다 — ㄷ자 노치는 두 코너 사이가 직선(κ≈0)
      이라 런이 둘로 쪼개져 회전량이 절반으로 측정된다(=ㄷ 를 V 로 오판). Ψ 의
      국소최대 -> 다음 국소최소 낙차로 재면 V/U/ㄷ 를 하나의 잣대로 잴 수 있다.
      이 값이 실측 대조표의 '카운터턴' 열이다(V 48~107도, 0813 160도).
    """
    n = len(cl)
    f = centerline_features(cl)
    psi = np.cumsum(f["kappa"] * f["seglen"])
    psi2 = np.concatenate([psi, psi + psi[-1]])      # 두 바퀴 — 시작점을 걸친 하강도 포착
    d = np.diff(psi2)
    out, i = [], 0
    while i < n:
        if d[i] < 0:
            j = i
            while j < len(d) and d[j] < 0:
                j += 1
            out.append((float(psi2[i] - psi2[j]), i % n, j % n))
            i = j
        else:
            i += 1
    return out


def notch_window(cl: np.ndarray):
    """장축 끝의 재진입 노치를 감싸는 (i0, i1, 카운터턴rad, 긴직선마스크). 실패 시 None.

    ★'가장 큰 카운터턴'만으로 고르면 안 된다 — 이 계열은 장축 중간(|x|≈5m)에도 90도급
      스텝 카운터턴이 있어 실제 V자 부분(장축 끝)과 경쟁한다(실측 8종에서 그쪽이 더 크다).
      끝단성(|x|/반장축)과 회전량을 함께 점수화해 '끝에 있는 큰 카운터턴'을 고른다.
    창의 양끝은 긴 직선 위에 둔다 — 그래야 캡의 진입/탈출 접선이 노이즈 없이 정의된다.
    """
    n = len(cl)
    f = centerline_features(cl)
    long_st = _long_straight_mask(f["kappa"], f["seglen"])
    if not long_st.any():
        return None
    al = _pca_align(cl)
    half = max(al[:, 0].ptp() * 0.5, 1e-6)
    best = None
    for amt, i, j in counter_turns(cl):
        if amt < COUNTER_DET_MIN:
            continue
        idx = (np.arange((j - i) % n + 1) + i) % n
        score = abs(al[idx, 0].mean()) / half + 0.5 * min(amt / np.pi, 1.2)
        if best is None or score > best[0]:
            best = (score, amt, idx[len(idx) // 2])
    if best is None:
        return None
    _, amt, c = best
    i0 = c
    for _ in range(n):
        i0 = (i0 - 1) % n
        if long_st[i0]:
            break
    else:
        return None
    i1 = c
    for _ in range(n):
        i1 = (i1 + 1) % n
        if long_st[i1]:
            break
    else:
        return None
    if (i1 - i0) % n < 12 or (i0 - i1) % n < 40:    # 창이 너무 짧거나 트랙 대부분을 먹으면 기각
        return None
    return i0, i1, float(amt), long_st


def _arc_disp(psi: float, alpha: float, r: float):
    """heading psi 에서 각 alpha(부호 포함)/반경 r 인 원호의 변위와 끝 heading."""
    if alpha == 0.0:
        return np.zeros(2), psi
    k = np.sign(alpha) / r
    return (np.array([np.sin(psi + alpha) - np.sin(psi),
                      np.cos(psi) - np.cos(psi + alpha)]) / k, psi + alpha)


def cap_plan(kind: str, dpsi: float, prm: dict):
    """캡의 원호부 고정 변위 V0 와 직선 방향들. 실패 시 None.

    직선 길이(a, b1, cross, b2, a')는 전부 끝점에 '선형'으로 들어간다 — 원호는 각/반경만
    으로 결정되고, 그 앞의 직선은 뒤따르는 전부를 평행이동시키기 때문. 덕분에 닫힘 조건
    2식을 선형해로 정확히 만족시킬 수 있다(진입/탈출 접선이 거의 반평행이라 a, a' 두 개만
    쓰면 특이해지는데, b1/b2/cross 를 함께 미지수로 두면 그 문제가 사라진다).
    """
    th = prm["th"]
    tot = dpsi + th
    phi1 = tot * prm["split"]
    phi2 = tot - phi1
    if not (np.radians(8) < phi1 < np.radians(330) and np.radians(8) < phi2 < np.radians(330)):
        return None
    psi, V0 = 0.0, np.zeros(2)
    dirs, names = [], []
    dirs.append(np.array([np.cos(psi), np.sin(psi)])); names.append("a")
    d, psi = _arc_disp(psi, phi1, prm["r1"]); V0 = V0 + d
    dirs.append(np.array([np.cos(psi), np.sin(psi)])); names.append("b1")
    if kind == NOTCH_D:
        d, psi = _arc_disp(psi, -0.5 * th, prm["rn"]); V0 = V0 + d
        dirs.append(np.array([np.cos(psi), np.sin(psi)])); names.append("cross")
        d, psi = _arc_disp(psi, -0.5 * th, prm["rn"]); V0 = V0 + d
    else:
        d, psi = _arc_disp(psi, -th, prm["rn"]); V0 = V0 + d
    dirs.append(np.array([np.cos(psi), np.sin(psi)])); names.append("b2")
    d, psi = _arc_disp(psi, phi2, prm["r2"]); V0 = V0 + d
    dirs.append(np.array([np.cos(psi), np.sin(psi)])); names.append("ap")
    return dict(V0=V0, U=np.stack(dirs, 1), names=names, phi1=phi1, phi2=phi2)


def solve_straights(plan: dict, prm: dict, A, psiA, B):
    """캡 끝점이 정확히 B 가 되게 하는 직선 길이들 — 공칭값에서 최소 편차 해. 실패 시 None."""
    c, s = np.cos(-psiA), np.sin(-psiA)
    D = np.array([[c, -s], [s, c]]) @ (B - A)          # 로컬 프레임(진입 heading = 0)
    U, names = plan["U"], plan["names"]
    nom = np.array([prm.get(k, 0.6) for k in names], float)
    w = np.array([_STRAIGHT_BOUND[k][0] for k in names], float)
    W = np.diag(w ** 2)
    G = U @ W @ U.T
    if abs(np.linalg.det(G)) < 1e-9:
        return None
    v = nom + W @ U.T @ np.linalg.solve(G, D - plan["V0"] - U @ nom)
    if not np.isfinite(v).all():
        return None
    for val, k in zip(v, names):
        if val < _STRAIGHT_BOUND[k][1] or val > _STRAIGHT_BOUND[k][2]:
            return None
    return dict(zip(names, v))


def cap_segments(kind: str, prm: dict, plan: dict, L: dict):
    """캡의 (길이, 곡률) 세그먼트 목록."""
    th, rn = prm["th"], prm["rn"]
    segs = [(L["a"], 0.0), (plan["phi1"] * prm["r1"], 1.0 / prm["r1"]), (L["b1"], 0.0)]
    if kind == NOTCH_D:
        segs += [(0.5 * th * rn, -1.0 / rn), (L["cross"], 0.0), (0.5 * th * rn, -1.0 / rn)]
    else:
        segs += [(th * rn, -1.0 / rn)]
    segs += [(L["b2"], 0.0), (plan["phi2"] * prm["r2"], 1.0 / prm["r2"]), (L["ap"], 0.0)]
    return segs


def integrate_segments(segs, ds: float = None):
    """(길이, 곡률) 목록 -> 원점 출발·heading 0 인 로컬 폴리라인."""
    ds = DS if ds is None else ds
    pts = [np.zeros(2)]
    psi, p = 0.0, np.zeros(2)
    for Lg, k in segs:
        if Lg <= 1e-9:
            continue
        m = max(1, int(round(Lg / ds)))
        h = Lg / m
        for _ in range(m):
            p = p + h * np.array([np.cos(psi + 0.5 * k * h), np.sin(psi + 0.5 * k * h)])
            psi += k * h
            pts.append(p.copy())
    return np.asarray(pts)


def sample_cap_params(kind: str, rng, cfg: dict) -> dict:
    p = dict(r1=rng.uniform(*cfg["r_side"]), r2=rng.uniform(*cfg["r_side"]),
             rn=rng.uniform(*cfg["rn"]),
             b1=rng.uniform(*cfg["leg"]), b2=rng.uniform(*cfg["leg"]),
             a=rng.uniform(0.4, 2.0), ap=rng.uniform(0.4, 2.0),
             split=rng.uniform(*cfg["split"]))
    if kind == NOTCH_D:
        p["th"] = np.radians(rng.uniform(*cfg["theta_D"]))
        p["cross"] = rng.uniform(*cfg["cross"])
    else:
        p["th"] = np.radians(rng.uniform(*cfg["theta_U"]))
    return p


def morph_notch(cl: np.ndarray, kind: str, rng, cfg: dict):
    """장축 끝 노치를 kind(U/ㄷ) 캡으로 교체한 중심선. 실패 시 None.

    ★검출·접합은 '평활 사본'에서 한다. 넘어오는 중심선은 실측 원해상도(SLAM 정점 노이즈
      포함)라 |κ| 가 전 구간에서 튀어 직선/카운터턴 판정 자체가 불가능하다. 유지되는
      구간은 원본(sharp) 그대로 이어 붙여 기존 V 밴드와 같은 pre-load 특성을 지킨다.
    """
    cl = _orient_ccw(np.asarray(cl, float))
    sm = _smooth_closed(cl, 11)
    nw = notch_window(sm)
    if nw is None:
        return None
    i0_ref, i1, _, long_st = nw
    n = len(cl)
    f = centerline_features(sm)
    kseg = f["kappa"] * f["seglen"]
    B = sm[i1]
    for _ in range(cfg["tries"]):
        prm = sample_cap_params(kind, rng, cfg)
        i0 = i0_ref
        for _pass in range(3):
            dpsi = kseg[(np.arange((i1 - i0) % n) + i0) % n].sum()
            plan = cap_plan(kind, dpsi, prm)
            if plan is None:
                break
            L = solve_straights(plan, prm, sm[i0], f["psi"][i0], B)
            if L is None:
                break
            if L["a"] >= -0.02:
                L["a"] = max(L["a"], 0.0)
                loc = integrate_segments(cap_segments(kind, prm, plan, L))
                c, s = np.cos(f["psi"][i0]), np.sin(f["psi"][i0])
                world = (np.array([[c, -s], [s, c]]) @ loc.T).T + sm[i0]
                keep = (np.arange((i0 - i1) % n + 1) + i1) % n     # i1 -> i0 (미변경부)
                return _resample_full(np.vstack([cl[keep], world[1:-1]]))
            # a < 0 = 캡이 A 보다 뒤에서 시작해야 한다 -> 창을 직선 쪽으로 물리고 재계산
            j, ok = i0, True
            for _ in range(int(np.ceil(-L["a"] / DS))):
                j = (j - 1) % n
                if not long_st[j]:
                    ok = False
                    break
            if not ok:
                break
            i0 = j
    return None


def rescale_long(cl: np.ndarray, target: float) -> np.ndarray:
    """장변을 target 으로 맞추는 '균일' 축척.

    캡 교체는 손가락이 바깥으로 뻗는 만큼 장변을 늘린다(측정 +0~4 m). perturb 가 잡아 둔
    bbox 를 되돌려야 U/ㄷ 도 V 와 같은 크기 분포에 들어간다. 비균일 축척은 캡의 원호를
    타원으로 만들어 설계한 반경(=통과 가능성)이 깨지므로 반드시 균일 축척으로 되돌린다.
    """
    al = _pca_align(cl)
    return _resample_full(al * (target / max(al[:, 0].ptp(), 1e-6)))


def make_width(n: int, rng) -> np.ndarray:
    """실측형 폭 프로파일(좁은 게이트~넓은 구간). apex 는 로더가 곡률로 클립."""
    u = np.arange(n) / n
    base = (0.6 * np.sin(2 * np.pi * (1 * u + rng.uniform()))
            + 0.3 * np.sin(2 * np.pi * (2 * u + rng.uniform()))
            + 0.1 * np.sin(2 * np.pi * (3 * u + rng.uniform())))
    lo = rng.uniform(*WIDTH_LO_RANGE)
    hi = lo + rng.uniform(*WIDTH_GAP_RANGE)
    w = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.clip(base * 1.25, -1, 1)
    if rng.uniform() < 0.5:                       # 병목 게이트 1~2개
        for _ in range(int(rng.integers(1, 3))):
            c = rng.uniform()
            hwid = rng.uniform(0.02, 0.05)
            d = np.minimum(np.abs(u - c), 1 - np.abs(u - c))
            g = np.exp(-(d / hwid) ** 2)
            w = w * (1 - g) + rng.uniform(*WIDTH_LO_RANGE) * g
    return np.clip(w, WIDTH_CLIP_LO, 3.0)


def write_csv(out_dir: str, name: str, cl: np.ndarray, w_r: np.ndarray, w_l: np.ndarray):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{name}_centerline.csv"), "w") as fp:
        fp.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")
        for (x, y), wr, wl in zip(cl, w_r, w_l):
            fp.write(f"{x:.6f}, {y:.6f}, {wr:.6f}, {wl:.6f}\n")


def post_load_stats(cl_gen: np.ndarray, w_r: np.ndarray, w_l: np.ndarray):
    """(중심선, 우/좌 폭) 을 학습 로더와 동일 전처리해 '로드 후' 형상 지표 반환.

    ★좌/우를 반드시 따로 받는다. 이전 버전은 합폭을 받아 0.5 씩 대칭으로 넣었는데,
      process_measured_centerline 의 첫 단계가 '중심선을 (w_l-w_r)/2 만큼 법선으로
      재정렬'하는 대칭화라서, 좌우 비대칭인 실측 맵은 대칭 입력과 곡률이 달라진다
      (실측 0811: 대칭 가정 |κ|max 2.42 / R+hw 0.86 vs 실제 CSV 2.78 / 0.80).
      즉 대칭으로 재던 값은 학습 로더가 보는 값이 아니었고, 그 상태로 밴드를 맞추면
      실측을 덮은 줄 알고 못 덮는다. 생성물은 좌우 대칭이라 영향이 없다.
    """
    center, hw_out = process_measured_centerline(cl_gen, w_r, w_l, P, scale=1.0)
    f = centerline_features(center)
    kap = np.abs(f["kappa"])
    bb = np.sort(_pca_align(center).ptp(0))[::-1]      # PCA bbox(회전 무관 실제 치수)
    # clear = min(R_curv + hw) = 바깥 라인 반경. 통과 가능성 판정과 커버리지 대조에 쓴다.
    clear = float((1.0 / np.maximum(kap, 1e-9) + hw_out).min())
    # counter = 장축 끝 노치의 회전량(도). V=48~107, U/ㄷ=145~200 으로 형상 계열을 가른다.
    # 5점 평활은 리샘플 잔결이 Ψ 를 미세하게 흔드는 것만 지운다(노치 회전량은 불변).
    nw = notch_window(_smooth_closed(_orient_ccw(center), 5))
    return dict(cl=center, hw=hw_out, lap=f["total_s"], long=bb[0], short=bb[1],
                kmax=float(kap.max()), kp99=float(np.quantile(kap, 0.99)),
                ksharp=float((kap > 1.0).mean()), clear=clear,
                counter=float(np.degrees(nw[2])) if nw else 0.0,
                hw_lo=float(hw_out.min()), hw_hi=float(hw_out.max()))


def main():
    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    # ---- (1) 연습 맵 전부를 '그대로' 포함 (실측 치수·폭) ----
    reals = 0
    real_stats = []
    for jp in find_lobby_maps():
        nm = "comp_" + lobby_name(jp)
        xy, w_r, w_l = load_lobby_map(jp)
        write_csv(args.out, nm, xy, w_r, w_l)
        s = post_load_stats(xy, w_r, w_l)        # 로드 후 지표(학습 로더와 동일)
        real_stats.append((nm, s))
        reals += 1

    if args.reals_only:
        print(f"[연습맵] {reals}종 기록 -> {args.out} (절차 생성본 미변경)")
        for nm, st in real_stats:
            print(f"    {nm:20s} 장{st['long']:.1f} 단{st['short']:.1f} 랩{st['lap']:.0f} "
                  f"|κ|max{st['kmax']:.2f} 급코너{st['ksharp']*100:.1f}% "
                  f"폭{2*st['hw_lo']:.2f}~{2*st['hw_hi']:.2f} "
                  f"카운터턴{st['counter']:3.0f}도({'U·ㄷ' if st['counter'] >= COUNTER_UD_MIN else 'V'})")
        return

    shapes = dict(load_seed_shapes())
    print(f"[시드] 형상 {len(shapes)}종: " + ", ".join(shapes))
    for b in BANDS:                       # 밴드가 요구하는 시드가 실제로 있는지 확인
        missing = [nm for nm in b[5] if nm not in shapes]
        if missing:
            raise SystemExit(f"[ERROR] 밴드 '{b[0]}' 의 시드 {missing} 를 찾을 수 없다. "
                             f"연습 맵 폴더를 확인할 것: {LOBBY_DIRS}")

    # ---- (2) 나머지를 밴드별 균등 생성 (로드 후 곡률로 검증) ----
    # --add_ud: 기존 comp### 를 세지 않고 보존한 채, 이름은 그 뒤 번호부터 이어 붙이고
    #           밴드는 U/ㄷ 만 쓴다. (기존 504종 전부 V 이므로 이렇게 해야 50/50 이 된다)
    kept = sorted(glob.glob(os.path.join(args.out, "comp[0-9]*")))
    if args.add_ud:
        pattern = BAND_PATTERN_UD
        idx0 = max([int(os.path.basename(p)[4:]) for p in kept], default=-1) + 1
        n_gen = args.n - reals - len(kept)
        if n_gen <= 0:
            raise SystemExit(f"[ERROR] --n {args.n} 이 이미 있는 {len(kept) + reals}종보다 크지 않다.")
        print(f"[증설] 기존 절차 생성본 {len(kept)}종 보존, comp{idx0:04d} 부터 U/ㄷ {n_gen}종 추가")
    else:
        pattern = BAND_PATTERN
        idx0, n_gen = 0, args.n - reals

    rng = np.random.default_rng(args.seed)
    made, tries, stats = 0, 0, []
    clear_min = P.wp_veh_r_min + PASS_MARGIN
    while made < n_gen and tries < n_gen * 120:
        tries += 1
        band = BANDS[pattern[made % len(pattern)]]
        pool, kind, notch = band[5], band[6], band[7]
        name_shape = pool[made % len(pool)]
        cl, long_m = perturb(shapes[name_shape], rng, band)
        if kind != NOTCH_V:
            cfg = dict(NOTCH_CFG); cfg.update(notch)
            cl = morph_notch(cl, kind, rng, cfg)
            if cl is None:
                continue
            cl = rescale_long(cl, long_m)      # 캡이 늘린 장변을 밴드 목표로 되돌린다
        w = make_width(len(cl), rng)
        s = post_load_stats(cl, 0.5 * w, 0.5 * w)   # 생성물은 좌우 대칭
        lo, hi = band[1]
        relax = tries > n_gen * 60
        if not (lo <= s["kmax"] <= hi) and not relax:
            continue
        # 통과 가능성: 바깥 라인 반경(R + hw) >= 차량 최소회전반경 + PASS_MARGIN.
        # ★이 검사는 relax 여부와 무관하게 항상 건다 — 주행 불가 코스는 못 쓴다.
        if s["clear"] < clear_min:
            continue
        if kind != NOTCH_V:
            # U/ㄷ 전용 게이트 — relax 여부와 무관하게 항상 건다.
            #  (a) 노치가 '로드 후에도' U/ㄷ 로 읽히는가: 캡을 155~205도로 만들어도 손가락이
            #      짧으면 로더 평활이 양옆 코너와 뭉개 V 급으로 내려앉는다. 그건 U/ㄷ 가 아니다.
            #  (b) 손가락 사이가 차가 못 지나갈 만큼 좁아지지 않았는가(로드 후 최소 폭).
            #  (c) 코스 규모가 기존 envelope 를 벗어나지 않았는가(랩/단변).
            #  (d) 중심선 자기교차 — 손가락이 다른 구간을 관통하면 벽 폴리라인이 꼬인다.
            if (s["counter"] < COUNTER_UD_MIN or 2 * s["hw_lo"] < WIDTH_MIN_UD
                    or s["lap"] > LAP_MAX_UD or s["short"] > SHORT_MAX
                    or _polyline_self_intersects(s["cl"][::2], min_gap=6)):
                continue
        name = f"comp{idx0 + made:03d}"
        write_csv(args.out, name, cl, 0.5 * w, 0.5 * w)     # 생성 CSV 는 pre-load(sharp)
        stats.append((s["lap"], s["long"], s["short"], s["hw_lo"], s["hw_hi"],
                      s["kmax"], s["ksharp"], s["kp99"], s["clear"], s["counter"],
                      band[0]))
        made += 1

    # ---- 요약 ----
    COLS = ("lap", "long", "short", "hw_lo", "hw_hi", "kmax", "ksharp", "kp99",
            "clear", "counter")
    s = np.array([r[:10] for r in stats], float)          # 이번 실행 생성분
    bands_of = [r[10] for r in stats]
    print(f"\n[생성] 이번 실행 변형본 {made}/{n_gen} (시도 {tries}) -> {args.out}")
    print(f"  * 지표는 모두 '로드 후'(학습이 실제 보는 값)")
    for nm, (lo, hi), *_ in BANDS:
        m = np.array([b == nm for b in bands_of], bool)
        cnt = int(m.sum())
        if not cnt:
            continue
        print(f"    {nm:7s} (|κ|max {lo:.2f}~{hi:.2f}, R {1/hi:.2f}~{1/lo:.2f}m): {cnt:4d}종  "
              f"|κ|max 중앙 {np.median(s[m,5]):.2f}  카운터턴 {s[m,9].min():3.0f}~{s[m,9].max():3.0f}도  "
              f"급코너비율 중앙 {np.median(s[m,6])*100:.1f}%")

    # ---- 최종 세트 전체(기존 보존분 포함) 재스캔 ----
    # 커버리지는 '학습이 실제 쓰게 될 최종 세트'로 판정해야 한다 (--add_ud 로 증설한
    # 경우 이번 실행 생성분만 보면 기존 V 분포가 빠져 오판한다).
    allp = sorted(glob.glob(os.path.join(args.out, "comp[0-9]*", "*_centerline.csv")))
    A, allnames = [], []
    for p in allp:
        rows = np.array([[float(v) for v in r[:4]] for r in csv.reader(open(p))
                         if r and not r[0].lstrip().startswith("#")])
        st = post_load_stats(rows[:, :2], rows[:, 2], rows[:, 3])
        A.append([st[k] for k in COLS])
        allnames.append(os.path.basename(os.path.dirname(p)))
    A = np.array(A, float)
    # 형상 계열은 '로드 후 카운터턴'으로 가른다 — 밴드 라벨이 아니라 학습이 보는 값 기준.
    is_ud = A[:, 9] >= COUNTER_UD_MIN
    print(f"\n[최종 세트] 총 {len(A) + reals}종 = 연습맵 {reals} + 절차 생성 {len(A)}")
    print(f"  형상 계열(로드 후 카운터턴 {COUNTER_UD_MIN:.0f}도 기준): "
          f"V자 {int((~is_ud).sum())}종 / U·ㄷ자 {int(is_ud.sum())}종  "
          f"(+연습맵 V {sum(1 for _, st in real_stats if st['counter'] < COUNTER_UD_MIN)} "
          f"/ U·ㄷ {sum(1 for _, st in real_stats if st['counter'] >= COUNTER_UD_MIN)})")
    for lab, m in (("V자  ", ~is_ud), ("U·ㄷ자", is_ud)):
        if not m.any():
            continue
        b = A[m]
        print(f"  {lab} {int(m.sum()):4d}종 | 랩 {b[:,0].min():4.1f}~{b[:,0].max():4.1f}  "
              f"장 {b[:,1].min():4.1f}~{b[:,1].max():4.1f}  단 {b[:,2].min():4.1f}~{b[:,2].max():4.1f}  "
              f"폭 {2*b[:,3].min():4.2f}~{2*b[:,4].max():4.2f}  "
              f"|κ|max {b[:,5].min():4.2f}~{b[:,5].max():4.2f} (중앙 {np.median(b[:,5]):.2f})  "
              f"R+hw {b[:,8].min():4.2f}~{b[:,8].max():4.2f}  "
              f"카운터턴 {b[:,9].min():3.0f}~{b[:,9].max():3.0f}도")
    km = A[:, 5]
    qs = np.quantile(km, [0.1, 0.25, 0.5, 0.75, 0.9])
    print("  |κ|max 분위: " + "  ".join(f"p{int(q*100)}={v:.2f}"
                                       for q, v in zip([0.1, 0.25, 0.5, 0.75, 0.9], qs)))
    print(f"  min(R+hw) : {A[:,8].min():4.2f} ~ {A[:,8].max():4.2f} m  "
          f"(통과 하한 {clear_min:.2f} = 최소회전반경 {P.wp_veh_r_min:.2f} + 여유 {PASS_MARGIN:.2f})")
    real_ks = np.median([st["ksharp"] for _, st in real_stats]) * 100
    print(f"  급코너(|κ|>1, 로드후) 랩 비율 중앙: {np.median(A[:,6])*100:.1f}%  "
          f"[연습맵 로드후 중앙 {real_ks:.1f}%]")

    # ---- 실측 연습 맵이 절차 생성 분포 '안'에 들어오는지 대조 ----
    # 이 생성기의 존재 이유가 '실측 형상을 학습 분포가 감싸게 하는 것'이므로,
    # 매 실행마다 지표별 포함 여부를 명시적으로 찍는다(X 가 뜨면 밴드를 조정할 것).
    # ★대조는 '같은 형상 계열' 안에서 한다 — V 맵을 U/ㄷ 분포와 비교하면 의미가 없다.
    print("  연습 맵(그대로 포함) — 괄호는 '같은 계열' 절차 생성 분포 포함 여부:")
    checks = [("장", 1), ("단", 2), ("랩", 0), ("|κ|max", 5), ("|κ|p99", 7),
              ("폭min", 3), ("R+hw", 8), ("카운터턴", 9)]
    for nm, st in real_stats:
        ud = st["counter"] >= COUNTER_UD_MIN
        c = A[is_ud] if ud else A[~is_ud]
        marks = []
        for lab, col in checks:
            v = 2 * st["hw_lo"] if col == 3 else st[COLS[col]]
            cc = 2 * c[:, col] if col == 3 else c[:, col]
            marks.append(f"{lab}{'O' if len(cc) and cc.min() <= v <= cc.max() else 'X'}")
        print(f"    {nm:20s}[{'U·ㄷ' if ud else 'V':>4s}] 장{st['long']:.1f} 단{st['short']:.1f} "
              f"랩{st['lap']:.0f} |κ|max{st['kmax']:.2f} p99 {st['kp99']:.2f} "
              f"R+hw {st['clear']:.2f} 폭{2*st['hw_lo']:.2f}~{2*st['hw_hi']:.2f} "
              f"카운터턴{st['counter']:3.0f}도  [{' '.join(marks)}]")


if __name__ == "__main__":
    main()
