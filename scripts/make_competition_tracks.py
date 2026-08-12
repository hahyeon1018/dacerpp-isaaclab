#!/usr/bin/env python3
"""대회 코스 변형본 512개 생성 (competition_tracks/<name>/<name>_centerline.csv).

■ 배경 (2026-07-30 실차 분석 rl_controller_analysis_20260730.txt 원인 A)
  구 생성기는 hangeong 중심선을 '규정 도면 치수'(22 x 8 m)로 리스케일했다. 그러나
  실제 설치·SLAM 된 코스(연습 맵 lobby_0723/0728/0730)는 16.4~16.9 x 5.6~5.9 m 로,
  구 생성물은 실제보다 1.4배 크고 곡률이 절반(|κ|max 중앙 0.90 vs 실제 1.85)이었다.
  = 실전 헤어핀(V자)이 학습에 사실상 없었고, 그래서 실차가 V자에서 벽에 박았다.

■ 이 생성기의 원칙
  (1) 시드 형상 = 실제 코스 9종(hangeong + 연습 맵 8개, 모두 같은 V/헤어핀 계열).
  (2) 크기 = 연습 맵 실측(장변 16.4~16.9 / 단변 5.6~5.9 m)을 넉넉히 브래킷
      (장변 13~20 / 단변 4.5~7.5 m). 곡률은 크기에 반비례하므로 실측 크기로
      되돌리는 것만으로 곡률이 실제 수준으로 회복된다.
  (3) V자 날카로움 스펙트럼 = '대형/완만'(|κ|max 1.18)부터 '초급함'(2.85, 실측
      최대 2.78을 넘어섬)까지 5밴드 배분. 급함+초급함(=62.5%, 315종)이 실전
      헤어핀을 직접 덮고, 그중 초급함 126종이 신규 연습 맵 0811/0812 대역이다.
  (4) ★검증은 '로드 후'(process_measured_centerline)로 한다. 학습 로더가 CSV 를
      읽을 때 11점 평활을 또 걸어 곡률이 줄므로, 생성 CSV 의 곡률이 아니라 '학습이
      실제 보는 곡률'로 밴드 판정해야 목표 분포가 학습에 그대로 들어간다.
      실측 맵은 좌/우 폭을 반드시 따로 넣는다(post_load_stats 주석 참조).
  (5) 연습 맵 전부는 competition 512 안에 '그대로'(실측 치수·폭) 포함한다.
  (6) ★생성 분포가 실측 연습 맵을 '감싸는지'를 매 실행 말미에 지표별로 찍는다.
      2026-08-12 기준 8종 x 7지표 전부 O (장/단/랩/|κ|max/|κ|p99/폭min/R+hw).

생성물은 f1tenth CSV 포맷(x_m, y_m, w_tr_right_m, w_tr_left_m)이라 기존 로더가
그대로 읽고, TrackField 가 축소 없이 실측 맵과 동급으로 취급한다.

■ 연습 맵 탐색 경로 (LOBBY_DIRS)
  generated/map_lobby/map_lobby_*/  (0723~0806) 와 generated/maps/lobby_*/  (0811~)
  두 곳을 모두 훑는다. 폴더명의 "map_" 접두사는 떼고 "comp_" 를 붙여 이름을 만든다
  (map_lobby_0806 -> comp_lobby_0806, lobby_0811 -> comp_lobby_0811).

실행:
  python scripts/make_competition_tracks.py --clean          # 512개(기본) 재생성
  python scripts/make_competition_tracks.py --n 60 --seed 7
  python scripts/make_competition_tracks.py --reals_only     # 연습 맵만 갱신/추가
       (절차 생성본 comp### 는 건드리지 않는다 — 새 연습 맵을 기존 세트에
        끼워 넣을 때 사용. 총 개수를 유지하려면 comp### 를 그만큼 지울 것.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dacerpp_lab.tracks import (TrackParams, _resample_equal_arclength,
                                _periodic_catmull_rom, centerline_features,
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
BANDS = [
    ("완만", (1.40, 1.68), (0.010, 0.045), (18.0, 21.0), (2.6, 3.2), SEEDS_BASE),
    ("중간", (1.68, 1.85), (0.010, 0.050), (16.0, 18.2), (2.7, 3.2), SEEDS_BASE),   # 실측(1.77~1.89) 대역
    ("급함", (1.85, 1.98), (0.015, 0.060), (14.0, 16.4), (2.8, 3.3), SEEDS_BASE),   # 실측 practice~구 통과 한계
    ("대형", (1.18, 1.52), (0.008, 0.035), (21.0, 24.0), (2.15, 2.75), SEEDS_BASE),  # 규정 22x8~여유 24x11
    # 초급함: 실측 0811(2.78)·0812(2.43) 를 안쪽에 두도록 1.98~2.85. bbox 는 실측과
    # 같은 15~19m 로 두어 랩 길이(≈40~48m)와 코스 규모는 실전 그대로 유지하고,
    # 날카로움만 시드 apex 에서 얻는다(= 실측 맵이 실제로 그런 형상이다).
    ("초급함", (1.98, 2.85), (0.015, 0.060), (15.0, 19.0), (2.5, 3.3), SEEDS_ULTRA),
]
# 배분: 실측 8종의 구성비(초급함 2/8=25%)를 그대로 반영. 급함+초급함 = 62.5% 로
# sharp 계열 지배를 유지하고, 완만/중간/대형은 각 12.5% tail.
# 504 = 63 x 8 이라 패턴이 정확히 나눠떨어진다(완만63 중간63 급함189 대형63 초급함126).
BAND_PATTERN = (0, 1, 2, 2, 2, 3, 4, 4)
SHORT_MAX = 11.0                 # 단변 상한(m)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=512, help="총 트랙 수(연습 맵 포함)")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--out", type=str, default=OUT_DIR)
parser.add_argument("--clean", action="store_true", help="기존 생성물 삭제 후 재생성")
parser.add_argument("--reals_only", action="store_true",
                    help="연습 맵(comp_lobby_*)만 쓰고 종료 — 절차 생성본은 그대로 둔다")
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


def perturb(base: np.ndarray, rng, band) -> np.ndarray:
    """실측 중심선을 '원해상도로 직접' 변형 — 저주파 법선 지터 + bbox 스케일 + 미러.
    구 방식의 제어점 다운샘플/Catmull-Rom(=apex 뭉갬)을 제거해 급한 V 곡률을 보존한다."""
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
    return _resample_full(cl)


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
    return dict(cl=center, hw=hw_out, lap=f["total_s"], long=bb[0], short=bb[1],
                kmax=float(kap.max()), kp99=float(np.quantile(kap, 0.99)),
                ksharp=float((kap > 1.0).mean()), clear=clear,
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
                  f"폭{2*st['hw_lo']:.2f}~{2*st['hw_hi']:.2f}")
        return

    shapes = dict(load_seed_shapes())
    print(f"[시드] 형상 {len(shapes)}종: " + ", ".join(shapes))
    for b in BANDS:                       # 밴드가 요구하는 시드가 실제로 있는지 확인
        missing = [nm for nm in b[5] if nm not in shapes]
        if missing:
            raise SystemExit(f"[ERROR] 밴드 '{b[0]}' 의 시드 {missing} 를 찾을 수 없다. "
                             f"연습 맵 폴더를 확인할 것: {LOBBY_DIRS}")

    # ---- (2) 나머지를 밴드별 균등 생성 (로드 후 곡률로 검증) ----
    n_gen = args.n - reals
    rng = np.random.default_rng(args.seed)
    made, tries, stats = 0, 0, []
    clear_min = P.wp_veh_r_min + PASS_MARGIN
    while made < n_gen and tries < n_gen * 120:
        tries += 1
        band = BANDS[BAND_PATTERN[made % len(BAND_PATTERN)]]
        pool = band[5]                     # 밴드 전용 시드 풀 (완만~대형 7종 / 초급함 2종)
        name_shape = pool[made % len(pool)]
        cl = perturb(shapes[name_shape], rng, band)
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
        name = f"comp{made:03d}"
        write_csv(args.out, name, cl, 0.5 * w, 0.5 * w)     # 생성 CSV 는 pre-load(sharp)
        stats.append((s["lap"], s["long"], s["short"], s["hw_lo"], s["hw_hi"],
                      s["kmax"], s["ksharp"], s["kp99"], s["clear"], band[0]))
        made += 1

    # ---- 요약 ----
    s = np.array([r[:9] for r in stats], float)
    bands_of = [r[9] for r in stats]
    print(f"\n[생성] 총 {made + reals}종 = 연습맵 {reals} + 변형본 {made}/{n_gen} "
          f"(시도 {tries}) -> {args.out}")
    print(f"  * 지표는 모두 '로드 후'(학습이 실제 보는 값)")
    print(f"  랩 길이 : {s[:,0].min():5.1f} ~ {s[:,0].max():5.1f} m (중앙 {np.median(s[:,0]):.1f})")
    print(f"  bbox    : 장변 {s[:,1].min():4.1f}~{s[:,1].max():4.1f}  "
          f"단변 {s[:,2].min():4.1f}~{s[:,2].max():4.1f} m   "
          f"[연습맵 장{16.4:.0f}~{16.9:.1f} 단{5.6:.1f}~{5.9:.1f}]")
    print(f"  폭      : {2*s[:,3].min():4.2f} ~ {2*s[:,4].max():4.2f} m")
    km = s[:, 5]
    print(f"  |κ|max  : {km.min():4.2f} ~ {km.max():4.2f}  (최소 R {1/km.max():.2f}~{1/km.min():.2f} m)")
    qs = np.quantile(km, [0.1, 0.25, 0.5, 0.75, 0.9])
    print("  |κ|max 분위: " + "  ".join(f"p{int(q*100)}={v:.2f}"
                                       for q, v in zip([0.1, 0.25, 0.5, 0.75, 0.9], qs)))
    print(f"  |κ|p99  : {s[:,7].min():4.2f} ~ {s[:,7].max():4.2f}")
    print(f"  min(R+hw): {s[:,8].min():4.2f} ~ {s[:,8].max():4.2f} m  "
          f"(통과 하한 {clear_min:.2f} = 최소회전반경 {P.wp_veh_r_min:.2f} + 여유 {PASS_MARGIN:.2f})")
    real_ks = np.median([st["ksharp"] for _, st in real_stats]) * 100
    print(f"  급코너(|κ|>1, 로드후) 랩 비율 중앙: {np.median(s[:,6])*100:.1f}%  "
          f"[연습맵 로드후 중앙 {real_ks:.1f}%]")
    for nm, (lo, hi), *_ in BANDS:
        m = np.array([bands_of[i] == nm for i in range(len(bands_of))])
        cnt = int(m.sum())
        ks = s[m, 6] if cnt else np.array([0.0])
        print(f"    {nm} (|κ|max {lo:.2f}~{hi:.2f}, R {1/hi:.2f}~{1/lo:.2f}m): {cnt}종  "
              f"급코너비율 중앙 {np.median(ks)*100:.1f}%")

    # ---- 실측 연습 맵이 절차 생성 분포 '안'에 들어오는지 대조 ----
    # 이 생성기의 존재 이유가 '실측 형상을 학습 분포가 감싸게 하는 것'이므로,
    # 매 실행마다 지표별 포함 여부를 명시적으로 찍는다(X 가 뜨면 밴드를 조정할 것).
    print("  연습 맵(그대로 포함) — 괄호는 절차 생성 분포 포함 여부:")
    checks = [("장", "long", 1), ("단", "short", 2), ("랩", "lap", 0),
              ("|κ|max", "kmax", 5), ("|κ|p99", "kp99", 7),
              ("폭min", None, 3), ("R+hw", "clear", 8)]
    for nm, st in real_stats:
        marks = []
        for lab, key, col in checks:
            v = 2 * st["hw_lo"] if key is None else st[key]
            c = 2 * s[:, col] if key is None else s[:, col]
            marks.append(f"{lab}{'O' if c.min() <= v <= c.max() else 'X'}")
        print(f"    {nm:20s} 장{st['long']:.1f} 단{st['short']:.1f} 랩{st['lap']:.0f} "
              f"|κ|max{st['kmax']:.2f} p99 {st['kp99']:.2f} R+hw {st['clear']:.2f} "
              f"폭{2*st['hw_lo']:.2f}~{2*st['hw_hi']:.2f}  [{' '.join(marks)}]")


if __name__ == "__main__":
    main()
