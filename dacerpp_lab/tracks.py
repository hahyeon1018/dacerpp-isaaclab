"""중심선(스플라인) 기반 절차적 레이스 서킷 생성.

- make_centerline : layout_id/difficulty 로 닫힌 중심선(폐곡선)을 생성.
    기저 반경에 layout 별 하모닉(사인 합)을 더해 다양한 코너 구성을 만든다.
    (자기교차가 없도록 진폭을 보수적으로 제한)
- build_track_mesh : 중심선 -> 노면(road) + 좌/우 벽(walls) trimesh 생성.

좌표계: 트랙은 캐논컬 로컬 프레임(원점 중심)에서 생성된다.
실제 배치(2048개 그리드)는 track_field.py 가 원점을 평행이동해 처리한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import trimesh


# 25종 서킷의 하모닉 프로파일: (harmonic_index, amplitude_ratio, phase)
# amplitude 는 difficulty 로 스케일된다. base_radius 대비 비율.
# 홀수/저차 하모닉(k=1,3,5..)과 비영 위상은 비대칭 서킷을 만든다.
# 검증/튜닝(2026-07-16, ds=0.15 리샘플 파이프라인 기준): 각 레이아웃은
#   d=0.6 에서 min(R_curv - hw_local) >= 0.35m, d=1.0 에서 >= 0.05m,
#   코리도(벽) 비접힘을 만족하는 최대 진폭으로 이분탐색 스케일.
# 0..8  : 기존 9종 (2,3,6,7 은 재색인된 폭 프로파일 기준으로 진폭 재튜닝)
# 9..15 : 복원 7종 (구 6,8,10,11,12,13,15 — 현 파이프라인 검증 통과)
# 16..24: 신규 9종 — 4~5 하모닉 + 전 항 비영 위상(고복잡·비대칭),
#         유효성 한계까지 진폭 확대. f1tenth 고정폭(1.1m) 트랙 대비
#         가변 폭(0.5~2.5m) 노출을 늘리는 목적 (실전 덕트 대회장 분포).
_LAYOUTS = {
    0: [(2, 0.057, 0.0), (4, 0.017, 1.1), (7, 0.023, 2.3), (5, 0.074, 1.7)],  # oval + 굴곡
    1: [(3, 0.14, 0.3), (6, 0.04, 1.7)],                       # 3코너 + 미세 시케인
    2: [(2, 0.058, 0.0), (4, 0.058, 0.8), (6, 0.017, 2.1), (5, 0.047, 1.7)],  # s_curve 복합(재튜닝)
    3: [(1, 0.054, 0.0), (5, 0.075, 0.5), (7, 0.01, 1.2), (6, 0.025, 2.1)],   # 헤어핀 비대칭(재튜닝)
    4: [(4, 0.16, 0.0), (8, 0.012, 0.6)],                      # 4코너 사각 + 미세 변조
    5: [(3, 0.099, 1.0), (6, 0.066, 0.2)],                     # 테크니컬 시케인
    6: [(5, 0.056, 0.0), (2, 0.046, 1.4), (7, 0.023, 0.7)],    # 복합 곡률(재튜닝)
    7: [(2, 0.063, 0.5), (4, 0.102, 1.8), (6, 0.039, 0.3)],    # 더블 헤어핀(재튜닝)
    8: [(2, 0.087, 1.2), (6, 0.048, 2.9), (11, 0.006, 0.5)],   # 스타디움 + 에세스
    9: [(2, 0.18, 0.4), (5, 0.04, 2.6)],                       # 긴 직선 + 두 고속코너 (복원)
    10: [(1, 0.06, 0.9), (2, 0.12, 0.0), (7, 0.04, 2.3)],      # 비대칭 oval + 킹크 (복원)
    11: [(1, 0.07, 2.0), (3, 0.1, 0.9), (8, 0.03, 1.5)],       # 비대칭 3코너 + 테크니컬 (복원)
    12: [(5, 0.073, 0.6), (2, 0.036, 1.9), (10, 0.006, 0.1)],  # 5코너 + 고주파 잔결 (복원)
    13: [(1, 0.05, 1.3), (3, 0.13, 2.4), (9, 0.02, 0.8)],      # 비대칭 + 시케인 조합 (복원)
    14: [(2, 0.14, 2.8), (3, 0.07, 0.4), (5, 0.05, 1.6)],      # 와이드 스위퍼 비대칭 (복원)
    15: [(4, 0.067, 0.3), (7, 0.029, 1.1), (10, 0.006, 2.2), (5, 0.029, 1.7), (3, 0.124, 0.8)],  # 최고난도 테크니컬 (복원, c=0.951 재튜닝: margin@0.6 0.118->0.368)
    16: [(1, 0.037, 0.7), (3, 0.068, 2.1), (5, 0.043, 0.9), (9, 0.012, 1.5)],   # 비대칭 3-5코너 복합+잔결
    17: [(2, 0.063, 1.8), (3, 0.05, 0.3), (7, 0.032, 2.7), (11, 0.008, 0.8)],   # 더블스위퍼+킹크 잔결
    18: [(1, 0.031, 2.4), (4, 0.074, 1.1), (6, 0.037, 0.5), (9, 0.011, 2.0)],   # 비대칭 4코너 테크니컬
    19: [(3, 0.047, 1.4), (5, 0.047, 0.2), (8, 0.018, 1.9), (12, 0.004, 0.4)],  # 촘촘 시케인 연속
    20: [(1, 0.036, 0.2), (2, 0.047, 2.6), (5, 0.041, 1.2), (7, 0.021, 0.6), (10, 0.006, 1.7)],  # 5하모닉 복합 비대칭
    21: [(2, 0.073, 0.9), (6, 0.04, 2.2), (9, 0.014, 0.1), (3, 0.2, 1.17)],                     # 와이드+에세스 고밀도
    22: [(1, 0.018, 1.6), (4, 0.046, 2.9), (7, 0.028, 1.3), (13, 0.003, 2.5)],  # 비대칭+고주파 미세굴곡
    23: [(3, 0.078, 0.6), (4, 0.045, 1.8), (8, 0.026, 0.9), (11, 0.009, 2.2)],  # 3-4중첩 테크니컬
    24: [(2, 0.025, 1.3), (5, 0.031, 2.0), (6, 0.016, 0.7), (9, 0.009, 1.1), (12, 0.003, 2.8)],  # 최고복잡 5하모닉
}
NUM_LAYOUTS = len(_LAYOUTS)


@dataclass
class TrackParams:
    base_radius: float = 9.33       # 트랙 평균 반경(m). 기존 14.0 의 2/3 로 축소(폭은 유지)
    track_width: float = 3.0        # 공칭 노면 폭(m) — 가변 폭 프로파일의 참고값
    width_min: float = 1.0          # 최소 노면 폭(m) — 실제 대회장 덕트 간격 최소
    width_max: float = 5.0          # 최대 노면 폭(m) — 실제 대회장 덕트 간격 최대
    wall_height: float = 0.3        # 충돌 벽 높이(m)
    n_points: int = 400             # 중심선 리샘플 점 수 (ds<=0 일 때만 사용)
    # ★ 모든 트랙 공통 등간격 호길이(m). 관측의 전방 곡률/폭 lookahead 와
    # 레이캐스트 윈도가 '중심선 인덱스' 단위라, 트랙마다 점 간격이 다르면
    # 같은 인덱스 오프셋이 다른 거리를 의미하게 된다(관측 기준 불일치).
    # 절차 트랙과 f1tenth 트랙(원본 0.33~0.46m 간격) 전부 이 값으로 리샘플해
    # '인덱스당 거리'를 통일한다. (+5/15/30/60/90 idx = 0.75/2.25/4.5/9/13.5m)
    ds: float = 0.15
    # ---- 웨이포인트 기반 생성기 (2026-07-21, 하모닉 방식 대체) ----
    # 하모닉(r=R+Σamp·sin) 방식은 극좌표 반경 변조라 '원점 기준 star-shaped' 위상만
    # 만들어져 S자/헤어핀/V자/급코너가 원리적으로 불가능했다. 실측 결과 학습 48종
    # 전부에서 '좁고(반폭<0.7m) 급한(R<2m)' 구간이 0% 였는데 실전 대회장 맵에는
    # 2.6%(hangeong 10.3%) 존재 -> 새 맵 실차 주행 실패의 직접 원인.
    # 랜덤점 -> convex hull -> 변 중점 랜덤 변위(오목부/S자) -> 스플라인 -> 목표
    # 랩길이로 스케일 정규화 방식으로 교체. 아래 값은 실전 4맵 통계에 맞춘 것:
    #   랩길이 31~43m, extent 6.4~8.1m, |κ|max 0.92~1.85, 반폭 0.47~3.13m
    wp_len_range: tuple = (20.0, 55.0)     # 목표 랩 길이(m) — 스케일 정규화 대상
    wp_kappa_max: float = 1.85             # |κ| 상한(R 0.54m=실전 hangeong 수준) — 초과 시 재생성
    wp_veh_r_min: float = 0.76             # 차량 기하 최소회전반경(축거 0.33/tan 0.42)
    wp_outer_margin: float = 0.10          # 바깥 라인 통과 여유(m)
    # 폭: 하단을 넓게(0.4~1.6) 뽑아 '좁은 게이트가 있는 코스'와 '전 구간이 넓은
    # 코스'(실전 hall 최소반폭 1.19 / teras 1.55)를 모두 생성한다.
    wp_hw_lo: tuple = (0.40, 1.85)         # 반폭 하단 랜덤 범위(m)
    # 상단 = 하단 + 이 범위(m). 하한을 0 근처로 둬서 '폭이 거의 일정한' 코스
    # (실전 hall: 1.83~1.99m)도 생성되게 한다.
    wp_hw_gap: tuple = (0.05, 1.80)
    wp_gate_p: float = 0.6                 # 게이트(병목)를 넣을 확률
    wp_gates: tuple = (1, 4)               # 넣는 경우의 개수 범위
    wp_gate_hw: tuple = (0.40, 0.65)       # 게이트 반폭(m)
    # ---- 곡률 기반 폭 제한 (모든 트랙 공통) ----
    # 반폭 hw 가 국소 곡률반경 R 보다 크면 '안쪽 벽'(중심선 - hw·n)이 곡률 중심을
    # 넘어 뒤집혀 자기교차한다 -> build_walls 로 만든 벽 폴리라인이 꼬여서
    # 레이캐스트 스캔이 엉뚱한 거리를 반환(관측 오염). 급코너에서는 실제로도
    # 코리도가 좁을 수밖에 없으므로 hw <= max(alpha*R, floor) 로 제한한다.
    # (floor 는 차량 통과 최소치 — 그 미만으로 좁히면 주행 자체가 불가)
    # alpha 0.9 -> 0.7: 안쪽 벽 반경(=R-hw)이 0.1R 밖에 안 되면 ds=0.15m 폴리라인으로
    # 그릴 때 세그먼트가 서로 교차한다(코너의 '작은 삼각형'). 0.3R 이상 남긴다.
    hw_curv_alpha: float = 0.7
    hw_curv_floor: float = 0.35
    # ---- 반폭 프로파일 평활 (벽 첨점 제거) ----
    # clip_width_by_curvature 는 곡률(1/R)에 맞춰 hw 를 '점 단위'로 깎는다. 코너
    # apex 는 R 이 국소 최소라 hw 도 한 점만 급강하(노치)하는데, 그 지점에서 법선이
    # 빠르게 회전하므로 안쪽 벽(cl - hw·n)이 이웃보다 안으로 튀어나온 V자 첨점이 된다
    # (실측: 안쪽 벽 국소 첨점 p50 10cm / max 15cm). 노치는 1~3점 폭이고 의도한
    # 게이트(병목)는 15~50점 폭이라, 아래 창(≈±0.6m)의 주기 박스카로 노치만 지우고
    # 게이트는 보존한다. 부수 효과로 apex 폭이 넓어져 주행성도 개선(첨점 max 4.5cm,
    # 최소 반폭 0.47->0.63m). 평활이 hw>R 접힘을 재유발하는 드문 경우(≈2%)는
    # 뒤따르는 shrink_width_until_clean 이 처리한다.
    hw_smooth_win: int = 9


def _periodic_catmull_rom(ctrl: np.ndarray, samples_per_seg: int = 20) -> np.ndarray:
    """닫힌 제어점열을 Catmull-Rom 으로 부드럽게 보간(주기적)."""
    n = len(ctrl)
    out = []
    for i in range(n):
        p0 = ctrl[(i - 1) % n]
        p1 = ctrl[i]
        p2 = ctrl[(i + 1) % n]
        p3 = ctrl[(i + 2) % n]
        t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
        t2, t3 = t * t, t * t * t
        seg = (0.5 * ((2 * p1)
                      + (-p0 + p2) * t
                      + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                      + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
        out.append(seg)
    return np.concatenate(out, axis=0)


def _resample_equal_arclength(xy: np.ndarray, n: int, attrs: np.ndarray | None = None):
    """폐곡선을 등호장 간격 n개 점으로 리샘플.

    attrs: (m, k) 정점별 부가 프로파일(예: half-width)도 같은 호길이 매개변수로
    함께 선형보간. 주어지면 (out_xy, out_attrs) 를 반환."""
    seg = np.roll(xy, -1, axis=0) - xy
    seglen = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = s[-1]
    target = np.linspace(0.0, total, n, endpoint=False)
    closed = np.vstack([xy, xy[0]])
    closed_at = np.vstack([attrs, attrs[0]]) if attrs is not None else None
    out = np.empty((n, 2))
    out_at = np.empty((n, attrs.shape[1])) if attrs is not None else None
    j = 0
    for k, ts in enumerate(target):
        while j < len(s) - 1 and s[j + 1] < ts:
            j += 1
        denom = max(s[j + 1] - s[j], 1e-9)
        a = (ts - s[j]) / denom
        out[k] = closed[j] * (1 - a) + closed[j + 1] * a
        if closed_at is not None:
            out_at[k] = closed_at[j] * (1 - a) + closed_at[j + 1] * a
    return (out, out_at) if attrs is not None else out


def make_centerline(layout_id: int, difficulty: float = 0.5,
                    params: TrackParams = TrackParams()) -> np.ndarray:
    """닫힌 중심선 (n_points, 2) 반환 (로컬 원점 중심)."""
    layout_id = int(layout_id) % NUM_LAYOUTS
    n_ctrl = 36
    theta = np.linspace(0.0, 2 * np.pi, n_ctrl, endpoint=False)
    r = np.full(n_ctrl, params.base_radius, dtype=np.float64)
    for (k, amp, phase) in _LAYOUTS[layout_id]:
        r += params.base_radius * amp * (0.4 + 0.6 * difficulty) * np.sin(k * theta + phase)
    ctrl = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    smooth = _periodic_catmull_rom(ctrl, samples_per_seg=24)
    if params.ds > 0:   # 공통 등간격(트랙 간 인덱스 거리 통일 — TrackParams.ds 주석)
        total = np.linalg.norm(np.roll(smooth, -1, axis=0) - smooth, axis=1).sum()
        n = max(64, int(round(total / params.ds)))
    else:
        n = params.n_points
    cl = _resample_equal_arclength(smooth, n)
    cl -= cl.mean(axis=0, keepdims=True)  # 무게중심 원점화
    return cl.astype(np.float64)


def make_width_profile(layout_id: int, difficulty: float = 0.5,
                       params: TrackParams = TrackParams(), n: int | None = None) -> np.ndarray:
    """중심선 각 정점의 half-width 프로파일 (n,) 반환.

    저주파 사인 합으로 폭이 [width_min, width_max] 사이에서 부드럽게 변한다
    (실제 F1TENTH 대회장: 덕트 간격 1~5m). 주기적(폐곡선 연속) 보장.
    layout_id 로 시드 고정 -> 같은 트랙은 항상 같은 프로파일.
    """
    if n is None:
        n = params.n_points
    hw_min = 0.5 * params.width_min
    hw_max = 0.5 * params.width_max
    mid = 0.5 * (hw_min + hw_max)
    amp = 0.5 * (hw_max - hw_min)
    rng = np.random.default_rng(1000 + int(layout_id))
    u = np.arange(n) / n
    prof = (0.6 * np.sin(2 * np.pi * (1 * u + rng.uniform()))
            + 0.3 * np.sin(2 * np.pi * (2 * u + rng.uniform()))
            + 0.1 * np.sin(2 * np.pi * (3 * u + rng.uniform())))   # [-1, 1]
    # 1.25x 과증폭 후 클립: 최소/최대 폭 구간이 실제로 등장(포화 구간 = 일정 폭 병목/광폭부)
    hw = mid + amp * 1.25 * prof
    return np.clip(hw, hw_min, hw_max).astype(np.float64)


# ---------------------------------------------------------------------------
# 웨이포인트 기반 절차 생성 (TrackParams.wp_* 주석 참조)
# ---------------------------------------------------------------------------
def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """CCW convex hull (Andrew monotone chain) — scipy 의존 없음."""
    p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(p) < 3:
        return p
    def half(ps):
        out = []
        for q in ps:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0]-a[0]) * (q[1]-a[1]) - (b[1]-a[1]) * (q[0]-a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(q)
        return out
    return np.asarray(half(p)[:-1] + half(p[::-1])[:-1])


def _displace_midpoints(poly: np.ndarray, rng, scale: float,
                        inward_bias: float = 0.35) -> np.ndarray:
    """각 변의 중점을 법선 방향으로 랜덤 변위 -> 오목부/S자 생성."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        out.append(a)
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            continue
        nrm = np.array([-seg[1], seg[0]]) / L        # CCW 폐곡선의 바깥 법선
        # 안쪽(-)으로 치우친 변위 -> 오목부(헤어핀/S자)가 자주 생김
        out.append((a + b) * 0.5 + nrm * (rng.normal(-inward_bias, 1.0) * scale * L))
    return np.asarray(out)


def _polyline_self_intersects(poly: np.ndarray, min_gap: int = 10) -> bool:
    """폐 폴리라인의 비인접 세그먼트가 서로 교차하는가 (벡터화)."""
    n = len(poly)
    if n < 8:
        return False
    a = poly
    d = np.roll(poly, -1, axis=0) - poly
    A, D = a[:, None, :], d[:, None, :]
    C, E = a[None, :, :], d[None, :, :]
    denom = D[..., 0] * E[..., 1] - D[..., 1] * E[..., 0]
    ok = np.abs(denom) > 1e-12
    dd = np.where(ok, denom, 1.0)
    AC = C - A
    t = (AC[..., 0] * E[..., 1] - AC[..., 1] * E[..., 0]) / dd
    u = (AC[..., 0] * D[..., 1] - AC[..., 1] * D[..., 0]) / dd
    hit = ok & (t > 1e-6) & (t < 1 - 1e-6) & (u > 1e-6) & (u < 1 - 1e-6)
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    gap = np.minimum(gap, n - gap)
    return bool((hit & (gap >= min_gap)).any())


def _corridor_self_interference(cl: np.ndarray, hw: np.ndarray, ds: float,
                                min_gap: int = 2) -> bool:
    """코리도(좌/우 벽)가 스스로 꼬이는가.

    중심선 ± hw·n 로 만든 벽 폴리라인의 세그먼트 교차를 직접 검사한다.
    (구 버전은 '중심선 점 사이 거리 < 폭합' 휴리스틱이었는데, 임계가 폭에
    비례해 커져서 넓은 트랙의 U자 구간 겹침을 놓쳤다 — 벽이 꼬이면
    build_walls 폴리라인이 뒤집혀 레이캐스트 스캔이 오염된다.)
    """
    step = max(1, int(round(0.3 / max(ds, 1e-6))))     # 0.3m 데시메이트
    p, w = cl[::step], hw[::step]
    psi = centerline_features(p)["psi"]
    nrm = np.stack([-np.sin(psi), np.cos(psi)], axis=1)
    for side in (+1.0, -1.0):
        # min_gap 2 = 0.6m: 코너에서 생기는 '작은 삼각형' 접힘까지 잡는다.
        # (hw <= R 은 무한소 조건일 뿐이라, 곡률이 급변하는 짧은 코너에서는
        #  그 조건을 만족해도 평행곡선이 교차한다 — 실측 227종 중 138종)
        if _polyline_self_intersects(p + side * nrm * w[:, None], min_gap=min_gap):
            return True
    return False


def shrink_width_until_clean(cl: np.ndarray, hw: np.ndarray,
                             params: TrackParams = TrackParams(),
                             max_iter: int = 8, factor: float = 0.88):
    """벽 교차가 사라질 때까지 반폭을 점진 축소. 실패 시 None.

    곡률/근접 클립으로도 남는 평행곡선 교차(급코너의 삼각형 접힘)를 확실히
    제거한다. 폭이 다소 좁아지지만, 꼬인 벽은 레이캐스트를 오염시키므로
    그대로 두는 것보다 낫다.
    """
    ds = params.ds if params.ds > 0 else 0.15
    for _ in range(max_iter):
        if not _corridor_self_interference(cl, hw, ds):
            return hw
        hw = np.maximum(hw * factor, 0.30)
    return None if _corridor_self_interference(cl, hw, ds) else hw


def smooth_half_width(hw: np.ndarray, params: TrackParams = TrackParams()) -> np.ndarray:
    """반폭 프로파일을 주기 박스카로 평활 — 코너 apex 노치(=벽 첨점) 제거.

    TrackParams.hw_smooth_win 주석 참조. 창은 노치(1~3점)만 지우고 게이트(15~50점)는
    보존하도록 잡혀 있다. 폐곡선이므로 양끝을 감싸 순환 평활한다.
    """
    win = int(params.hw_smooth_win)
    if win < 3 or len(hw) <= win:
        return hw
    h = win // 2
    k = np.ones(win) / win
    pad = np.concatenate([hw[-h:], hw, hw[:h]])
    return np.convolve(pad, k, mode="valid")[:len(hw)].astype(np.float64)


def clip_width_by_curvature(cl: np.ndarray, hw: np.ndarray,
                            params: TrackParams = TrackParams()) -> np.ndarray:
    """안쪽 벽이 뒤집히지 않도록 반폭을 국소 곡률로 제한.

    hw > R_curv 이면 중심선 - hw·n 이 곡률 중심을 넘어 자기교차한다
    (벽 폴리라인이 꼬여 레이캐스트가 오염). TrackParams.hw_curv_* 주석 참조.
    """
    R = 1.0 / np.maximum(np.abs(centerline_features(cl)["kappa"]), 1e-6)
    return np.minimum(hw, np.maximum(params.hw_curv_alpha * R, params.hw_curv_floor))


def clip_width_by_proximity(cl: np.ndarray, hw: np.ndarray,
                            params: TrackParams = TrackParams(),
                            min_arc: float = 2.0, safety: float = 0.95) -> np.ndarray:
    """서로 다른 구간의 코리도가 겹치지 않도록 반폭을 이웃 구간 거리로 제한.

    hw_i <= (호길이 min_arc 이상 떨어진 중심선까지의 최소 거리) / 2.
    실전 SLAM 맵(hall/teras)은 d_left/d_right 가 '넓은 홀 공간' 전체를 재므로
    폭이 트랙 간격보다 커져 벽이 서로 통과해 버린다 — 중심선±hw 대칭 코리도
    모델로는 그 형상을 표현할 수 없으므로 실제 통로 폭으로 좁힌다.
    """
    ds = params.ds if params.ds > 0 else 0.15
    n_full = len(cl)
    step = max(1, int(round(0.3 / ds)))
    p = cl[::step]
    n = len(p)
    idx = np.arange(n)
    di = np.abs(idx[:, None] - idx[None, :])
    arc = np.minimum(di, n - di) * step * ds
    dist = np.linalg.norm(p[:, None] - p[None, :], axis=2)
    dmin = np.where(arc > min_arc, dist, np.inf).min(axis=1) * 0.5 * safety
    if not np.isfinite(dmin).all():                 # 트랙이 매우 짧은 경우
        dmin = np.where(np.isfinite(dmin), dmin, hw.max())
    full = np.interp(np.arange(n_full), np.arange(0, n_full, step)[:n], dmin,
                     period=n_full)
    return np.minimum(hw, full)


def make_waypoint_width(n: int, rng, params: TrackParams) -> np.ndarray:
    """실전형 반폭 프로파일: 완만한 저주파 + 급격한 게이트(병목).

    기존 하모닉판(make_width_profile)은 저주파 사인 합뿐이라 폭이 완만하게만
    변했다. 실전 대회장은 넓은 구간에서 좁은 게이트로 급전이하므로,
    가우시안 병목을 1~3개 얹어 그 패턴을 만든다.
    """
    u = np.arange(n) / n
    base = (0.6 * np.sin(2 * np.pi * (1 * u + rng.uniform()))
            + 0.3 * np.sin(2 * np.pi * (2 * u + rng.uniform()))
            + 0.1 * np.sin(2 * np.pi * (3 * u + rng.uniform())))
    lo = rng.uniform(*params.wp_hw_lo)
    hi = lo + rng.uniform(*params.wp_hw_gap)
    hw = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.clip(base * 1.25, -1, 1)
    if rng.uniform() < params.wp_gate_p:   # 일부는 게이트 없이(전 구간 넓은 코스)
        for _ in range(int(rng.integers(*params.wp_gates))):
            c = rng.uniform()
            halfw = rng.uniform(0.02, 0.06)                # 랩의 2~6% 구간
            d = np.minimum(np.abs(u - c), 1 - np.abs(u - c))
            gate = np.exp(-(d / halfw) ** 2)
            hw = hw * (1 - gate) + rng.uniform(*params.wp_gate_hw) * gate
    return np.clip(hw, 0.38, 3.2).astype(np.float64)


def make_waypoint_track(seed: int, params: TrackParams = TrackParams(),
                        max_retry: int = 40):
    """웨이포인트 기반 트랙 (cl, hw) 생성. 실패 시 None.

    통과 가능성 기준은 '코너 바깥 라인 반경(R_curv + hw) >= 차량 최소회전반경'.
    (중심선 R < hw 는 넓은 코너에서 정상 — 실전 4맵도 전부 그렇다)
    """
    ds = params.ds if params.ds > 0 else 0.15
    rng = np.random.default_rng(seed)
    for _ in range(max_retry):
        pts = rng.uniform(-1, 1, (int(rng.integers(6, 11)), 2))   # 형상만(단위 스케일)
        poly = _convex_hull(pts)
        if len(poly) < 5:
            continue
        # 변위 1~2회만 (과하면 전 구간이 꼬불꼬불 — 실전은 '직선 + 급코너 몇 개').
        # 하한을 낮게 둬서 '넓고 완만한' 코스(실전 hall/teras 유형)도 나오게 한다
        # — 굴곡이 세면 곡률 클립에 걸려 전 구간이 좁아지기 때문.
        scale = rng.uniform(0.05, 0.32)
        for _ in range(int(rng.integers(1, 3))):
            poly = _displace_midpoints(poly, rng, scale)
            scale *= rng.uniform(0.35, 0.6)
        smooth = _periodic_catmull_rom(poly, samples_per_seg=16)
        total = float(np.linalg.norm(np.roll(smooth, -1, 0) - smooth, axis=1).sum())
        if total < 1e-6:
            continue
        # 목표 랩 길이로 전체 스케일 정규화 -> extent/곡률이 실전 규모로 맞춰진다
        smooth = smooth * (rng.uniform(*params.wp_len_range) / total)
        total = float(np.linalg.norm(np.roll(smooth, -1, 0) - smooth, axis=1).sum())
        cl = _resample_equal_arclength(smooth, max(64, int(round(total / ds))))
        for _ in range(2):                                  # ±0.6m boxcar 2회 평활
            k = np.ones(9) / 9.0
            pad = np.vstack([cl[-4:], cl, cl[:4]])
            cl = np.stack([np.convolve(pad[:, d], k, mode="valid") for d in (0, 1)], 1)
        total = float(np.linalg.norm(np.roll(cl, -1, 0) - cl, axis=1).sum())
        cl = _resample_equal_arclength(cl, max(64, int(round(total / ds))))
        cl -= cl.mean(axis=0, keepdims=True)
        kap = np.abs(centerline_features(cl)["kappa"])
        if kap.max() > params.wp_kappa_max:                 # 남은 킹크 배제
            continue
        hw = clip_width_by_curvature(cl, make_waypoint_width(len(cl), rng, params), params)
        hw = clip_width_by_proximity(cl, hw, params)
        hw = smooth_half_width(hw, params)                # 코너 apex 노치(=벽 첨점) 제거
        hw = shrink_width_until_clean(cl, hw, params)     # 잔여 평행곡선 교차 제거
        if hw is None:
            continue
        if (1.0 / np.maximum(kap, 1e-6) + hw).min() < params.wp_veh_r_min + params.wp_outer_margin:
            continue
        return cl.astype(np.float64), hw
    return None


def discover_f1tenth_tracks(root: str) -> list[str]:
    """root 아래 <트랙명>/<트랙명>_centerline.csv 목록 (이름순 정렬)."""
    import glob as _glob
    return sorted(_glob.glob(os.path.join(root, "*", "*_centerline.csv")))


def load_f1tenth_centerline(csv_path: str, params: TrackParams = TrackParams(),
                            scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """f1tenth_racetracks *_centerline.csv -> (중심선 (n,2), half-width (n,)).

    - 컬럼: x_m, y_m, w_tr_right_m, w_tr_left_m (경계까지의 거리).
    - 좌/우 폭이 비대칭이면 중심선을 법선 방향으로 재정렬해 대칭 half-width 로
      변환한다 (우리 파이프라인은 중심선 ± hw 대칭 트랙을 가정).
    - ★ 절차 트랙과 동일한 등간격 params.ds 로 리샘플 — 인덱스 기반 관측
      (전방 곡률/폭 lookahead, 레이캐스트 윈도)의 인덱스당 거리를 통일.
    - 무게중심 원점화 (월드 배치는 env_origins 가 담당).
    - scale: 중심선 좌표만 축소(폭은 원본 유지 — 차량 크기가 고정이므로).
      2026-07-21: 실서킷은 실전 대회장(31~43m) 대비 6~13배라 1/3 축소를 기본으로
      하되, 축소하면 곡률이 1/s 배가 되므로 트랙별로 |κ|max 상한을 넘지 않는
      선까지만 축소한다(급한 서킷은 덜 축소 -> 곡률 분포가 고르게 유지).
    """
    import csv as _csv
    rows = np.array([[float(v) for v in r[:4]] for r in _csv.reader(open(csv_path))
                     if r and not r[0].lstrip().startswith("#")])
    xy = rows[:, :2]      # 축소는 평활 뒤에 (정점 노이즈가 곡률 판정을 왜곡하므로)
    w_r = rows[:, 2]      # 폭은 원본 유지 (차량 크기 고정 — 축소하면 주행 불가)
    w_l = rows[:, 3]
    if np.linalg.norm(xy[0] - xy[-1]) < 1e-6:      # 폐곡선 중복 끝점 제거
        xy, w_r, w_l = xy[:-1], w_r[:-1], w_l[:-1]
    return process_measured_centerline(xy, w_r, w_l, params, scale,
                                       tag=os.path.basename(csv_path))


def process_measured_centerline(xy: np.ndarray, w_r: np.ndarray, w_l: np.ndarray,
                                params: TrackParams = TrackParams(), scale: float = 1.0,
                                tag: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """실측형 (중심선 xy + 좌/우 폭) -> 학습이 쓰는 (중심선, half-width).

    load_f1tenth_centerline(파일) 과 make_competition_tracks(생성 배열)가 공유하는
    전처리 본체. 대칭화 → 등간격 리샘플 → 정점 노이즈 평활 → (옵션)적응 축소 →
    곡률/근접 폭 클립 → apex 노치 평활 → 벽 교차 제거. 생성기가 이 함수로 검증하면
    학습 로더와 100% 동일한 곡률/폭을 보게 되어 '생성 CSV vs 로드 후'의 곡률 괴리
    (로더 추가 평활로 |κ|가 더 줄던 문제)를 원천 차단한다.
    """
    _, normal, _ = _offsets(xy)
    center = xy + normal * ((w_l - w_r) * 0.5)[:, None]   # 대칭화 재정렬
    hw = 0.5 * (w_l + w_r)
    ds = params.ds if params.ds > 0 else 0.15
    # ---- 정점 노이즈 평활 ----
    # 원본 폴리라인(0.33~0.46m 간격)의 꺾임은 0.15m 리샘플 후 곡률 스파이크
    # (|κ|~2.8, R 0.35m — 실서킷에 없는 값)로 나타나 곡률 관측을 오염시킨다.
    # 1차 등간격 리샘플 -> 주기적 boxcar(±0.75m) 2회(≈삼각커널 1.6m 창) ->
    # 재리샘플. 실제 코너(R>=1.5m)는 보존되고 정점 킹크만 제거된다.
    total = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1).sum()
    n = max(64, int(round(total / ds)))
    center, at = _resample_equal_arclength(center, n, attrs=hw[:, None])
    k = np.ones(11) / 11.0                      # 11점 @0.15m = ±0.75m
    for _ in range(2):
        pad = np.vstack([center[-5:], center, center[:5]])
        center = np.stack([np.convolve(pad[:, d], k, mode="valid") for d in (0, 1)], axis=1)
        padw = np.concatenate([at[-5:, 0], at[:, 0], at[:5, 0]])
        at = np.convolve(padw, k, mode="valid")[:, None]
    # 평활로 호길이가 미세 변화 -> 등간격 재확정
    total = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1).sum()
    n = max(64, int(round(total / ds)))
    center, at = _resample_equal_arclength(center, n, attrs=at)
    center -= center.mean(axis=0, keepdims=True)
    center = center.astype(np.float64)
    # ---- 트랙별 적응 축소 (평활 후 곡률 기준) ----
    # 축소하면 곡률이 1/s 배가 된다. |κ|max 가 wp_kappa_max(=R 0.45m)를 넘으면
    # '벽 접힘 없음(hw<=R)'과 '차량 통과 가능(R+hw>=최소회전반경)'을 동시에
    # 만족할 수 없으므로(그 경계가 R≈0.45m), 그 선까지만 축소한다.
    if scale != 1.0:
        k_hi = float(np.abs(centerline_features(center)["kappa"]).max())
        s = min(1.0, max(scale, k_hi / max(params.wp_kappa_max, 1e-6)))
        if s < 1.0:
            center = center * s
            total = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1).sum()
            center, at = _resample_equal_arclength(
                center, max(64, int(round(total / ds))), attrs=at)
    # 안쪽 벽 뒤집힘 방지 (SLAM 맵은 코너 안쪽의 '열린 공간'까지 폭으로 재므로
    # 급코너에서 hw > R 이 되는 구간이 실제로 존재한다)
    hw_out = clip_width_by_curvature(center, at[:, 0].astype(np.float64), params)
    hw_out = clip_width_by_proximity(center, hw_out, params)
    hw_out = smooth_half_width(hw_out, params)            # 코너 apex 노치(=벽 첨점) 제거
    cleaned = shrink_width_until_clean(center, hw_out, params)
    if cleaned is None:      # 실측/실서킷은 폐기할 수 없으므로 최대한 축소한 값 사용
        cleaned = np.maximum(hw_out * 0.88 ** 8, 0.30)
        if tag:
            print(f"[tracks] 경고: {tag} 벽 교차가 남아 폭을 최대 축소")
    return center, cleaned


def _offsets(cl: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """접선/법선 계산. 반환 (tangent_angle psi, normal(N,2))."""
    nxt = np.roll(cl, -1, axis=0)
    seg = nxt - cl
    psi = np.arctan2(seg[:, 1], seg[:, 0])
    normal = np.stack([-np.sin(psi), np.cos(psi)], axis=1)  # 좌측(+) 법선
    return psi, normal, seg


def build_track_mesh(cl: np.ndarray, params: TrackParams,
                     half_widths: np.ndarray | None = None) -> trimesh.Trimesh:
    """중심선 -> 노면 + 벽 통합 메시. (z=0 평면 위 노면, 양옆 수직 벽)"""
    hw = half_widths[:, None] if half_widths is not None else params.track_width * 0.5
    _, normal, _ = _offsets(cl)
    left = cl + normal * hw
    right = cl - normal * hw
    n = len(cl)

    # ---- 노면(road) : left/right 를 잇는 quad strip ----
    z0 = np.zeros((n, 1))
    Lv = np.hstack([left, z0])
    Rv = np.hstack([right, z0])
    verts = np.vstack([Lv, Rv])           # [0..n-1]=left, [n..2n-1]=right
    faces = []
    for i in range(n):
        j = (i + 1) % n
        li, lj = i, j
        ri, rj = n + i, n + j
        faces.append([li, ri, rj])
        faces.append([li, rj, lj])
    road = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)

    # ---- 좌/우 벽(walls) : 수직 quad ----
    h = params.wall_height
    def wall(boundary):
        b0 = np.hstack([boundary, np.zeros((n, 1))])
        b1 = np.hstack([boundary, np.full((n, 1), h)])
        v = np.vstack([b0, b1])
        f = []
        for i in range(n):
            j = (i + 1) % n
            f.append([i, n + i, n + j])
            f.append([i, n + j, j])
        return trimesh.Trimesh(vertices=v, faces=np.array(f), process=False)

    mesh = trimesh.util.concatenate([road, wall(left), wall(right)])
    return mesh


def centerline_features(cl: np.ndarray, closed: bool = True):
    """중심선으로부터 seg/seglen/s/psi/kappa 계산(numpy). vectorized_track 가 사용."""
    nxt = np.roll(cl, -1, axis=0) if closed else np.vstack([cl[1:], cl[-1]])
    seg = nxt - cl
    seglen = np.maximum(np.linalg.norm(seg, axis=1), 1e-9)
    s = np.concatenate([[0.0], np.cumsum(seglen)[:-1]])
    total_s = float(np.sum(seglen)) if closed else float(s[-1])
    psi = np.arctan2(seg[:, 1], seg[:, 0])
    dpsi = (np.diff(psi, append=psi[0]) + np.pi) % (2 * np.pi) - np.pi
    kappa = dpsi / seglen
    return dict(seg=seg, seglen=seglen, s=s, total_s=total_s, psi=psi, kappa=kappa)
