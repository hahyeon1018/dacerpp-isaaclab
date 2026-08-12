"""환경들에 트랙(절차 생성 9종 + f1tenth 실서킷)을 배정하고 시작 포즈를 계산한다.

물리는 평면(plane)에서 진행하고, 트랙은 '중심선+폭'으로 해석적으로 정의된다.
(코스이탈은 보상/종료로 처리 -> 거대한 충돌 메시 불필요)

트랙 구성 (총 9 + 23 = 32종):
  - [0..NUM_LAYOUTS-1]   : 절차 생성 서킷 (하모닉 프로파일, 가변 폭 1~5m)
  - [NUM_LAYOUTS..]      : f1tenth_racetracks 실서킷 (1:10 스케일, 폭 CSV 값)
★ 모든 중심선은 공통 등간격 ds(TrackParams.ds=0.15m)로 리샘플된다 —
  인덱스 기반 관측(전방 곡률/폭 lookahead, 레이캐스트 윈도)의 '인덱스당 거리'가
  트랙 종류와 무관하게 동일해야 관측 기준이 일치하기 때문 (tracks.py 참조).

제공 정보:
  - centerlines / half_widths : 트랙별 캐논컬 중심선·반폭 (로컬 원점 기준, 가변 길이)
  - env_track_type            : (num_envs,) 각 환경의 트랙 종류
  - start_xy / start_yaw (로컬): 중심선 시작점/접선
env_origins(월드 그리드)은 Isaac Lab scene 이 잡아주며, 로컬=world-origin 으로 변환.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .tracks import (make_centerline, make_width_profile, TrackParams, NUM_LAYOUTS,
                     centerline_features, discover_f1tenth_tracks, load_f1tenth_centerline,
                     make_waypoint_track)


@dataclass
class TrackFieldCfg:
    num_envs: int = 2048
    difficulty: float = 0.6                # 절차 트랙 난이도 (고정 분포)
    params: TrackParams = field(default_factory=TrackParams)
    start_lateral_offset: float = 0.6      # 두 차량 좌/우 출발 간격(m)
    start_longitudinal_gap: int = 0        # 그리드 출발 시 인덱스 차(0=나란히)
    # ---- 절차 트랙 생성 (2026-07-21: 하모닉 -> 웨이포인트 기반으로 교체) ----
    # 하모닉 방식은 star-shaped 위상만 만들어 S자/헤어핀/급코너가 불가능했고,
    # 그 결과 '좁고 급한 구간'이 학습 분포에 0% 라 새 실전 맵에서 주행 실패.
    # 웨이포인트 생성기는 실행마다 num_procedural 종을 새로 뽑는다(무한 다양성).
    num_procedural: int = 697
    procedural_seed: int | None = None     # None = 실행마다 다른 트랙 세트
    use_waypoint_gen: bool = True          # False = 구 하모닉 레이아웃(NUM_LAYOUTS 종)
    # f1tenth_racetracks 경로 ("" = <repo>/f1tenth_racetracks 자동 탐색).
    # 폴더가 없으면 절차 트랙만 사용.
    f1tenth_dir: str = ""
    # f1tenth 실서킷 + 그 폴더의 실측 SLAM 맵(hangeong/hall 등) 로드 여부.
    # False + num_procedural=0 이면 comp 폴더만 사용(대회 코스 형상 집중 학습).
    # 근거(2026-07-31 측정): f1tenth 는 장변 62m/랩 199m 로 실측 대회코스(16m/43m)의
    # 4배 크기·완만(급코너 1.2% vs 실측 5%)이라 OOD, 절차는 크기는 맞지만 뭉툭(|κ|max
    # 중앙 1.41 vs 실측 1.89)해 급한 V 노출을 희석. 코스 형상이 알려진 대회에선 comp+실측
    # 집중이 유리(단, 전혀 다른 race-day 코스 강건성은↓ — 연습맵=배포코스라 수용).
    use_f1tenth: bool = True
    # 대회 코스 형상 변형본 ("" = <repo>/competition_tracks 자동 탐색).
    # scripts/make_competition_tracks.py 로 생성. 실측 맵과 동급(축소 없음)으로
    # 취급하고, 배포 대상 코스이므로 배정 가중치를 줄 수 있다(comp_weight).
    comp_dir: str = ""
    comp_weight: int = 1                   # 1 = 다른 트랙과 동일 비중
    # 실서킷 중심선 축소율(폭은 원본 유지). 1/3 = 실전 대회장 규모에 근접.
    # 트랙별로 곡률 상한을 넘지 않는 선까지만 축소된다(tracks.load_f1tenth_centerline).
    f1tenth_scale: float = 1.0 / 3.0
    # 평가용 랜덤 트랙 배정: True 면 env 마다 전체 트랙 중 무작위로 뽑는다
    # (seed=None -> 실행마다 다름). 학습(train.py)은 기본 False = i%G 균형 배정 유지.
    random_tracks: bool = False
    random_seed: int | None = None
    # 지정 트랙 고정 배정(평가용): 트랙 이름 목록을 주면 앞쪽 env 에 순서대로
    # 배정하고, 남는 env 는 '지정 트랙을 제외한' 나머지에서 무작위로 채운다.
    # (예: ("hall","teras") + num_envs=6 -> hall, teras, 랜덤4). random_tracks 보다 우선.
    pinned_tracks: tuple = ()


class TrackField:
    def __init__(self, cfg: TrackFieldCfg):
        self.cfg = cfg
        # ---- 절차 생성 트랙 ----
        self.centerlines: List[np.ndarray] = []
        self.half_widths: List[np.ndarray] = []
        self.track_names: List[str] = []
        if cfg.use_waypoint_gen:      # 웨이포인트 기반 (기본) — 실행마다 새 세트
            if cfg.procedural_seed is None:
                # ★ 결정한 시드를 cfg 에 되써서, 같은 cfg 로 TrackField 를 다시
                # 만들어도(예: env_spacing 계산용 + RacingEnv 내부) 동일한 트랙
                # 세트가 나오게 한다. 세트가 어긋나면 spacing 이 실제 트랙 크기와
                # 안 맞아 이웃 env 트랙이 겹칠 수 있다.
                cfg.procedural_seed = int(np.random.SeedSequence().entropy % (2 ** 31))
            seed0 = cfg.procedural_seed
            self.procedural_seed = seed0
            s, made = 0, 0
            while made < cfg.num_procedural and s < cfg.num_procedural * 20:
                r = make_waypoint_track(seed0 + s, cfg.params)
                s += 1
                if r is None:
                    continue
                self.centerlines.append(r[0])
                self.half_widths.append(r[1])
                self.track_names.append(f"gen{made}")
                made += 1
        else:                          # 구 하모닉 레이아웃 (호환용)
            self.procedural_seed = None
            for g in range(NUM_LAYOUTS):
                cl = make_centerline(g, cfg.difficulty, cfg.params)
                self.centerlines.append(cl)
                self.half_widths.append(
                    make_width_profile(g, cfg.difficulty, cfg.params, len(cl)))
                self.track_names.append(f"proc{g}")
        n_proc = len(self.centerlines)

        # ---- f1tenth 실서킷 (같은 ds 로 리샘플, 폭은 CSV 값) ----
        root = cfg.f1tenth_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "f1tenth_racetracks")
        csvs = (discover_f1tenth_tracks(root)
                if (cfg.use_f1tenth and os.path.isdir(root)) else [])
        for p in csvs:
            # 실측 대회장 맵(race_stack global_waypoints.json 에서 변환)은 실제 크기가
            # 곧 배포 대상이므로 축소하지 않는다. f1tenth 실서킷(F1/DTM 1:10)만 축소.
            is_measured = os.path.isfile(os.path.join(os.path.dirname(p),
                                                      "global_waypoints.json"))
            cl, hw = load_f1tenth_centerline(
                p, cfg.params, 1.0 if is_measured else cfg.f1tenth_scale)
            self.centerlines.append(cl)
            self.half_widths.append(hw)
            self.track_names.append(os.path.basename(p).replace("_centerline.csv", ""))
        n_f1 = len(csvs)

        # ---- 대회 코스 변형본 (배포 대상 형상 — 축소 없이 실측 동급) ----
        comp_root = cfg.comp_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "competition_tracks")
        comp_csvs = discover_f1tenth_tracks(comp_root) if os.path.isdir(comp_root) else []
        self.comp_ids: List[int] = []
        for p in comp_csvs:
            cl, hw = load_f1tenth_centerline(p, cfg.params, 1.0)
            self.comp_ids.append(len(self.centerlines))
            self.centerlines.append(cl)
            self.half_widths.append(hw)
            self.track_names.append(os.path.basename(p).replace("_centerline.csv", ""))

        self.num_track_types = len(self.centerlines)
        self.features = [centerline_features(cl) for cl in self.centerlines]
        G = self.num_track_types

        # ---- env -> 트랙 배정 ----
        pinned: list[int] = []
        if cfg.pinned_tracks:   # 이름 -> 인덱스 (대소문자 무시)
            by_name = {n.lower(): i for i, n in enumerate(self.track_names)}
            for nm in cfg.pinned_tracks:
                key = str(nm).strip().lower()
                if key not in by_name:
                    raise ValueError(
                        f"[TrackField] 알 수 없는 트랙 이름: {nm!r}\n"
                        f"  사용 가능({G}종): {', '.join(self.track_names)}")
                pinned.append(by_name[key])

        if pinned:               # 평가용: 지정 트랙 우선 배정 + 나머지 랜덤 충원
            rng = np.random.default_rng(cfg.random_seed)
            n_pin = min(len(pinned), cfg.num_envs)
            if n_pin < len(pinned):
                print(f"[TrackField] 경고: num_envs({cfg.num_envs}) < 지정 트랙 수"
                      f"({len(pinned)}) -> 앞의 {n_pin}개만 배정")
            n_fill = cfg.num_envs - n_pin
            rest = [g for g in range(G) if g not in set(pinned)]
            if n_fill > 0:
                pool = rest if rest else pinned      # 전 트랙이 지정된 극단 케이스
                fill = rng.choice(pool, size=n_fill, replace=len(pool) < n_fill)
            else:
                fill = np.zeros(0, dtype=np.int64)
            self.env_track_type = np.concatenate(
                [np.asarray(pinned[:n_pin], dtype=np.int64),
                 np.asarray(fill, dtype=np.int64)])
        elif cfg.random_tracks:  # 평가용: 전체 무작위 배정
            rng = np.random.default_rng(cfg.random_seed)
            self.env_track_type = rng.integers(0, G, cfg.num_envs).astype(np.int64)
        else:                    # 학습용: env i -> track (i % G) 균형 배정
            # comp_weight>1 이면 대회 코스를 그 배수만큼 풀에 중복 투입 (노출 강화)
            pool = np.arange(G)
            if cfg.comp_weight > 1 and self.comp_ids:
                extra = np.tile(np.asarray(self.comp_ids, dtype=np.int64),
                                cfg.comp_weight - 1)
                pool = np.concatenate([pool, extra])
            self.env_track_type = pool[np.arange(cfg.num_envs) % len(pool)].astype(np.int64)

        # 트랙별 시작점/접선(로컬)
        start_xy = np.zeros((cfg.num_envs, 2), np.float32)
        start_yaw = np.zeros((cfg.num_envs,), np.float32)
        for i in range(cfg.num_envs):
            g = self.env_track_type[i]
            cl = self.centerlines[g]
            start_xy[i] = cl[0]
            start_yaw[i] = self.features[g]["psi"][0]
        self.start_xy = start_xy
        self.start_yaw = start_yaw

        # 트랙 바운딩(그리드 spacing 계산용) — 최대 폭 기준으로 벽 돌출까지 포함.
        # f1tenth 대형 서킷(extent ~95m)이 spacing 을 지배한다 (Monza 기준 ~197m).
        max_extent = max(np.abs(cl).max() for cl in self.centerlines)
        self.suggested_env_spacing = float(2.0 * max_extent + cfg.params.width_max + 6.0)
        n_pts = [len(c) for c in self.centerlines]
        gen = f"웨이포인트 {n_proc}종(seed {self.procedural_seed})" if cfg.use_waypoint_gen \
              else f"하모닉 {n_proc}종"
        comp_txt = f" + 대회코스 {len(comp_csvs)}종(x{cfg.comp_weight})" if comp_csvs else ""
        print(f"[TrackField] 트랙 {G}종 ({gen} + f1tenth {n_f1}{comp_txt}) | "
              f"점수 {min(n_pts)}~{max(n_pts)} @ds={cfg.params.ds}m | "
              f"spacing {self.suggested_env_spacing:.0f}m")

    def start_pose_offsets(self):
        """car_a(보수/CVaR), car_b(공격/Pow) 의 로컬 출발 오프셋(법선방향 ±)."""
        off = self.cfg.start_lateral_offset
        nrm = np.stack([-np.sin(self.start_yaw), np.cos(self.start_yaw)], axis=1)  # (N,2)
        xy_a = self.start_xy + nrm * (+0.5 * off)
        xy_b = self.start_xy + nrm * (-0.5 * off)
        return xy_a.astype(np.float32), xy_b.astype(np.float32), self.start_yaw
