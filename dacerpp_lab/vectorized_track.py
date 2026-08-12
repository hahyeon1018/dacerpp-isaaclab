"""torch 로 벡터화한 중심선 투영/진행도 계산.

업로드본 track.py(ReferencePath, numpy, 단일차량)를 다중환경용으로 재작성.
- 8종 트랙의 중심선을 (G, L, 2) 로 스택(패딩)해 device 에 상주.
- project(local_pos, track_type) 가 전 차량(N대)을 한 번에 처리:
    s(진행도), signed_lateral(횡오차), seg_index, psi_ref(접선각)
- 좌표는 '로컬'(차량 world - env_origin). 같은 track_type 은 같은 캐논컬 중심선 공유.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

from .tracks import centerline_features


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + torch.pi) % (2 * torch.pi) - torch.pi


def overlay_opponent(scan: torch.Tensor, beam_angles: torch.Tensor,
                     rel_body_xy: torch.Tensor, max_range: float,
                     opp_radius: float, min_half_ext: float = 0.0) -> torch.Tensor:
    """스캔 (N,B) 에 상대(원판 등가) footprint 를 반영.

    방위각 ±half_ext(=atan(r/rng)) 범위의 빔에 표면 거리(rng - r)를 min 으로
    합성한다(벽이 더 가까우면 가림 유지).

    min_half_ext: half_ext 의 하한(rad). 감지 박스(12×12cm)처럼 작은 물체는 단일
    시점 32빔에선 빔 사이로 빠지지만, 실측 Livox 는 ~100ms 누적으로 그 섹터를 채워
    반드시 검출한다. 이를 재현하려 half_ext 를 최소 '반 섹터'(beam_spacing/2)로 바닥쳐
    물체가 든 섹터의 빔이 항상 잡히게 한다(pseudo-scan 정합).
    """
    rng = rel_body_xy.norm(dim=1)                                       # (N,)
    brg = torch.atan2(rel_body_xy[:, 1], rel_body_xy[:, 0])             # (N,)
    half_ext = torch.atan2(torch.full_like(rng, opp_radius), rng).clamp_min(min_half_ext)
    diff = wrap_to_pi(beam_angles.unsqueeze(0) - brg.unsqueeze(1)).abs()  # (N,B)
    surf = (rng - opp_radius).clamp_min(0.01).unsqueeze(1)              # (N,1)
    hit = (diff <= half_ext.unsqueeze(1)) & (rng < max_range).unsqueeze(1)
    return torch.where(hit, torch.minimum(scan, surf), scan)


def overlay_obstacles(scan: torch.Tensor, origin_xy: torch.Tensor,
                      beam_world_angles: torch.Tensor, obs_verts: torch.Tensor,
                      obs_active: torch.Tensor, max_range: float) -> torch.Tensor:
    """스캔 (N,B) 에 다각형 장애물 footprint 를 ray-segment 로 반영.

    상대차(overlay_opponent)는 원판 근사지만, 장애물은 크기·모양(직육면체=사각형,
    삼각기둥=삼각형, 원기둥=팔각형 근사)이 다양하므로 각 장애물을 2D 볼록 다각형
    footprint 로 두고 각 빔이 그 변들과 만나는 최소 거리를 스캔에 min 합성한다.
    실차 32빔 의사 스캔에 물리 장애물이 잡히는 것과 동일한 채널.

    scan            : (N,B) 벽 스캔(오버레이 대상)
    origin_xy       : (N,2) 빔 원점(차량 로컬 좌표)
    beam_world_angles: (N,B) 빔의 월드 방위각(= yaw + beam_angle)
    obs_verts       : (N,K,V,2) 장애물 다각형 정점(로컬 좌표, CCW, 패딩=마지막 정점
                      반복 -> 퇴화 edge 는 denom≈0 으로 자동 무시)
    obs_active      : (N,K) bool — 활성 장애물만 반영
    """
    N, B = scan.shape
    K, V = obs_verts.shape[1], obs_verts.shape[2]
    p1 = obs_verts                                       # (N,K,V,2)
    p2 = torch.roll(obs_verts, -1, dims=2)               # (N,K,V,2) edge 끝점
    ex = (p2 - p1)[..., 0]                               # (N,K,V)
    ey = (p2 - p1)[..., 1]
    dx = torch.cos(beam_world_angles)                    # (N,B)
    dy = torch.sin(beam_world_angles)
    # 브로드캐스트: 빔 (N,B,1,1) x edge (N,1,K,V)
    dx_ = dx[:, :, None, None]; dy_ = dy[:, :, None, None]
    ex_ = ex[:, None]; ey_ = ey[:, None]                 # (N,1,K,V)
    denom = dx_ * ey_ - dy_ * ex_                        # (N,B,K,V) = cross(d, e)
    ok = denom.abs() > 1e-9
    denom_s = torch.where(ok, denom, torch.ones_like(denom))
    ox = origin_xy[:, 0][:, None, None, None]
    oy = origin_xy[:, 1][:, None, None, None]
    diffx = p1[..., 0][:, None] - ox                     # (N,B,K,V) = (p1 - O).x
    diffy = p1[..., 1][:, None] - oy
    t = (diffx * ey_ - diffy * ex_) / denom_s            # 빔 진행거리(단위 방향)
    u = (diffx * dy_ - diffy * dx_) / denom_s            # edge 매개변수 [0,1]
    active = obs_active[:, None, :, None]                # (N,1,K,1)
    hit = ok & active & (t > 1e-4) & (t < max_range) & (u >= 0.0) & (u <= 1.0)
    t = torch.where(hit, t, torch.full_like(t, float("inf")))
    obs_dist = t.amin(dim=(2, 3))                        # (N,B) 빔별 최근접 장애물 거리
    return torch.minimum(scan, obs_dist)


class VectorizedTracks:
    def __init__(self, centerlines: List[np.ndarray], device="cuda:0", closed: bool = True,
                 half_widths: List[np.ndarray] | float | None = None):
        """half_widths: 트랙별 (n,) half-width 프로파일 리스트, 스칼라(전 구간 동일),
        None(기본 1.5m). 벽/이탈판정/폭 관측에 사용."""
        self.device = torch.device(device)
        self.G = len(centerlines)
        self.L = max(len(cl) for cl in centerlines)
        self.closed = closed

        pts = np.zeros((self.G, self.L, 2), np.float32)
        seg = np.zeros((self.G, self.L, 2), np.float32)
        seglen = np.ones((self.G, self.L), np.float32)
        s = np.zeros((self.G, self.L), np.float32)
        psi = np.zeros((self.G, self.L), np.float32)
        kappa = np.zeros((self.G, self.L), np.float32)
        hw = np.full((self.G, self.L), 1.5, np.float32)
        valid = np.zeros((self.G, self.L), np.float32)
        total_s = np.zeros((self.G,), np.float32)
        length = np.zeros((self.G,), np.int64)

        for g, cl in enumerate(centerlines):
            n = len(cl)
            f = centerline_features(cl, closed=closed)
            pts[g, :n] = cl
            seg[g, :n] = f["seg"]
            seglen[g, :n] = f["seglen"]
            s[g, :n] = f["s"]
            psi[g, :n] = f["psi"]
            kappa[g, :n] = f["kappa"]
            if half_widths is not None:
                hw[g, :n] = half_widths if np.isscalar(half_widths) else half_widths[g][:n]
            valid[g, :n] = 1.0
            total_s[g] = f["total_s"]
            length[g] = n

        t = lambda x: torch.as_tensor(x, device=self.device)
        self.pts = t(pts); self.seg = t(seg); self.seglen = t(seglen)
        self.s = t(s); self.psi = t(psi); self.kappa = t(kappa); self.hw = t(hw)
        self.valid = t(valid); self.total_s = t(total_s); self.length = t(length)

        # track_type 별 gather 캐시 (환경의 track_type 은 불변 -> 매 호출 재수집 방지)
        self._cache_tt: torch.Tensor | None = None
        self._cache: dict | None = None

        # 벽 폴리라인 (build_walls 로 생성)
        self._wall_pts: tuple[torch.Tensor, torch.Tensor] | None = None
        self._wall_seg: tuple[torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------ #
    # 벽 레이캐스트 (실제 트랙의 덕트 벽 대응 — 곡률이 정확히 반영된 스캔)
    # ------------------------------------------------------------------ #
    def build_walls(self, offset: float = 0.0) -> None:
        """중심선 좌/우 지역 half-width(+offset) 오프셋 폴리라인(벽)을 저장.

        폭은 정점별 self.hw 프로파일을 사용한다(가변 폭 트랙 지원)."""
        nrm = torch.stack([-torch.sin(self.psi), torch.cos(self.psi)], dim=-1)  # (G,L,2)
        w = (self.hw + offset).unsqueeze(-1)                                    # (G,L,1)
        wl = self.pts + w * nrm
        wr = self.pts - w * nrm
        # 세그먼트 i 는 벽점 i -> (i+1)%Lg 를 잇는다 (패딩 행은 window 가 접근 안 함)
        idx = torch.arange(self.L, device=self.device).unsqueeze(0)             # (1,L)
        nxt = ((idx + 1) % self.length.unsqueeze(1)).unsqueeze(-1).expand(-1, -1, 2)
        self._wall_pts = (wl, wr)
        self._wall_seg = (wl.gather(1, nxt) - wl, wr.gather(1, nxt) - wr)

    @torch.no_grad()
    def raycast(self, pos: torch.Tensor, ang: torch.Tensor, track_type: torch.Tensor,
                idx0: torch.Tensor, half_window: int, max_range: float) -> torch.Tensor:
        """좌/우 벽 폴리라인에 대한 2D 레이캐스트.

        pos:(N,2) 로컬 위치, ang:(N,B) 월드 기준 빔 각, idx0:(N,) 현재 중심선 인덱스.
        idx0 주변 ±half_window 세그먼트만 검사한다(빔 최대거리보다 넓게 잡을 것).
        주의: 폐곡선 반대편(인필드 건너편) 벽은 윈도 밖이라 안 잡힌다 —
        트랙 지름이 scan_max_range 보다 크면 실제로도 범위 밖이므로 무시 가능.
        반환 (N,B): 최근접 벽까지 거리(무히트 = max_range).
        """
        assert self._wall_pts is not None, "build_walls() 를 먼저 호출할 것"
        Lg = self.length[track_type]                                        # (N,)
        offs = torch.arange(-half_window, half_window, device=self.device)  # (2W,)
        j = (idx0.unsqueeze(1) + offs.unsqueeze(0)) % Lg.unsqueeze(1)       # (N,2W)
        ttj = track_type.unsqueeze(1).expand_as(j)
        a = torch.cat([self._wall_pts[0][ttj, j], self._wall_pts[1][ttj, j]], dim=1)  # (N,S,2)
        s = torch.cat([self._wall_seg[0][ttj, j], self._wall_seg[1][ttj, j]], dim=1)  # (N,S,2)

        # 광선 o + t*d 와 세그먼트 a + u*s 의 교차: t=cross(e,s)/cross(d,s), u=cross(e,d)/cross(d,s)
        e = a - pos.unsqueeze(1)                                            # (N,S,2)
        d = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)           # (N,B,2)
        denom = (d[..., 0].unsqueeze(2) * s[..., 1].unsqueeze(1)
                 - d[..., 1].unsqueeze(2) * s[..., 0].unsqueeze(1))         # (N,B,S)
        e_x_s = (e[..., 0] * s[..., 1] - e[..., 1] * s[..., 0]).unsqueeze(1)  # (N,1,S)
        e_x_d = (e[..., 0].unsqueeze(1) * d[..., 1].unsqueeze(2)
                 - e[..., 1].unsqueeze(1) * d[..., 0].unsqueeze(2))         # (N,B,S)
        ok = denom.abs() > 1e-8
        denom = torch.where(ok, denom, torch.ones_like(denom))
        t = e_x_s / denom
        u = e_x_d / denom
        # u 허용오차: 빔이 정점을 정확히 지날 때 fp 오차로 양쪽 세그먼트 모두
        # 미세하게 벗어나 "틈새로 새는" 것을 방지 (중복 히트는 min 이라 무해)
        eps = 1e-4
        hit = ok & (u >= -eps) & (u <= 1.0 + eps) & (t >= 0.0)
        t = torch.where(hit, t, torch.full_like(t, float("inf")))
        return t.min(dim=2).values.clamp(max=max_range)

    def _gathered(self, track_type: torch.Tensor) -> dict:
        """(N,) track_type 으로 인덱싱한 (N,L,...) 텐서 묶음. 같은 텐서 객체면 캐시 재사용."""
        if self._cache_tt is not track_type:
            self._cache = dict(
                pts=self.pts[track_type], seg=self.seg[track_type],
                seglen=self.seglen[track_type], s=self.s[track_type],
                psi=self.psi[track_type], valid=self.valid[track_type],
                kappa=self.kappa[track_type], hw=self.hw[track_type],
                length=self.length[track_type],
            )
            self._cache_tt = track_type
        return self._cache

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def project(self, pos_local: torch.Tensor, track_type: torch.Tensor):
        """pos_local:(N,2), track_type:(N,) -> dict(s, lateral, idx, psi, kappa_idx).

        가장 가까운 정점 i0 를 찾고, 인접 두 세그먼트(i0-1, i0)에 투영해 더 가까운 쪽 채택.
        """
        N = pos_local.shape[0]
        ar = torch.arange(N, device=self.device)
        gb = self._gathered(track_type)
        pts = gb["pts"]                     # (N,L,2)
        seg = gb["seg"]                     # (N,L,2)
        seglen = gb["seglen"]               # (N,L)
        s_arr = gb["s"]                     # (N,L)
        psi = gb["psi"]                     # (N,L)
        valid = gb["valid"]                 # (N,L)
        Lg = gb["length"]                   # (N,)

        d2 = ((pts - pos_local.unsqueeze(1)) ** 2).sum(-1)        # (N,L)
        d2 = torch.where(valid > 0, d2, torch.full_like(d2, 1e12))
        i0 = d2.argmin(dim=1)                                     # (N,)

        def project_seg(idx):
            idx = idx % Lg                                       # 폐곡선 wrap
            a = pts[ar, idx]                                     # (N,2)
            sg = seg[ar, idx]                                    # (N,2)
            L2 = (seglen[ar, idx] ** 2).clamp_min(1e-9)
            tparam = ((pos_local - a) * sg).sum(-1) / L2
            tparam = tparam.clamp(0.0, 1.0)
            proj = a + tparam.unsqueeze(-1) * sg
            diff = pos_local - proj
            dist = diff.norm(dim=-1)
            cross = sg[:, 0] * diff[:, 1] - sg[:, 1] * diff[:, 0]
            signed = dist * torch.where(cross >= 0, 1.0, -1.0)
            s_val = s_arr[ar, idx] + tparam * seglen[ar, idx]
            return dist, s_val, signed, idx

        d_a, s_a, lat_a, idx_a = project_seg(i0)
        d_b, s_b, lat_b, idx_b = project_seg((i0 - 1) % Lg)
        pick_a = d_a <= d_b
        s_val = torch.where(pick_a, s_a, s_b)
        lateral = torch.where(pick_a, lat_a, lat_b)
        seg_idx = torch.where(pick_a, idx_a, idx_b)
        psi_ref = psi[ar, seg_idx]
        return dict(s=s_val, lateral=lateral, idx=seg_idx, psi=psi_ref)

    @torch.no_grad()
    def lookahead_curvature(self, idx: torch.Tensor, track_type: torch.Tensor,
                            offsets: Sequence[int]) -> torch.Tensor:
        """(N, len(offsets)) 전방 곡률."""
        N = idx.shape[0]
        ar = torch.arange(N, device=self.device)
        gb = self._gathered(track_type)
        Lg = gb["length"]
        kappa = gb["kappa"]
        cols = []
        for off in offsets:
            j = (idx + off) % Lg
            cols.append(kappa[ar, j])
        return torch.stack(cols, dim=1)

    @torch.no_grad()
    def lookahead_width(self, idx: torch.Tensor, track_type: torch.Tensor,
                        offsets: Sequence[int]) -> torch.Tensor:
        """(N, len(offsets)) 지역/전방 half-width. offsets 에 0 을 넣으면 현재 폭."""
        N = idx.shape[0]
        ar = torch.arange(N, device=self.device)
        gb = self._gathered(track_type)
        Lg = gb["length"]
        hw = gb["hw"]
        cols = []
        for off in offsets:
            j = (idx + off) % Lg
            cols.append(hw[ar, j])
        return torch.stack(cols, dim=1)

    def total_s_of(self, track_type: torch.Tensor) -> torch.Tensor:
        return self.total_s[track_type]
