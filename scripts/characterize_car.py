#!/usr/bin/env python3
"""차량 물리 특성 개방루프 측정 (정책 없이 스크립트 명령).

목적: 시뮬 차량이 실제로 낼 수 있는 (1) 횡그립 한계(최대 a_lat), (2) 조향각별
요레이트 응답, (3) 제동 한계를 측정한다. 학습 정책의 과감속/언더스티어가
'물리 한계' 때문인지 '정책의 요구 부족' 때문인지 판별하는 근거.

RacingEnv 를 상속하되 종료/리스폰/보상을 전부 무력화 -> 차는 평면 위에서
자유롭게 원을 그린다(물리 조건은 학습과 동일: 같은 지면 재질/차량 USD/솔버).

실행:
  python -u scripts/characterize_car.py --headless
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--physics_hz", type=int, default=120, help="물리 주파수 (컨트롤은 30Hz 고정)")
parser.add_argument("--pos_iters", type=int, default=None, help="솔버 position iterations 오버라이드")
parser.add_argument("--vel_iters", type=int, default=None, help="솔버 velocity iterations 오버라이드")
parser.add_argument("--friction_offset", type=float, default=None,
                    help="physx.friction_offset_threshold (기본 0.04m — 5cm 바퀴에 과대 의심)")
parser.add_argument("--friction_corr", type=float, default=None,
                    help="physx.friction_correlation_distance (기본 0.025m)")
parser.add_argument("--contact_offset", type=float, default=None, help="충돌체 contactOffset")
parser.add_argument("--rest_offset", type=float, default=None, help="충돌체 restOffset")
parser.add_argument("--pgs", action="store_true", help="TGS 대신 PGS 솔버")
parser.add_argument("--no_gyro", action="store_true", help="자이로스코픽 힘 비활성 (고속 자전 바디 적분 안정성 테스트)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from dacerpp_lab.env_cfg import RacingEnvCfg
from dacerpp_lab.racing_env import RacingEnv

MU_G = 0.65 * 9.81   # main() 에서 cfg.tire.mu 로 갱신
L_WB = 0.33   # 실측 축거


class OpenLoopEnv(RacingEnv):
    """종료/보상/리스폰 없는 물리 전용 환경."""

    def _get_dones(self):
        N, dev = self.num_envs, self.device
        z = torch.zeros(N, dtype=torch.bool, device=dev)
        self._term_a, self._term_b, self._trunc = z, z.clone(), z.clone()
        self._rew_a = torch.zeros(N, device=dev)
        self._rew_b = torch.zeros(N, device=dev)
        return z, z.clone()

    def _get_observations(self):
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = RacingEnvCfg()
    global MU_G
    MU_G = cfg.tire.mu * 9.81   # 마찰 한계 횡가속 (env_cfg.TireModelCfg.mu 기준)
    cfg.sim.dt = 1.0 / args.physics_hz
    cfg.decimation = max(1, args.physics_hz // 30)     # 컨트롤 30Hz 유지
    cfg.sim.render_interval = cfg.decimation
    if args.friction_offset is not None:
        cfg.sim.physx.friction_offset_threshold = args.friction_offset
    if args.friction_corr is not None:
        cfg.sim.physx.friction_correlation_distance = args.friction_corr
    if args.pgs:
        cfg.sim.physx.solver_type = 0
    if args.no_gyro or any(x is not None for x in (args.pos_iters, args.vel_iters,
                                                   args.contact_offset, args.rest_offset)):
        import isaaclab.sim as sim_utils
        from dacerpp_lab import car_cfg as _cc
        _orig = _cc.make_car_cfg
        def _patched(prim_path, pd):
            c = _orig(prim_path, pd)
            if args.pos_iters is not None:
                c.spawn.articulation_props.solver_position_iteration_count = args.pos_iters
            if args.vel_iters is not None:
                c.spawn.articulation_props.solver_velocity_iteration_count = args.vel_iters
            if args.contact_offset is not None or args.rest_offset is not None:
                c.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
                    contact_offset=args.contact_offset, rest_offset=args.rest_offset)
            if args.no_gyro:
                c.spawn.rigid_props.enable_gyroscopic_forces = False
            return c
        _cc.make_car_cfg = _patched
        import dacerpp_lab.racing_env as _re
        _re.make_car_cfg = _patched
    print(f"[cfg] physics {args.physics_hz}Hz decim {cfg.decimation} "
          f"pos/vel_iters={args.pos_iters}/{args.vel_iters} "
          f"fric_off={args.friction_offset} fric_corr={args.friction_corr} "
          f"contact/rest={args.contact_offset}/{args.rest_offset} solver={'PGS' if args.pgs else 'TGS'}")
    cfg.scene.num_envs = args.num_envs
    cfg.project_dir = project_dir
    cfg.track_field.num_envs = args.num_envs
    cfg.wall_visuals = False
    cfg.episode_length_s = 1.0e9
    cfg.observation_space = 2 * cfg.racing.obs_dim()
    from dacerpp_lab.track_field import TrackField
    cfg.scene.env_spacing = TrackField(cfg.track_field).suggested_env_spacing

    cfg.racing.v_cap_a_range = None   # 특성 측정: A 속도 핸디캡 비활성 (양쪽 동일 명령)
    cfg.tire.mu_range = None          # 특성 측정: 마찰 랜덤화 비활성 (공칭 mu 고정)
    env = OpenLoopEnv(cfg)
    rc = cfg.racing
    dev = env.device
    N = env.num_envs
    env.reset()

    def thr_for(v):   # v_cmd -> throttle 명령
        return (v - rc.v_min) / (0.5 * (rc.v_max - rc.v_min)) - 1.0

    # env i -> 목표속도 그룹 (4개 그룹)
    v_groups = [3.0, 5.0, 7.0, 9.0]
    v_tgt = torch.tensor([v_groups[i % 4] for i in range(N)], device=dev)
    thr = torch.tensor([thr_for(v) for v in v_tgt.tolist()], device=dev)

    from dacerpp_lab.car_cfg import FRONT_WHEEL_JOINTS, DRIVE_JOINTS
    fw_ids, _ = env.car_a.find_joints(FRONT_WHEEL_JOINTS)
    rw_ids, _ = env.car_a.find_joints(DRIVE_JOINTS)

    def run_phase(steer_cmd, thr_cmd, settle, measure):
        """고정 명령으로 settle+measure 스텝 주행, measure 구간 통계 반환."""
        yr_acc, v_acc, z_acc, wf_acc, wr_acc = [], [], [], [], []
        act = torch.zeros(N, 4, device=dev)
        act[:, 0] = steer_cmd
        act[:, 1] = thr_cmd
        act[:, 2] = steer_cmd   # car_b 도 동일 (표본 2배)
        act[:, 3] = thr_cmd
        for t in range(settle + measure):
            env.step(act)
            if t >= settle:
                for car in (env.car_a, env.car_b):
                    v = torch.linalg.norm(car.data.root_lin_vel_w[:, :2], dim=1)
                    yr = car.data.root_ang_vel_w[:, 2]
                    v_acc.append(v); yr_acc.append(yr)
                    z_acc.append(car.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2])
                    wf_acc.append(car.data.joint_vel[:, fw_ids].mean(dim=1))
                    wr_acc.append(car.data.joint_vel[:, rw_ids].mean(dim=1))
        return (torch.stack(v_acc), torch.stack(yr_acc), torch.stack(z_acc),
                torch.stack(wf_acc), torch.stack(wr_acc))

    print("\n===== A0. 직진(δ=0) 자유회전 전륜 스핀 검사 =====")
    v_m, yr_m, z_m, wf_m, wr_m = run_phase(0.0, thr, settle=120, measure=30)
    # 좌/우 개별 ω + 전륜 바디 높이 + 섀시 피치 (접지/역회전/3륜주행 판별)
    jv = env.car_a.data.joint_vel
    fw_names = [env.car_a.joint_names[i] for i in fw_ids]
    body_ids, body_names = env.car_a.find_bodies(["front_left_wheel", "front_right_wheel",
                                                  "rear_left_wheel", "rear_right_wheel"])
    bz = env.car_a.data.body_pos_w[:, body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    grav = env.car_a.data.projected_gravity_b
    for g, vg in enumerate(v_groups):
        mask = torch.tensor([i % 4 == g for i in range(N)], device=dev)
        v = v_m[:, mask].mean()
        per = " ".join(f"{fw_names[k]}={jv[mask][:, fw_ids[k]].mean():+7.1f}" for k in range(len(fw_ids)))
        bzm = bz[mask].mean(dim=0)
        print(f"  v_tgt={vg:.0f}: v실측={v:.2f} | ω전륜평균={wf_m[:, mask].mean():7.1f} "
              f"(롤링기대 {v/rc.wheel_radius:6.1f}) | ω후륜={wr_m[:, mask].mean():7.1f}")
        print(f"      개별: {per} | 바퀴z FL/FR/RL/RR="
              f"{bzm[0]:.4f}/{bzm[1]:.4f}/{bzm[2]:.4f}/{bzm[3]:.4f} "
              f"| grav_b=({grav[mask][:,0].mean():+.3f},{grav[mask][:,1].mean():+.3f},{grav[mask][:,2].mean():+.3f})")

    print("\n===== A. 정상상태 선회: 조향각 스윕 x 목표속도 그룹 =====")
    print(f"{'δcmd':>6} {'v_tgt':>6} {'v실측':>6} {'yaw_r':>7} {'기대yr':>7} {'추종비':>6} "
          f"{'a_lat':>6} {'a_lat/μg':>8} {'반경':>6} "
          f"{'ω앞바퀴':>7} {'ω롤링기대':>8} {'ω뒷바퀴':>7}")
    grip_ceiling = torch.zeros(4, device=dev)  # 그룹별 최대 a_lat
    for sc in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        v_m, yr_m, z_m, wf_m, wr_m = run_phase(sc, thr, settle=90, measure=60)
        for g, vg in enumerate(v_groups):
            mask = torch.tensor([i % 4 == g for i in range(N)], device=dev)
            v = v_m[:, mask].mean()
            yr = yr_m[:, mask].mean()
            wf = wf_m[:, mask].mean()
            wr = wr_m[:, mask].mean()
            delta = sc * rc.max_steering_angle
            expect = v / L_WB * torch.tan(torch.tensor(delta))
            alat = (v * yr).abs()
            grip_ceiling[g] = torch.maximum(grip_ceiling[g], alat)
            rad = (v / yr.abs().clamp(min=1e-3))
            print(f"{sc:>6.2f} {vg:>6.1f} {v:>6.2f} {yr:>7.3f} {expect:>7.3f} "
                  f"{(yr/expect).abs():>6.2f} {alat:>6.2f} {alat/MU_G:>8.2f} "
                  f"{rad:>6.2f} {wf:>7.1f} {v/rc.wheel_radius:>8.1f} {wr:>7.1f}")
    print(f"\n[A 결론] 목표속도별 실측 횡그립 한계 a_lat_max = "
          f"{[f'{v_groups[g]}m/s: {grip_ceiling[g]:.1f}' for g in range(4)]} m/s² "
          f"(이론 μg={MU_G:.1f})")

    print("\n===== B. 직선 제동: 풀스로틀 가속 -> 풀브레이크(thr=-1, δ=0) =====")
    run_phase(0.0, 1.0, settle=150, measure=1)         # 가속
    v0 = torch.linalg.norm(env.car_a.data.root_lin_vel_w[:, :2], dim=1).mean()
    dt = env.step_dt
    act = torch.zeros(N, 4, device=dev); act[:, 1] = -1.0; act[:, 3] = -1.0
    vs = []
    for t in range(90):
        env.step(act)
        vs.append(torch.linalg.norm(env.car_a.data.root_lin_vel_w[:, :2], dim=1).mean())
    vs = torch.stack(vs)
    # 최대 감속 (10-90% 구간 기울기)
    dvs = (vs[1:] - vs[:-1]) / dt
    print(f"  v0={v0:.2f} -> 1초 후 {vs[29]:.2f}, 2초 후 {vs[59]:.2f} m/s | "
          f"최대 감속 {dvs.min():.2f} m/s² (후륜 전용 이론 ≈ -5~-6)")

    print("\n===== C. 제동+조향(플라우 제동): thr=-1, δ=1.0 =====")
    run_phase(0.0, 1.0, settle=150, measure=1)
    v0 = torch.linalg.norm(env.car_a.data.root_lin_vel_w[:, :2], dim=1).mean()
    act = torch.zeros(N, 4, device=dev); act[:, 0] = 1.0; act[:, 1] = -1.0
    act[:, 2] = 1.0; act[:, 3] = -1.0
    vs = []
    for t in range(90):
        env.step(act)
        vs.append(torch.linalg.norm(env.car_a.data.root_lin_vel_w[:, :2], dim=1).mean())
    vs = torch.stack(vs)
    dvs = (vs[1:] - vs[:-1]) / dt
    print(f"  v0={v0:.2f} -> 1초 후 {vs[29]:.2f} m/s | 최대 감속 {dvs.min():.2f} m/s²")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
