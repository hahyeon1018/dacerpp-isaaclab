"""1/10 스케일 F1 차량 ArticulationCfg 템플릿.

★ 사용자가 제공해야 하는 것:
  - 차량 USD 파일 (URDF -> USD 변환). 아래 USD_PATH 를 실제 경로로 교체.
  - 아래 JOINT 이름 패턴을, 당신 USD 의 실제 조인트 이름에 맞게 수정.
    (조향: 앞바퀴 2개 revolute, 구동: 뒷바퀴 또는 전체 4륜 continuous)

F1TENTH 계열 1/10 차량 기준의 보수적 기본값을 넣어두었다.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# ---------------------------------------------------------------------------
# TODO: 실제 차량 USD 경로로 교체 (예: 변환 결과물)
USD_PATH = "{PROJECT}/assets/f1tenth/f1tenth.usd"

# TODO: USD 내부 조인트 이름에 맞게 정규식 수정
STEERING_JOINTS = ["steering_joint_left", "steering_joint_right"]   # 앞바퀴 조향
DRIVE_JOINTS = ["wheel_joint_rear_left", "wheel_joint_rear_right"]  # 뒷바퀴 구동
FRONT_WHEEL_JOINTS = ["wheel_joint_front_left", "wheel_joint_front_right"]  # 앞바퀴 자유회전
ALL_WHEEL_JOINTS = DRIVE_JOINTS + FRONT_WHEEL_JOINTS

WHEEL_RADIUS = 0.06    # m (실측 2026-07-16) — action(speed)->wheel ang.vel 변환에 사용
MAX_STEERING_ANGLE = 0.42    # rad (기계 한계 0.44, 과열 방지 명령 상한 0.42 — 실측)
# ---------------------------------------------------------------------------


def make_car_cfg(prim_path: str, project_dir: str) -> ArticulationCfg:
    usd = USD_PATH.replace("{PROJECT}", project_dir)
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                # v_max(10m/s)+여유. 과도한 상한은 탄도비행 exploit 을 허용한다.
                max_linear_velocity=12.0,
                # ★ 이 클램프는 '모든 강체'(바퀴 포함)에 적용된다. 바퀴는 v/r =
                # 10/0.05 = 200 rad/s 로 돌아야 하므로 20 이면 바퀴 표면속도가
                # 1 m/s 로 제한 -> 1 m/s 이상에서 네 바퀴가 영구 종방향 슬립 ->
                # 마찰이 전부 종방향으로 소모되어 횡그립 붕괴(조향 불능).
                # (개방루프 실측: 5m/s 에서 a_lat 0.6 m/s² = μg 의 5%.)
                # 섀시 회전 exploit 은 전복 termination 이 차단하므로 여유 있게.
                max_angular_velocity=250.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.06),   # = 바퀴 반경 (실측 0.06)
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            # 조향: 위치 제어(stiffness 높게).
            # 서보 속도 실측(2026-07-17 확정, 240fps 슬로모 — 갤러리 메타데이터):
            # 무부하 풀락 스윕 0.88rad ≈ 10.5프레임 = 43.8ms -> ≈20 rad/s.
            #
            # ★★2026-08-18 시뮬 계단응답 측정 결과 — velocity_limit_sim 은 무효다.
            #   ('차량특성_조향지연_측정결과.txt' PART 2 측정 1, measure_steer_delay.py)
            #   조향 응답이 계단 진폭(0.42 vs 0.21rad)과 차속(2/5/8 m/s)에 무관하게
            #   t63 = 100.0ms 로 소수점까지 동일했다. 슬루 제한이면 절반 진폭은 절반
            #   시간이어야 하므로, velocity_limit_sim=20 에는 한 번도 도달하지 않는다.
            #   effort_limit_sim=10 도 마찬가지(걸렸다면 큰 진폭이 더 걸려야 한다).
            #   실제 조향 대역을 정하는 것은 오직 PD 지배극이다:
            #       tau = damping/stiffness = 8/80 = 0.100s  (측정 t63 100.0ms 와 일치,
            #                                                 t90/t63 = 2.25 ≈ 1차지연 2.30)
            #   즉 위 '20 rad/s' 실측은 코드에 적혀 있을 뿐 반영되어 있지 않다.
            #   속도 무관인 이유: 해석 타이어가 횡력을 섀시에 외력으로 인가하고 지면
            #   마찰이 0 이라 조향축에 정렬토크가 아예 걸리지 않는다(설계상 불가피).
            #
            # ★ 그래도 damping 을 낮추지 않은 이유 (2026-08-18 판단):
            #   측정 문서는 tau 를 25ms(damping 2.0)로 낮출 것을 제안하지만, 정책이
            #   실제로 겪는 양은 '명령->조향각'이 아니라 '명령->요레이트'다. 그 값을
            #   양쪽에서 재면 차이가 문서가 말하는 2~5배가 아니다:
            #       시뮬(학습 구성)  명령->요레이트 t63 = 100ms  (측정 문서 PART 2 측정 2)
            #       실차            명령->요레이트 상호상관 지연 = 135ms
            #                       (loc_debug_0817_2326, 상관 +0.917. r = v*tan(delta)/L
            #                        예측과 IMU 요레이트의 최적 시간시프트)
            #   즉 총 루프로 보면 시뮬이 오히려 실차보다 빠르다. 조향축만 빠르게 하면
            #   그 격차가 더 벌어진다. 또 문서 3.4 주의 2 대로 조향을 빠르게 하면 PWM
            #   뱅뱅 조향이 물리적으로 쉬워져 k_smooth_steer(0.08) 재조정이 따라와야 한다.
            #   -> 바꾸려면 먼저 '같은 추정기'로 양쪽을 재라: 시뮬 롤아웃에서도
            #      상호상관 지연을 뽑아 실차 135ms 와 직접 비교할 것. 계단응답 t63 과
            #      상호상관 지연은 서로 다른 양이라 그대로 비교하면 안 된다.
            "steering": ImplicitActuatorCfg(
                joint_names_expr=STEERING_JOINTS,
                # 아래 두 리밋은 현재 구성에서 도달하지 않는다(위 측정). 안전장치로 남긴다.
                effort_limit_sim=10.0, velocity_limit_sim=20.0,
                stiffness=80.0, damping=8.0,   # tau = damping/stiffness = 100ms
            ),
            # 구동: 속도 제어(stiffness 0, damping 으로 속도 추종).
            # ★ 해석 타이어 전환 후 추진은 외력(TireModelCfg)이 담당하고 이 드라이브는
            # 바퀴 회전 '비주얼' 전용이다(지면 μ=0). 토크를 비주얼 수준으로 낮춰
            # 조인트 반력(휠리 모멘트, 정적 여유 5.5 N·m)에 의한 발진 전복을 방지.
            "drive": ImplicitActuatorCfg(
                joint_names_expr=DRIVE_JOINTS,
                # velocity_limit_sim >= v_max/wheel_radius (10/0.05=200) + 여유
                effort_limit_sim=0.3, velocity_limit_sim=250.0,
                stiffness=0.0, damping=0.5,
            ),
            # 앞바퀴: 비주얼 회전용 미세 속도드라이브.
            # ★ 액추에이터를 반드시 명시해야 한다 — 미설정 조인트는 USD 변환 시 구워진
            # position drive(stiffness 100 N·m/rad, target 0, maxForce 무제한)가 그대로
            # 살아남아 바퀴를 잠근다. 지면 마찰이 0(해석 타이어)이라 앞바퀴는 스스로
            # 구르지 않으므로, 작은 damping 으로 현재 차속에 맞춰 돌려준다(물리 영향 무시).
            "front_wheels": ImplicitActuatorCfg(
                joint_names_expr=FRONT_WHEEL_JOINTS,
                effort_limit_sim=0.1, velocity_limit_sim=250.0,
                stiffness=0.0, damping=0.05,
            ),
        },
    )
