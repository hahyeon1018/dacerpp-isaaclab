# DACER++ × Isaac Lab — 다중 트랙 병렬 레이싱 학습

기존 ROS2/F1TENTH 단일 환경용 **DACER++**(Distributional Diffusion Actor-Critic, 리스크 민감)
알고리즘을 **NVIDIA Isaac Lab 2.x** 의 대규모 병렬 환경으로 이식한 프로젝트입니다.

- **2048개 환경 × 8종 절차적 스플라인 서킷** 을 동시에 학습(순차 X → 일반화 정책).
- **각 환경에 2대 경쟁 주행**: `car_a = CVaR(보수적)`, `car_b = Pow(공격적)`.
  - 2048대 `car_a` 는 **하나의 CVaR 파라미터** 를 공유 학습.
  - 2048대 `car_b` 는 **하나의 Pow 파라미터** 를 공유 학습.
- 물리는 **평면**에서 진행하고, 트랙은 **중심선+폭으로 해석적 정의**(코스 이탈 = 보상/종료).
  거대한 충돌 메시(2048개 트랙 = 수백만 면)를 피해 가볍고 빠릅니다.

---

## 프로젝트 구조

```
dacerpp_isaaclab/
├── pyproject.toml              # editable 설치 (pip install -e .)
├── README.md
├── dacer_pp/                   # ── DACER++ 알고리즘 코어 (이식, GPU 배치화) ──
│   ├── config.py               #    DACERppConfig/RiskConfig + CVaR/Pow 프리셋
│   ├── agent.py                #    DACERpp (act_batch 추가: torch 네이티브)
│   ├── networks.py             #    IQN QVN(psi/phi/f) + Diffusion 정책망
│   ├── diffusion.py            #    DDPM 역확산 샘플링
│   ├── losses.py               #    quantile Huber loss
│   ├── risk.py                 #    distortion 측도(neutral/cvar/pow)
│   ├── entropy.py              #    GMM 엔트로피 추정(온도 alpha 조절)
│   ├── replay_buffer.py        #    ★ GPU 텐서 버퍼 + add_batch (다중환경용 재작성)
│   └── utils.py
├── dacerpp_lab/                # ── Isaac Lab 연동 계층 (신규) ──
│   ├── tracks.py               #    절차적 스플라인 중심선/메시 생성 (검증됨)
│   ├── vectorized_track.py     #    torch 벡터화 투영/진행도 (전 환경 동시, 검증됨)
│   ├── track_field.py          #    2048환경 트랙 배정/시작포즈
│   ├── car_cfg.py              #    ★ 1/10 차량 ArticulationCfg (USD 필요: 사용자 제공)
│   ├── env_cfg.py              #    RacingCfg + RacingEnvCfg
│   └── racing_env.py           #    ★ 2대 경쟁 DirectRLEnv
├── scripts/
│   ├── train.py                #    듀얼 에이전트 동시 학습 루프
│   └── preview_tracks.py       #    8종 트랙 미리보기 PNG (Isaac Lab 불필요)
├── assets/                     #    여기에 차량 USD 배치 (assets/f1tenth/f1tenth.usd)
└── generated/                  #    트랙 미리보기 등 산출물
```

---

## 1. 개발 환경 세팅

이미 `env_isaaclab`(conda, Python 3.11, Isaac Lab 2.x)이 구성되어 있다고 가정합니다.

```bash
conda activate env_isaaclab

# 프로젝트를 editable 설치 (의존: numpy, scipy, trimesh, scikit-learn)
cd /path/to/dacerpp_isaaclab
pip install -e .
```

torch / isaaclab 은 `env_isaaclab` 에 이미 설치돼 있으므로 건드리지 않습니다.

트랙 생성만 먼저 눈으로 확인하려면(시뮬레이터 불필요):

```bash
python scripts/preview_tracks.py    # generated/tracks_preview.png
```

## 2. 차량 USD 준비 (필수)

`car_cfg.py` 상단의 `USD_PATH` / 조인트 이름을 **당신의 1/10 차량**에 맞게 수정합니다.

```python
USD_PATH = "{PROJECT}/assets/f1tenth/f1tenth.usd"
STEERING_JOINTS = ["steering_joint_left", "steering_joint_right"]
DRIVE_JOINTS    = ["wheel_joint_rear_left", "wheel_joint_rear_right"]
WHEEL_RADIUS    = 0.05
```

URDF만 있다면 Isaac Lab 변환기로 USD 생성:

```bash
python /home/sscc/Desktop/hahyeon/IsaacLab/scripts/tools/convert_urdf.py \
    /path/to/f1tenth.urdf  \
    /path/to/dacerpp_isaaclab/assets/f1tenth/f1tenth.usd \
    --merge-joints --make-instanceable
```

## 3. 학습 실행

```bash
conda activate env_isaaclab
cd /path/to/dacerpp_isaaclab

python scripts/train.py --num_envs 2048 --headless \
    --cvar_eta 0.5 --pow_eta 1.5 --batch_size 1024
```

- `./isaaclab.sh -p scripts/train.py ...` 도 동일하게 동작합니다(편의 래퍼).
- 체크포인트는 `~/dacerpp_runs/{cvar.pt, pow.pt}` 에 저장됩니다.
- GPU 메모리에 따라 `--num_envs` 를 조절하세요(우선 256~512로 검증 후 증량 권장).

---

## 학습 방식 핵심

- **동시 학습**: 8종 서킷을 2048환경에 펼쳐 한꺼번에 학습 → 트랙 기하에 대한 도메인 랜덤화.
  정책은 서킷 ID 없이 **센서 관측만으로** 주행하므로 새 서킷에 일반화됩니다.
- **두 리스크 스타일**: 동일 보상에 대해 CVaR(하위 분위수 집중→보수) vs Pow(상위 낙관→공격)
  로 주행 성향이 갈립니다. 보상은 같게 두어 비교가 깨끗합니다(`k_overtake=0` 기본).
- **에피소드**: 트랙 단위. 두 차량 중 하나라도 이탈/충돌하거나 시간초과면 해당 트랙 리셋.
  단, **차량별 terminated/truncated 를 따로** 산출해 각 에이전트가 자기 done 으로 학습합니다.

---

## ⚠️ 검증 상태 (정직한 고지)

- **검증 완료(시뮬레이터 불필요 부분)**:
  - `dacer_pp/*` 알고리즘 코어 — CPU 스모크 테스트로 act_batch + update 동작 확인.
  - `tracks.py`, `vectorized_track.py`, `track_field.py` — 트랙 8종 생성/투영/배정 정확성 확인.
- **미검증(실제 Isaac Lab + 차량 USD 필요)**:
  - `racing_env.py` 의 시뮬레이터 연동(조인트 제어/루트상태 쓰기)은 Isaac Lab 2.x 관례를
    따르지만 **차량 USD 와 패치 버전에 맞춘 디버그 패스가 한 번 필요**합니다.

이 저장소는 **잘 구조화된 출발점**이지, 바로 학습이 수렴하는 turnkey 시스템이 아닙니다.

---

## 사용자가 추가로 제공/조정해야 하는 것

1. **차량 USD** (1/10 F1TENTH류). `assets/f1tenth/f1tenth.usd` 배치 + `car_cfg.py` 경로 수정.
2. **조인트 이름/휠 반경** 을 USD 실제 값에 맞게 `car_cfg.py` 수정.
3. **`racing_env.py` 1회 디버그**: `write_root_state_to_sim`, `find_joints`,
   `set_joint_*_target` 시그니처가 패치별로 다를 수 있어 첫 실행 시 점검 필요.
4. **(선택) 진짜 LiDAR**: 현재 `_synthetic_scan` 은 해석적 근사 프록시입니다.
   Isaac Lab `RayCaster` 센서로 교체하면 더 사실적입니다(드롭인 가능).
5. **보상 가중치/`env_spacing`/`num_envs` 튜닝**: `TrackField.suggested_env_spacing`(~40m)
   를 기준으로, GPU 메모리에 맞춰 환경 수를 조절하세요.
6. **(선택) 난이도 커리큘럼**: 우선 `difficulty` 고정으로 학습이 도는지 검증한 뒤,
   성공률에 따라 트랙 난이도를 올리는 커리큘럼을 도입하세요(오프폴리시 버퍼와 분포 시프트 주의).
