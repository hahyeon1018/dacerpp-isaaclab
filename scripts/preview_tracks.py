#!/usr/bin/env python3
"""학습에 실제 쓰이는 트랙(웨이포인트 생성 + f1tenth + 실전 대회장 맵) 미리보기 PNG.

Isaac Lab 불필요 (순수 numpy + matplotlib). TrackField 를 그대로 사용하므로
학습이 보는 것과 동일한 중심선/폭(등간격 ds 리샘플 후)이 그려진다.

실행:
  python scripts/preview_tracks.py                       # 생성 트랙 16종
  python scripts/preview_tracks.py --n 24 --seed 1234    # 시드 고정(재현)
  python scripts/preview_tracks.py --kind real           # 실전 대회장 4맵
  python scripts/preview_tracks.py --kind all            # 종류별 골고루
  python scripts/preview_tracks.py --tracks hall,teras,Monza,gen3
  python scripts/preview_tracks.py --harmonic            # 구 하모닉 레이아웃(호환)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dacerpp_lab.track_field import TrackField, TrackFieldCfg
from dacerpp_lab.tracks import centerline_features

REAL_MAPS = ("hall", "hangeong", "hyeongnam", "teras")   # 실전 대회장 SLAM 맵

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=16, help="미리볼 트랙 수")
parser.add_argument("--seed", type=int, default=None,
                    help="절차 생성 시드 고정 (미지정 시 실행마다 다름)")
parser.add_argument("--kind", type=str, default="gen",
                    choices=["gen", "f1tenth", "real", "comp", "all"],
                    help="gen=웨이포인트 생성, f1tenth=실서킷, real=실측 대회장, "
                         "comp=대회 코스 변형본, all=골고루")
parser.add_argument("--tracks", type=str, default="",
                    help="특정 트랙 이름(콤마 구분). 지정 시 --kind/--n 무시")
parser.add_argument("--cols", type=int, default=0,
                    help="열 수 (0 = 트랙 수에 맞춰 자동)")
parser.add_argument("--compact", action="store_true",
                    help="한 장에 많이 보기: 축/범례 생략, 셀 축소 (수십~수백 종 일람용)")
parser.add_argument("--out", type=str, default="tracks_preview.png")
parser.add_argument("--harmonic", action="store_true",
                    help="구 하모닉 레이아웃(use_waypoint_gen=False) 미리보기")
args = parser.parse_args()

# --tracks 에 genN 이 있으면 그만큼은 생성해야 인덱싱 가능
want = [t.strip() for t in args.tracks.split(",") if t.strip()]
need_gen = max([int(t[3:]) + 1 for t in want if t.startswith("gen") and t[3:].isdigit()]
               or [0])
cfg = TrackFieldCfg(num_envs=1, procedural_seed=args.seed,
                    use_waypoint_gen=not args.harmonic,
                    num_procedural=max(args.n, need_gen, 32))
tf = TrackField(cfg)
names = tf.track_names

# ---- 표시할 트랙 선택 ----
gen_ids = [i for i, n in enumerate(names) if n.startswith(("gen", "proc"))]
real_ids = [i for i, n in enumerate(names) if n in REAL_MAPS]
comp_ids = [i for i, n in enumerate(names) if n.startswith("comp")]
f1_ids = [i for i in range(len(names))
          if i not in gen_ids and i not in real_ids and i not in comp_ids]

if want:
    lower = {n.lower(): i for i, n in enumerate(names)}
    ids = []
    for t in want:
        if t.lower() not in lower:
            sys.exit(f"[ERROR] 알 수 없는 트랙: {t!r}\n  사용 가능: {', '.join(names)}")
        ids.append(lower[t.lower()])
elif args.kind == "gen":
    ids = gen_ids[:args.n]
elif args.kind == "f1tenth":
    ids = f1_ids[:args.n]
elif args.kind == "real":
    ids = real_ids[:args.n]
elif args.kind == "comp":
    ids = comp_ids[:args.n]
else:                                    # all: 네 종류를 번갈아
    ids, pools = [], [gen_ids, f1_ids, real_ids, comp_ids]
    while len(ids) < args.n and any(pools):
        for p in pools:
            if p and len(ids) < args.n:
                ids.append(p.pop(0))

compact = args.compact or len(ids) > 40      # 많으면 자동 compact
cols = args.cols if args.cols > 0 else (
    int(np.ceil(np.sqrt(len(ids) * 1.4))) if compact else min(4, len(ids)))
cols = max(1, cols)
rows = (len(ids) + cols - 1) // cols
cell = 2.0 if compact else 4.0
fig, axs = plt.subplots(rows, cols, figsize=(cell * cols, cell * 1.05 * rows))
axs = np.atleast_2d(axs)

for k, g in enumerate(ids):
    cl, hw = tf.centerlines[g], tf.half_widths[g]
    f = centerline_features(cl)
    psi, kap = f["psi"], np.abs(f["kappa"])
    nrm = np.stack([-np.sin(psi), np.cos(psi)], axis=1)
    left, right = cl + nrm * hw[:, None], cl - nrm * hw[:, None]
    ax = axs[k // cols][k % cols]
    lw = 0.6 if compact else 1.0
    ax.plot(cl[:, 0], cl[:, 1], "--", color="0.6", lw=0.4 if compact else 0.6)
    ax.plot(left[:, 0], left[:, 1], "-b", lw=lw)
    ax.plot(right[:, 0], right[:, 1], "-r", lw=lw)
    # 좁은 구간(게이트/병목)과 급코너를 강조 — 실전 맵의 핵심 특징
    narrow = hw < 0.6
    tight = kap > 0.6                    # 회전반경 < 1.67m
    if narrow.any():
        ax.plot(cl[narrow, 0], cl[narrow, 1], ".", color="darkorange",
                ms=1.2 if compact else 2.5,
                label=f"게이트 hw<0.6 ({narrow.mean():.0%})")
    if tight.any():
        ax.plot(cl[tight, 0], cl[tight, 1], ".", color="purple",
                ms=1.0 if compact else 2.0,
                label=f"급코너 R<1.67 ({tight.mean():.0%})")
    ax.plot(cl[0, 0], cl[0, 1], "o", color="green", ms=3 if compact else 6)   # 시작점
    if not compact:
        d = cl[3] - cl[0]
        ax.arrow(cl[0, 0], cl[0, 1], d[0], d[1], head_width=0.35,
                 color="green", length_includes_head=True)      # 진행 방향
    ax.set_aspect("equal")
    if compact:
        ax.set_title(f"{names[g]} R{1/max(kap.max(), 1e-6):.2f}", fontsize=5.5, pad=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
    else:
        ax.set_title(f"{names[g]}  |  {f['total_s']:.0f}m  "
                     f"폭 {2*hw.min():.1f}~{2*hw.max():.1f}m  "
                     f"Rmin {1/max(kap.max(), 1e-6):.2f}m", fontsize=9)
        ax.grid(True, alpha=0.3)
        if narrow.any() or tight.any():
            ax.legend(fontsize=6, loc="lower right", framealpha=0.8)

for k in range(len(ids), rows * cols):
    axs[k // cols][k % cols].axis("off")

out_dir = os.path.join(os.path.dirname(__file__), "..", "generated")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, args.out)
plt.tight_layout()
plt.savefig(out, dpi=90)

# ---- 콘솔 요약 ----
L = [tf.features[g]["total_s"] for g in ids]
K = [np.abs(tf.features[g]["kappa"]).max() for g in ids]
HW = [(tf.half_widths[g].min(), tf.half_widths[g].max()) for g in ids]
print(f"[preview] {len(ids)}종 ({args.kind if not want else 'custom'}) "
      f"| 길이 {min(L):.0f}~{max(L):.0f}m | Rmin {1/max(K):.2f}~{1/min(K):.2f}m "
      f"| 반폭 {min(h[0] for h in HW):.2f}~{max(h[1] for h in HW):.2f}m")
if cfg.procedural_seed is not None:
    print(f"[preview] 절차 생성 시드 = {cfg.procedural_seed} "
          f"(--seed {cfg.procedural_seed} 로 재현 가능)")
print("saved:", os.path.abspath(out))
