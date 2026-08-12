#!/usr/bin/env python3
"""학습에 들어가는 맵 중 '가장 급한 V자' 상위 N개를 시각화 (로드 후 기준).

로드 후(process_measured_centerline) |κ|max 로 정렬해 가장 급한 코스를 뽑고,
좌/우 벽 + 급코너(보라)·좁은 게이트(주황)를 표시한다. 연습 맵 3개는 항상 포함.

실행:
  python scripts/preview_sharp_v.py                 # 상위 9 + 연습맵 3
  python scripts/preview_sharp_v.py --topn 12
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

parser = argparse.ArgumentParser()
parser.add_argument("--topn", type=int, default=9, help="가장 급한 상위 N개")
parser.add_argument("--num_procedural", type=int, default=285)
parser.add_argument("--out", type=str, default="sharp_v_preview.png")
args = parser.parse_args()

cfg = TrackFieldCfg(num_envs=1, num_procedural=args.num_procedural, procedural_seed=1)
tf = TrackField(cfg)
names = tf.track_names

# 각 트랙 로드 후 |κ|max (등간격 리샘플된 저장본 기준)
kmax = np.array([np.abs(centerline_features(cl)["kappa"]).max() for cl in tf.centerlines])

# 연습 맵 3개 인덱스 (항상 포함)
lobby = [i for i, n in enumerate(names) if "lobby" in n]
# 급한 순 정렬 (연습맵 제외한 상위 N)
order = [i for i in np.argsort(-kmax) if i not in lobby][:args.topn]
ids = lobby + order        # 연습맵 먼저, 그다음 급한 순
# 최종적으로 급한 순으로 재정렬해 보기 좋게
ids = sorted(ids, key=lambda i: -kmax[i])

cols = min(4, len(ids))
rows = (len(ids) + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
axs = np.atleast_2d(axs)

for k, g in enumerate(ids):
    cl, hw = tf.centerlines[g], tf.half_widths[g]
    f = centerline_features(cl)
    psi, kap = f["psi"], np.abs(f["kappa"])
    nrm = np.stack([-np.sin(psi), np.cos(psi)], axis=1)
    left, right = cl + nrm * hw[:, None], cl - nrm * hw[:, None]
    ax = axs[k // cols][k % cols]
    ax.plot(cl[:, 0], cl[:, 1], "--", color="0.7", lw=0.5)
    ax.plot(left[:, 0], left[:, 1], "-b", lw=1.0)
    ax.plot(right[:, 0], right[:, 1], "-r", lw=1.0)
    # 급코너(V apex): |κ|>1.4 강조 (R<0.71m)
    apex = kap > 1.4
    if apex.any():
        ax.plot(cl[apex, 0], cl[apex, 1], ".", color="purple", ms=4,
                label="sharp |k|>1.4")
    narrow = hw < 0.55
    if narrow.any():
        ax.plot(cl[narrow, 0], cl[narrow, 1], ".", color="darkorange", ms=2.5,
                label="narrow hw<0.55")
    ax.plot(cl[0, 0], cl[0, 1], "o", color="green", ms=6)
    is_lobby = "lobby" in names[g]
    ttl = f"{'* ' if is_lobby else ''}{names[g]}\n"
    ttl += f"|κ|max={kap.max():.2f} (Rmin {1/kap.max():.2f}m)  " \
           f"w {2*hw.min():.2f}~{2*hw.max():.2f}m"
    ax.set_title(ttl, fontsize=9, color="darkgreen" if is_lobby else "black")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if apex.any() or narrow.any():
        ax.legend(fontsize=6, loc="lower right", framealpha=0.8)

for k in range(len(ids), rows * cols):
    axs[k // cols][k % cols].axis("off")

out_dir = os.path.join(os.path.dirname(__file__), "..", "generated")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, args.out)
plt.suptitle("Sharpest V-corners in training maps (asterisk = real practice maps, post-load)", fontsize=12)
plt.tight_layout()
plt.savefig(out, dpi=100)
print(f"[급한V] 상위 {len(ids)}개 (연습맵 {len(lobby)} 포함)")
for g in ids:
    print(f"  {'*' if 'lobby' in names[g] else ' '} {names[g]:18s} "
          f"|κ|max={kmax[g]:.2f}  Rmin={1/kmax[g]:.2f}m")
print("saved:", os.path.abspath(out))
