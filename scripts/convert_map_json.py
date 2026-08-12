#!/usr/bin/env python3
"""race_stack global_waypoints.json -> f1tenth 표준 *_centerline.csv 변환.

실대회장 SLAM 맵(forzaETH race_stack 포맷)의 global_waypoints.json 에는
중심선 외에 레이싱라인(iqp/sp)·트랙바운드 등 여러 웨이포인트가 들어 있다.
RL 학습에는 'centerline_waypoints.wpnts' 의 (x_m, y_m, d_right, d_left)만
사용한다 — d_left/d_right 는 중심선에서 좌/우 벽까지 거리로, f1tenth CSV 의
w_tr_left/right 와 동일 의미(track_field 로더가 자동 인식·리샘플·평활).

사용법:
  1) 맵 폴더(<name>/global_waypoints.json)를 f1tenth_racetracks/ 아래로 이동
  2) python scripts/convert_map_json.py
  -> f1tenth_racetracks/<name>/<name>_centerline.csv 생성(있으면 갱신),
     이후 학습/평가가 자동으로 트랙 목록에 포함한다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default=os.path.join(ROOT, "f1tenth_racetracks"),
                    help="global_waypoints.json 을 탐색할 루트 (하위 1단계 폴더)")
args = parser.parse_args()

found = sorted(glob.glob(os.path.join(args.dir, "*", "global_waypoints.json")))
if not found:
    raise SystemExit(f"[ERROR] {args.dir} 아래에 global_waypoints.json 이 없습니다.")

for jp in found:
    name = os.path.basename(os.path.dirname(jp))
    out = os.path.join(os.path.dirname(jp), f"{name}_centerline.csv")
    wpnts = json.load(open(jp))["centerline_waypoints"]["wpnts"]
    with open(out, "w") as f:
        f.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")
        for w in wpnts:
            f.write(f"{w['x_m']:.6f}, {w['y_m']:.6f}, {w['d_right']:.6f}, {w['d_left']:.6f}\n")
    print(f"[OK] {name}: {len(wpnts)}점 -> {out}")
