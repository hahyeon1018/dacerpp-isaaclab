#!/usr/bin/env python3
"""f1tenth.usd 의 링크별 질량을 f1tenth.urdf 값과 일치시킨다.

왜 필요한가:
  해석 타이어 모델(env_cfg.TireModelCfg)은 수직하중 Nf/Nr 을 TireModelCfg.mass 로
  계산하고, 실제 운동 적분은 PhysX 가 USD 의 질량으로 한다. 이 둘이 어긋나면
  "4.24kg 만큼의 그립으로 3.94kg 를 가속"하는 비물리가 되어, 가속·제동·하중이동이
  전부 틀어진다. URDF 합계 = TireModelCfg.mass 여야 하고, USD 는 URDF 를 따라야 한다.

왜 URDF -> USD 전체 재변환이 아니라 이 패치인가:
  현재 USD 에는 이미 patch_usd_tire.py / patch_usd_wheel_capsule.py 의 패치가
  적용돼 있다. 전체 재변환하면 그게 전부 날아가 두 스크립트를 다시 돌려야 한다.
  질량만 바뀐 경우에는 이 스크립트로 국소 수정하는 편이 안전하다.
  (형상/조인트가 바뀌었다면 재변환 + 세 패치 스크립트 순차 실행이 맞다:
     convert_urdf.py -> patch_usd_tire.py -> patch_usd_wheel_capsule.py
     -> patch_usd_mass.py)

관성 텐서는 건드리지 않는다:
  URDF 쪽에서도 갱신하지 않았다 — 추가 질량의 공간 분포를 모르면 계산할 수 없다.
  질량 대비 관성이 덜 늘어 요 응답이 실제보다 약간 빠를 수 있다는 점은 감수한다.
  분포를 알게 되면 URDF 의 ixx/iyy/izz 를 고치고 --inertia 로 함께 반영할 것.

실행:
  python scripts/patch_usd_mass.py              # URDF 값으로 동기화
  python scripts/patch_usd_mass.py --check      # 비교만 하고 수정하지 않음
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD_PATH = os.path.join(ROOT, "assets", "f1tenth", "f1tenth.usd")
URDF_PATH = os.path.join(ROOT, "assets", "f1tenth", "f1tenth.urdf")
PRIM_ROOT = "/f1tenth"
TOL = 1e-6


def _pxr_search_roots() -> list[str]:
    """omni.usd.libs(=pxr) 가 있을 만한 곳. 설치 형태가 두 가지라 둘 다 본다."""
    roots = []
    if os.environ.get("ISAACSIM_PATH"):
        roots.append(os.environ["ISAACSIM_PATH"])
    # (1) conda env 안에 pip 설치된 isaacsim 패키지
    site = os.path.join(sys.prefix, "lib",
                        f"python{sys.version_info.major}.{sys.version_info.minor}",
                        "site-packages", "isaacsim")
    roots.append(site)
    # (2) 독립 설치본 (예: <repo>/../isaacsim)
    roots.append(os.path.join(os.path.dirname(ROOT), "isaacsim"))
    return roots


def _reexec_with_pxr_env():
    """pxr(USD 코어)는 isaacsim extscache 에 있다 — 경로/링커 설정 후 재실행."""
    ext = None
    for root in _pxr_search_roots():
        hits = sorted(glob.glob(os.path.join(root, "extscache", "omni.usd.libs-*")))
        if hits:
            ext = hits[0]
            break
    if ext is None:
        sys.exit("[ERROR] omni.usd.libs(=pxr) 를 찾지 못했습니다.\n"
                 "  env_isaacsim 계열 python 으로 실행하거나, 독립 설치본 경로를\n"
                 "  ISAACSIM_PATH=/path/to/isaacsim 로 지정하세요.\n"
                 f"  탐색한 곳: {_pxr_search_roots()}")
    env = os.environ.copy()
    env["PYTHONPATH"] = ext + os.pathsep + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = (os.path.join(ext, "bin") + os.pathsep
                              + os.path.join(sys.prefix, "lib") + os.pathsep
                              + env.get("LD_LIBRARY_PATH", ""))
    env["_PXR_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


try:
    from pxr import Usd, UsdPhysics
except ImportError:
    if os.environ.get("_PXR_REEXEC"):
        raise
    _reexec_with_pxr_env()


def urdf_masses(path: str) -> dict[str, float]:
    """URDF 링크별 질량. 이게 단일 진실 원천(single source of truth)이다."""
    tree = ET.parse(path)
    out = {}
    for link in tree.getroot().findall("link"):
        m = link.find("inertial/mass")
        if m is not None:
            out[link.get("name")] = float(m.get("value"))
    if not out:
        sys.exit(f"[ERROR] URDF 에서 질량을 읽지 못했습니다: {path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="비교만 하고 수정하지 않는다")
    ap.add_argument("--usd", default=USD_PATH)
    ap.add_argument("--urdf", default=URDF_PATH)
    args = ap.parse_args()

    want = urdf_masses(args.urdf)
    print(f"[URDF] {args.urdf}")
    print(f"       링크 {len(want)}개, 합계 {sum(want.values()):.3f} kg")

    stage = Usd.Stage.Open(args.usd)
    if stage is None:
        sys.exit(f"[ERROR] USD 를 열 수 없습니다: {args.usd}")

    plan, missing = [], []
    for name, m_want in sorted(want.items()):
        prim = stage.GetPrimAtPath(f"{PRIM_ROOT}/{name}")
        if not prim or not prim.IsValid():
            missing.append(name)
            continue
        attr = UsdPhysics.MassAPI(prim).GetMassAttr() if prim.HasAPI(UsdPhysics.MassAPI) else None
        cur = float(attr.Get()) if (attr and attr.HasAuthoredValue()) else None
        plan.append((name, prim, cur, m_want))

    if missing:
        print(f"[경고] USD 에 없는 링크(패치 생략): {missing}")

    print(f"\n{'링크':<24}{'USD 현재':>12}{'URDF 목표':>12}   상태")
    n_change = 0
    for name, _, cur, m_want in plan:
        same = cur is not None and abs(cur - m_want) < TOL
        n_change += 0 if same else 1
        cur_s = "<미기입>" if cur is None else f"{cur:.3f}"
        print(f"{name:<24}{cur_s:>12}{m_want:>12.3f}   {'일치' if same else '★수정'}")
    usd_total = sum((c if c is not None else 0.0) for _, _, c, _ in plan)
    print(f"{'합계':<24}{usd_total:>12.3f}{sum(w for *_, w in plan):>12.3f}")

    if n_change == 0:
        print("\n[OK] 이미 일치합니다. 수정할 것이 없습니다.")
        return 0
    if args.check:
        print(f"\n[--check] {n_change}개 불일치. 수정하려면 --check 없이 실행하세요.")
        return 1

    bak = f"{args.usd}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
    shutil.copy2(args.usd, bak)
    print(f"\n[백업] {bak}")

    for name, prim, cur, m_want in plan:
        if cur is not None and abs(cur - m_want) < TOL:
            continue
        api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) \
            else UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr().Set(float(m_want))
        print(f"  {name}: {cur} -> {m_want}")

    stage.GetRootLayer().Save()
    print(f"[저장] {args.usd}")

    # 저장 결과를 다시 읽어 검증 (쓰기가 실제로 반영됐는지)
    verify = Usd.Stage.Open(args.usd)
    tot = 0.0
    for name in want:
        p = verify.GetPrimAtPath(f"{PRIM_ROOT}/{name}")
        if p and p.IsValid() and p.HasAPI(UsdPhysics.MassAPI):
            a = UsdPhysics.MassAPI(p).GetMassAttr()
            if a and a.HasAuthoredValue():
                tot += float(a.Get())
    ok = abs(tot - sum(want.values())) < 1e-4
    print(f"[검증] USD 질량 합계 {tot:.3f} kg vs URDF {sum(want.values()):.3f} kg -> "
          f"{'일치' if ok else '불일치!'}")
    if not ok:
        return 1
    print("\n★ env_cfg.py TireModelCfg.mass 도 이 합계와 같은지 확인할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
