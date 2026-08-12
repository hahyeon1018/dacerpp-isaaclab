#!/usr/bin/env python3
"""f1tenth.usd 에 타이어 고무 물리 재질을 패치한다.

URDF 는 물리 재질(마찰)을 표현할 수 없으므로, URDF -> USD 재변환
(IsaacLab scripts/tools/convert_urdf.py) 을 할 때마다 이 스크립트를
다시 실행해야 한다. 순서:

  1) python -u ~/Desktop/hahyeon/IsaacLab/scripts/tools/convert_urdf.py \
       assets/f1tenth/f1tenth.urdf assets/f1tenth/f1tenth.usd --headless
  2) python scripts/patch_usd_tire.py

패치 내용:
  - /f1tenth/PhysicsMaterials/tire_rubber 재질 생성
    (카펫 위 RC 고무타이어: 정지 1.4 / 운동 1.25 / 반발 0)
  - 바퀴 4개 body Xform 에 physics purpose 로 바인딩.
    (collisions prim 은 instanceable 이라 직접 편집 불가 -> body 에
     strongerThanDescendants 로 바인딩해 하위로 상속시킨다)
  - 지면(카펫) 쪽 재질은 env_cfg.py 의 TerrainImporterCfg.physics_material
    (1.3/1.15) 이 담당. average 결합 -> 유효 μ ≈ 1.35/1.2.

실행: conda 의 env_isaacsim python 으로 실행하면 pxr 경로를 자동 설정해
재실행하므로 그냥 `python scripts/patch_usd_tire.py` 로 충분하다.
"""
from __future__ import annotations

import glob
import os
import sys

USD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "f1tenth", "f1tenth.usd")
WHEELS = ["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"]
STATIC_FRICTION = 1.4
DYNAMIC_FRICTION = 1.25


def _reexec_with_pxr_env():
    """pxr(USD 코어)는 isaacsim extscache 에 있다 — 경로/링커 설정 후 재실행."""
    site = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                        "site-packages")
    hits = glob.glob(os.path.join(site, "isaacsim", "extscache", "omni.usd.libs-*"))
    if not hits:
        sys.exit("[ERROR] isaacsim extscache 에서 omni.usd.libs 를 찾지 못했습니다. "
                 "env_isaacsim 환경의 python 으로 실행하세요.")
    ext = hits[0]
    env = os.environ.copy()
    env["PYTHONPATH"] = ext + os.pathsep + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = (os.path.join(ext, "bin") + os.pathsep
                              + os.path.join(sys.prefix, "lib") + os.pathsep
                              + env.get("LD_LIBRARY_PATH", ""))
    env["_PXR_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


try:
    from pxr import Usd, UsdShade, UsdPhysics, Sdf
except ImportError:
    if os.environ.get("_PXR_REEXEC"):
        raise
    _reexec_with_pxr_env()


def main():
    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        sys.exit(f"[ERROR] USD 를 열 수 없습니다: {USD_PATH}")

    mat = UsdShade.Material.Define(stage, Sdf.Path("/f1tenth/PhysicsMaterials/tire_rubber"))
    pmat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pmat.CreateStaticFrictionAttr().Set(STATIC_FRICTION)
    pmat.CreateDynamicFrictionAttr().Set(DYNAMIC_FRICTION)
    pmat.CreateRestitutionAttr().Set(0.0)

    for name in WHEELS:
        prim = stage.GetPrimAtPath(f"/f1tenth/{name}")
        if not prim.IsValid():
            sys.exit(f"[ERROR] 바퀴 prim 없음: /f1tenth/{name} (USD 구조가 바뀌었나?)")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics")
    stage.GetRootLayer().Save()

    # 재검증: 새로 열어 바인딩 확인
    stage2 = Usd.Stage.Open(USD_PATH)
    api = UsdPhysics.MaterialAPI(stage2.GetPrimAtPath("/f1tenth/PhysicsMaterials/tire_rubber"))
    print(f"[OK] tire_rubber: static={api.GetStaticFrictionAttr().Get()}, "
          f"dynamic={api.GetDynamicFrictionAttr().Get()}")
    for name in WHEELS:
        b = UsdShade.MaterialBindingAPI(stage2.GetPrimAtPath(f"/f1tenth/{name}"))
        target = b.GetDirectBinding(materialPurpose="physics").GetMaterialPath()
        assert str(target) == "/f1tenth/PhysicsMaterials/tire_rubber", (name, target)
        print(f"[OK] {name} -> {target}")
    print(f"[DONE] patched: {USD_PATH}")


if __name__ == "__main__":
    main()
