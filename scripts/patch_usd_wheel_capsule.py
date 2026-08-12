#!/usr/bin/env python3
"""바퀴 '충돌' 지오메트리를 Cylinder -> Capsule 로 패치한다.

왜: PhysX GPU 파이프라인은 해석적 실린더를 지원하지 않아 Cylinder Gprim
충돌체를 '각진 convex hull'로 근사한다. 각진 다각기둥 바퀴는 구를 때 접촉이
모서리 사이를 튀며 횡마찰이 형성되지 않는다 (개방루프 실측: 잠긴 바퀴의
종방향 마찰은 μ≈1.1 정상, 구르는 바퀴의 횡그립은 μ_eff≈0.05~0.12 로 붕괴
-> 5m/s 이상에서 조향 불능 = "바퀴는 꺾이는데 직진"). Capsule 은 GPU-네이티브
해석 형상이라 매끄럽게 구른다. (URDF 변환 옵션 replace_cylinders_with_capsules
=True 와 동일한 효과. 시각 메시는 Cylinder 그대로 둔다.)

URDF -> USD 재변환 시마다 patch_usd_tire.py 와 함께 재실행할 것:
  1) convert_urdf.py ...
  2) python scripts/patch_usd_tire.py
  3) python scripts/patch_usd_wheel_capsule.py
"""
from __future__ import annotations

import glob
import os
import sys

USD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "f1tenth", "f1tenth.usd")
WHEELS = ["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"]


def _reexec_with_pxr_env():
    site = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                        "site-packages")
    hits = glob.glob(os.path.join(site, "isaacsim", "extscache", "omni.usd.libs-*"))
    if not hits:
        sys.exit("[ERROR] env_isaacsim 환경의 python 으로 실행하세요.")
    ext = hits[0]
    env = os.environ.copy()
    env["PYTHONPATH"] = ext + os.pathsep + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = (os.path.join(ext, "bin") + os.pathsep
                              + os.path.join(sys.prefix, "lib") + os.pathsep
                              + env.get("LD_LIBRARY_PATH", ""))
    env["_PXR_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


try:
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf
except ImportError:
    if os.environ.get("_PXR_REEXEC"):
        raise
    _reexec_with_pxr_env()


def main():
    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        sys.exit(f"[ERROR] USD 열기 실패: {USD_PATH}")

    # 인스턴스 프록시 포함 순회로 바퀴 '충돌' Cylinder prim 의 정의 스펙(레이어) 위치 파악.
    # instanceable 참조 구조라 4바퀴가 소수의 프로토타입 스펙을 공유할 수 있다.
    targets = {}   # (layer_id, spec_path) -> layer
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/f1tenth"), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if (prim.GetTypeName() == "Cylinder" and "/collisions/" in path
                and any(w in path for w in WHEELS)):
            for spec in prim.GetPrimStack():
                if spec.typeName == "Cylinder":
                    targets[(spec.layer.identifier, str(spec.path))] = spec.layer

    if not targets:
        # 이미 패치됐는지 확인
        done = [str(p.GetPath()) for p in
                Usd.PrimRange(stage.GetPrimAtPath("/f1tenth"), Usd.TraverseInstanceProxies())
                if p.GetTypeName() == "Capsule" and "/collisions/" in str(p.GetPath())]
        if done:
            print(f"[OK] 이미 패치됨 (Capsule 충돌체 {len(done)}개). 변경 없음.")
            return
        sys.exit("[ERROR] 바퀴 충돌 Cylinder prim 을 찾지 못했습니다.")

    edited = set()
    for (layer_id, spec_path), layer in targets.items():
        spec = layer.GetPrimAtPath(spec_path)
        r = spec.attributes["radius"].default if "radius" in spec.attributes else 0.05
        h = spec.attributes["height"].default if "height" in spec.attributes else 0.045
        spec.typeName = "Capsule"   # radius/height/axis 속성은 Capsule 스키마와 호환
        # extent 갱신 (캡슐 = 실린더 + 반구 캡: 축방향 ±(h/2 + r))
        ext = spec.attributes.get("extent")
        if ext is not None:
            half = float(h) * 0.5 + float(r)
            ext.default = [(-float(r), -float(r), -half), (float(r), float(r), half)]
        edited.add(layer_id)
        print(f"[PATCH] {layer_id}\n        {spec_path}: Cylinder -> Capsule "
              f"(radius={r}, height={h})")

    for layer_id in edited:
        Sdf.Layer.Find(layer_id).Save()

    # ---- 재검증 ----
    stage2 = Usd.Stage.Open(USD_PATH)
    n_cap = n_cyl = 0
    for prim in Usd.PrimRange(stage2.GetPrimAtPath("/f1tenth"), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if "/collisions/" in path and any(w in path for w in WHEELS):
            if prim.GetTypeName() == "Capsule":
                n_cap += 1
            elif prim.GetTypeName() == "Cylinder":
                n_cyl += 1
    assert n_cyl == 0 and n_cap >= 4, (n_cap, n_cyl)
    print(f"[DONE] 바퀴 충돌체 Capsule {n_cap}개 / 잔여 Cylinder {n_cyl}개. "
          f"시각 메시는 Cylinder 유지.")


if __name__ == "__main__":
    main()
