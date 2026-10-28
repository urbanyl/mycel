"""Native backend: lower a `.myc` ledger to a self-contained executable.

The ledger declares the view, the camera, the body and the policy. This
backend emits a Go program that links the dependency-free `mycel/runtime`
package (Win32 window, software rasteriser with z-buffer, GDI text) and
compiles it with the Go toolchain.

The shipped binary needs no Python, no runtime interpreter, and no DLLs
beyond the Windows system libraries.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from ..parser import Entry, parse_source

ROOT = Path(__file__).resolve().parent.parent.parent
RUNTIME = ROOT / "runtime" / "myc"

VEC = r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]"


class NativeError(RuntimeError):
    pass


def _vec(text: str, key: str) -> Tuple[float, float, float]:
    m = re.search(key + r"\s*" + VEC, text)
    if not m:
        raise NativeError(f"cannot parse {key} [x,y,z]")
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def _directives(entries: List[Entry]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for e in entries:
        if e.role != "directive":
            continue
        body = e.raw[1:].strip()
        name, _, rest = body.partition(" ")
        out[name] = rest.strip()
    return out


def build_native(
    source_path: str | Path,
    out_dir: str | Path | None = None,
    build: bool = True,
    echo=print,
) -> Path:
    source_path = Path(source_path)
    text = source_path.read_text(encoding="utf-8")
    entries = parse_source(text)
    d = _directives(entries)

    view = d.get("view", "1280x720")
    m = re.match(r"(\d+)x(\d+)", view)
    if not m:
        raise NativeError(f"bad $ view '{view}', expected e.g. 1280x720")
    width, height = int(m.group(1)), int(m.group(2))
    title_m = re.search(r'title\s+"([^"]*)"', view)
    title = title_m.group(1) if title_m else "MYCEL"

    style = d.get("style", "3d").strip().lower()
    cam_text = d.get("camera", "at [4.2,1.9,-5.0] look [0,0.75,0] fov 38")
    pos = _vec(cam_text, "at")
    look = _vec(cam_text, "look")
    fov_m = re.search(r"fov\s+([\d.]+)", cam_text)
    fov = float(fov_m.group(1)) if fov_m else 38.0

    follow_m = re.search(r"follow\s+([\d.-]+)", cam_text)
    follow = float(follow_m.group(1)) if follow_m else 0.0
    if style == "2d" and "follow" not in cam_text:
        pos = (0.0, 0.72, -8.6)
        look = (0.0, 0.72, 0.0)
        fov = 26.0
        follow = 0.0

    body = d.get("body", "Biped2D").strip()
    if body not in ("Biped2D", "Biped3D", "biped"):
        raise NativeError(f"unknown body '{body}' (only Biped2D is implemented)")

    policy_rel = d.get("policy", "")
    if not policy_rel:
        raise NativeError("missing '$ policy <file.npz>' directive")
    policy_path = (source_path.parent / policy_rel).resolve()
    if not policy_path.exists():
        alt = (ROOT / policy_rel).resolve()
        if alt.exists():
            policy_path = alt
        else:
            raise NativeError(f"policy file not found: {policy_rel}")

    import numpy as np

    theta = np.load(policy_path)["theta"].reshape(-1)

    name = d.get("name", source_path.stem).strip()
    target = Path(out_dir) if out_dir else ROOT / "build" / name
    target.mkdir(parents=True, exist_ok=True)

    policy_go = "package main\n\nvar policy = []float32{\n"
    for i in range(0, len(theta), 8):
        chunk = ", ".join(f"{float(v):.7g}" for v in theta[i : i + 8])
        policy_go += "\t" + chunk + ",\n"
    policy_go += "}\n"
    (target / "policy.go").write_text(policy_go, encoding="utf-8")

    rel_runtime = Path(
        subprocess.run(
            ["go", "env", "GOROOT"], capture_output=True, text=True, check=False
        ).stdout.strip()
        or "."
    )
    del rel_runtime
    runtime_rel = Path(
        __import__("os").path.relpath(RUNTIME, target)
    ).as_posix()

    (target / "go.mod").write_text(
        "module mycelbuild\n\ngo 1.21\n\nrequire mycel/runtime v0.0.0\n\n"
        f"replace mycel/runtime => {runtime_rel}\n",
        encoding="utf-8",
    )

    grid = "false" if style == "2d" else "true"
    panel = d.get("panel", "brain").strip().lower() not in ("none", "off", "0")
    panel_w = 420
    if panel and width < 1100:
        panel = False

    main_go = f'''package main

import myc "mycel/runtime"

func main() {{
\tscene := &myc.Scene{{
\t\tBody:     myc.NewBiped(policy),
\t\tCam:      myc.NewCamera(myc.V({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}),
\t\t\tmyc.V({look[0]:.4f}, {look[1]:.4f}, {look[2]:.4f}), {fov:.3f}, 0.08, 120),
\t\tGrid:     {grid},
\t\tTrailLen: 12,
\t\tFollowX:  {follow:.4f},
\t\tLookY:    {look[1]:.4f},
\t\tPanel:    {str(panel).lower()},
\t\tPanelW:   {panel_w},
\t}}
\tapp := &myc.App{{Title: {title!r}, Width: {width}, Height: {height}}}
\tapp.Frame = func(g *myc.Gfx, _ float64, dt float64) {{ scene.Step(g, dt) }}
\tmyc.Run(app)
}}
'''
    main_go = main_go.replace("'", '"')
    (target / "main.go").write_text(main_go, encoding="utf-8")

    if not build:
        echo(f"emitted {target}")
        return target

    go = shutil.which("go")
    if not go:
        raise NativeError("the Go toolchain is required for the native backend")

    exe = target / f"{name}.exe"
    echo(f"  go build -> {exe}")
    proc = subprocess.run(
        [go, "build", "-trimpath", "-ldflags", "-s -w", "-o", str(exe), "."],
        cwd=str(target),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise NativeError("go build failed")

    size = exe.stat().st_size
    echo(f"  {exe.name}  {size/1_048_576:.2f} MiB  (no Python, no DLLs)")
    return exe
