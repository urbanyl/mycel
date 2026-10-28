import re
from pathlib import Path

import numpy as np
import pytest

from mycel.native.gen import NativeError, build_native

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "06-biped-native.myc"


def test_emit_only_produces_a_native_program(tmp_path):
    artifact_dir = build_native(EXAMPLE, tmp_path / "out", build=False, echo=lambda *_: None)

    main = (artifact_dir / "main.go").read_text(encoding="utf-8")
    assert "myc.Run(app)" in main
    assert "Width: 1480" in main and "Height: 800" in main
    assert "MYCEL / native 3D" in main
    assert "Panel:" in main

    policy = (artifact_dir / "policy.go").read_text(encoding="utf-8")
    values = re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", policy.split("{", 1)[1])
    assert len(values) == 1168, "the tissue has 1168 parameters"

    gomod = (artifact_dir / "go.mod").read_text(encoding="utf-8")
    assert "replace mycel/runtime =>" in gomod


def test_missing_policy_is_reported(tmp_path):
    src = tmp_path / "bad.myc"
    src.write_text(
        '$ view 320x240 title "x"\n$ body Biped2D\n$ policy nowhere.npz\n',
        encoding="utf-8",
    )
    with pytest.raises(NativeError):
        build_native(src, tmp_path / "out", build=False, echo=lambda *_: None)


def test_unknown_body_is_reported(tmp_path):
    src = tmp_path / "bad.myc"
    src.write_text(
        '$ view 320x240\n$ body Quadruped\n$ policy nowhere.npz\n', encoding="utf-8"
    )
    with pytest.raises(NativeError):
        build_native(src, tmp_path / "out", build=False, echo=lambda *_: None)


def test_policy_file_matches_the_controller_slices():
    from mycel.walker.controller import N_PARAMS

    theta = np.load(ROOT / "runs" / "demo" / "theta.npz")["theta"].reshape(-1)
    assert theta.size == N_PARAMS
