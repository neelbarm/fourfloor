"""How demucs is invoked, checked without running it."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

import fourfloor.stems as stems_mod


def _fake_demucs(calls: list):
    """A subprocess.run stand-in that records argv, reads the input the way
    demucs would, and writes four float stems."""

    real = subprocess.run

    def run(argv, **kwargs):
        if "demucs" not in argv:            # ffmpeg decoding the stems
            return real(argv, **kwargs)
        calls.append(list(argv))
        out = Path(argv[argv.index("-o") + 1])
        src = Path(argv[-1])
        data, sr = sf.read(str(src), dtype="float32", always_2d=True)
        calls.append(data)
        d = out / stems_mod.DEMUCS_MODEL / src.stem
        d.mkdir(parents=True)
        for name in ("vocals", "other", "drums", "bass"):
            sf.write(str(d / f"{name}.wav"), data * 0.25, sr, subtype="FLOAT")
        return subprocess.CompletedProcess(argv, 0, "", "")

    return run


def test_demucs_runs_without_its_random_shift_and_without_rescaling(monkeypatch) -> None:
    """``--shifts 1`` (demucs's default) separates at a random offset, so the
    same command gave different stems -- and a different remix -- every run."""
    calls: list = []
    monkeypatch.setattr(stems_mod, "demucs_available", lambda: True)
    monkeypatch.setattr(stems_mod.subprocess, "run", _fake_demucs(calls))
    sr = 44100
    x = np.zeros((sr, 2), dtype=np.float32)
    stems_mod.separate_demucs(x, sr)
    argv = calls[0]
    assert argv[argv.index("--shifts") + 1] == "0"
    assert "--float32" in argv
    assert argv[argv.index("--clip-mode") + 1] == "none"


def test_demucs_input_is_not_clipped(monkeypatch) -> None:
    """A hot master decodes above full scale; a 24-bit input file used to
    hard-clip it into every stem before demucs saw it."""
    calls: list = []
    monkeypatch.setattr(stems_mod, "demucs_available", lambda: True)
    monkeypatch.setattr(stems_mod.subprocess, "run", _fake_demucs(calls))
    sr = 44100
    t = np.arange(sr) / sr
    x = (1.4 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    x = np.stack([x, x], axis=1)
    st = stems_mod.separate_demucs(x, sr)
    fed = calls[1]
    assert np.abs(fed).max() > 1.3
    np.testing.assert_allclose(fed, x, atol=1e-6)
    # and the float stems come back as written, not rescaled or clipped
    np.testing.assert_allclose(st.vocals, x * 0.25, atol=1e-6)
