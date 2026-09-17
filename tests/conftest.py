"""Shared fixtures and synthetic signal generators for the test suite."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

SR = 44100
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lofi-7.mp3"


@pytest.fixture(autouse=True, scope="session")
def isolated_home(tmp_path_factory):
    """Point ``FOURFLOOR_HOME`` at a temp folder for the whole run.

    Kits live under that home, and the engine uses the most recently built one
    by default. Without this the suite's results depend on which records the
    person running it happens to have sampled -- a remix rendered with somebody's
    kit is a different remix, and a test that passes on a clean machine could
    fail on a working one.
    """
    home = tmp_path_factory.mktemp("fourfloor-home")
    before = os.environ.get("FOURFLOOR_HOME")
    os.environ["FOURFLOOR_HOME"] = str(home)
    yield home
    if before is None:
        os.environ.pop("FOURFLOOR_HOME", None)
    else:
        os.environ["FOURFLOOR_HOME"] = before


@pytest.fixture(scope="session")
def sr() -> int:
    return SR


def click_track(bpm: float, seconds: float = 20.0, sr: int = SR,
                accent_first: bool = False, seed: int = 0) -> np.ndarray:
    """A metronome: short noise bursts on every beat at ``bpm``.

    With ``accent_first`` every fourth click gets a loud low-frequency thump, so
    the downbeat detector has a bar phase to find.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    t = np.arange(int(0.05 * sr)) / sr
    click = (rng.standard_normal(len(t)).astype(np.float32) * np.exp(-t / 0.004))
    thump = (np.sin(2 * np.pi * 55.0 * t) * np.exp(-t / 0.06)).astype(np.float32)
    i = 0
    while True:
        pos = int(round(i * period * sr))
        if pos + len(click) >= n:
            break
        x[pos:pos + len(click)] += click * 0.6
        if accent_first and i % 4 == 0:
            x[pos:pos + len(thump)] += thump * 1.4
        i += 1
    return x / max(float(np.max(np.abs(x))), 1e-9)


def chord_track(root_pc: int, minor: bool, seconds: float = 12.0,
                sr: int = SR) -> np.ndarray:
    """A I-IV-V-I progression in a given key, as sine triads with harmonics."""
    degrees = [0, 5, 7, 0]
    triad = (0, 3, 7) if minor else (0, 4, 7)
    per = seconds / len(degrees)
    out = []
    for d in degrees:
        n = int(per * sr)
        t = np.arange(n) / sr
        seg = np.zeros(n, dtype=np.float32)
        for iv in triad:
            midi = 60 + root_pc + d + iv
            f = 440.0 * 2 ** ((midi - 69) / 12.0)
            for h, amp in ((1, 1.0), (2, 0.4), (3, 0.2)):
                seg += (np.sin(2 * np.pi * f * h * t) * amp).astype(np.float32)
        env = np.ones(n, dtype=np.float32)
        env[: int(0.01 * sr)] = np.linspace(0, 1, int(0.01 * sr))
        env[-int(0.01 * sr):] = np.linspace(1, 0, int(0.01 * sr))
        out.append(seg * env)
    y = np.concatenate(out)
    return (y / max(float(np.max(np.abs(y))), 1e-9)).astype(np.float32)


def sine(freq: float, seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def peak_freq(y: np.ndarray, sr: int = SR, n: int = 65536, offset: int = SR // 2) -> float:
    """Dominant frequency by parabolic interpolation on the log magnitude."""
    seg = y[offset: offset + n]
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    spec = np.abs(np.fft.rfft(seg * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    k = int(np.argmax(spec))
    if k <= 0 or k >= len(spec) - 1:
        return float(freqs[k])
    a, b, c = (float(np.log(spec[i] + 1e-12)) for i in (k - 1, k, k + 1))
    den = a - 2 * b + c
    shift = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if abs(den) > 1e-9 else 0.0
    return float(freqs[k] + shift * (freqs[1] - freqs[0]))


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    if not FIXTURE.is_file():
        pytest.skip(f"fixture missing: {FIXTURE}")
    return FIXTURE


@pytest.fixture(scope="session")
def fixture_analysis(fixture_path):
    """Analysis of the bundled fixture, shared across the suite."""
    from fourfloor.analysis import analyze
    return analyze(fixture_path)


@pytest.fixture(scope="session")
def quiet_clip(tmp_path_factory) -> Path:
    """A very quiet synthetic source: -66 dBFS peak, with a beat to track.

    Quiet and near-silent sources are where normalisation and the energy curve
    are most likely to divide by something close to zero.
    """
    import soundfile as sf

    x = click_track(100.0, seconds=20.0) * 0.0005
    path = tmp_path_factory.mktemp("quiet") / "quiet.wav"
    sf.write(str(path), np.stack([x, x], axis=1), SR, subtype="PCM_24")
    return path


@pytest.fixture(scope="module")
def remix_of_a_quiet_clip(quiet_clip, tmp_path_factory):
    """One short remix of that clip, reused by the non-finite / level checks."""
    from fourfloor.remix import RemixOptions, remix

    out = tmp_path_factory.mktemp("quietmix") / "quiet.house.mp3"
    return remix(quiet_clip, out, RemixOptions(target_bpm=124.0, length="2:00",
                                               wav=False))
