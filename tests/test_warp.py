"""The warp's edges: the lead-in before the song starts, and no grid at all."""

from __future__ import annotations

import numpy as np
import pytest

from fourfloor.analysis import Analysis
from fourfloor.analysis.key import KeyEstimate
from fourfloor.analysis.tempo import BeatGrid
from fourfloor.audio import Clip
from fourfloor.warp import build

SR = 44100


def analysis_of(x: np.ndarray, bpm: float, beats: np.ndarray) -> Analysis:
    return Analysis(path="synthetic.wav", duration=len(x) / SR, sr=SR,
                    grid=BeatGrid(bpm=bpm, beats=np.asarray(beats, dtype=float),
                                  downbeat_index=0),
                    key=KeyEstimate(tonic=0, is_minor=False, confidence=0.9),
                    sections=[], chords=[], rms_db=-12.0, peak_db=-1.0,
                    clip=Clip(samples=x, sr=SR, path="synthetic.wav"))


def test_the_lead_in_before_the_song_is_silence_not_a_drone() -> None:
    """Warped time before the first knot read source frame 0 over and over:
    the remix intro opened on a sustained drone of the song's first 23 ms."""
    seconds = 12.0
    t = np.arange(int(seconds * SR)) / SR
    tone = (0.5 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)
    x = np.stack([tone, tone], axis=1)
    bpm = 120.0
    beats = np.arange(0.2, seconds - 0.5, 60.0 / bpm)
    y, wm = build(analysis_of(x, bpm, beats), 128.0, 1.0)
    head = int(round(float(wm.out_times[0]) * SR))
    assert head > SR, "the stimulus needs a lead-in to test"
    assert float(np.abs(y[: head]).max()) == 0.0
    # and the song itself is all there after it
    after = y[head + SR // 10: head + SR]
    assert float(np.sqrt(np.mean(after ** 2))) > 0.25


@pytest.mark.parametrize("multiple", [1.0, 2.0])
def test_no_grid_is_stretched_toward_the_target_not_away_from_it(multiple) -> None:
    """With under two beats, 103 -> 128 BPM was slowed to 83 and the map
    stayed the identity, disagreeing with the buffer it described."""
    x = np.zeros((int(4.0 * SR), 2), dtype=np.float32)
    x[:: SR // 8] = 0.5
    src = 103.0 if multiple == 1.0 else 64.0
    y, wm = build(analysis_of(x, src, np.array([0.1])), 128.0, multiple)
    rate = 128.0 / (src * multiple)
    assert len(y) == pytest.approx(len(x) / rate, abs=2)
    assert wm.duration == pytest.approx(len(y) / SR, abs=1e-6)
    assert float(wm.to_warped(4.0)) == pytest.approx(4.0 / rate, rel=1e-6)


def test_no_grid_with_a_key_change_keeps_its_length() -> None:
    x = (0.3 * np.sin(2 * np.pi * 220.0 * np.arange(int(3.0 * SR)) / SR)).astype(np.float32)
    x = np.stack([x, x], axis=1)
    y, wm = build(analysis_of(x, 103.0, np.array([0.1])), 128.0, 1.0, semitones=3)
    assert len(y) == pytest.approx(len(x) * 103.0 / 128.0, rel=0.01)
    assert wm.duration == pytest.approx(len(y) / SR, abs=1e-6)
