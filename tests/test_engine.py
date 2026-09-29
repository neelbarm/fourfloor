"""Engine-level regressions that need no real song."""

from __future__ import annotations

import numpy as np
import pytest

from fourfloor.house.engine import _loop_to


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("remainder", [1, 7, 500, 1057])
def test_loop_to_survives_a_last_repeat_shorter_than_the_crossfade(stereo, remainder) -> None:
    """``want % period`` in (0, fade) used to raise a broadcast ValueError.

    The sampled-kit drum bed asks for ``round(total bars)`` samples looped at
    ``round(8 bars)``, and the two roundings leave a few samples over at 118,
    119, 123, 129 and 130 BPM -- the render died after all the analysis and
    separation work was done.
    """
    sr = 44100
    period = sr                                   # fade = 0.024 * sr = 1058
    want = 3 * period + remainder
    rng = np.random.default_rng(0)
    src = rng.standard_normal((2 * period, 2) if stereo else 2 * period).astype(np.float32)
    out = _loop_to(src, 0, want, period, sr)
    assert out.shape[0] == want
    assert np.all(np.isfinite(out))
    # the tail is the new repeat faded in over the previous repeat's overlap
    # (the material past the loop point) faded out
    fade = int(0.024 * sr)
    rise = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:remainder]
    if stereo:
        rise = rise[:, None]
    expect = src[:remainder] * rise + src[period:period + remainder] * (1.0 - rise)
    np.testing.assert_allclose(out[3 * period:], expect, rtol=1e-5, atol=1e-6)


def test_sampled_drum_bed_at_129_bpm_does_not_crash() -> None:
    """The exact shape of the drum-bed call at a tempo that used to crash."""
    sr = 44100
    bpm = 129.0
    bar = 4 * 60.0 / bpm
    period = int(round(8 * bar * sr))
    total = int(round(144 * bar * sr))
    assert 0 < total % period < int(0.024 * sr)    # the case that crashed
    loop = np.zeros((period, 2), dtype=np.float32)
    loop[:: sr // 4] = 1.0
    bed = _loop_to(np.concatenate([loop, loop]), 0, total, period, sr)
    assert bed.shape == (total, 2)


@pytest.fixture(scope="module")
def chop_engine(fixture_analysis):
    from fourfloor.arrange import plan
    from fourfloor.house.engine import Engine, Stems

    sr = 44100
    p = plan(fixture_analysis, 128.0, 2.0, length=60.0)
    n = int(p.total_bars * p.bar_dur * sr) + sr
    zero = np.zeros((n, 2), dtype=np.float32)
    return Engine(sr=sr, plan=p, stems=Stems(harmonic=zero, percussive=zero),
                  chords=fixture_analysis.chords, beat_multiple=2.0,
                  vocal_mode="chop")


def test_a_cut_short_chop_slice_ends_faded_not_mid_syllable(chop_engine) -> None:
    """A 'stut' reuses the first beat of a two-beat slice, whose fade-out was at
    its second beat; the cut used to step from full level to silence -- a click
    once every four bars through every chopped drop. The last slice, cut to the
    slot, did the same. Every piece has to end at (near) silence."""
    eng = chop_engine
    sr = eng.sr
    beat_n = max(64, int(round(eng.beat * sr)))
    want = 16 * beat_n * 2 + beat_n // 2               # two units and a half-beat stub
    t = np.arange(want + 8 * beat_n) / sr
    # a sustained voice with a syllable accent every eighth, so there are
    # onsets to cut at and no natural silence to hide a hard edge in
    tone = 0.5 * np.sin(2 * np.pi * 220.0 * t) * (0.6 + 0.4 * (np.mod(t, eng.beat / 2) < 0.03))
    voc = np.stack([tone, tone], axis=1).astype(np.float32)
    out = eng._chop_vocal(voc, want)
    assert out.shape[0] == want
    peak = float(np.abs(out).max())
    assert peak > 0.1
    ends, pos = [], 0
    while pos < want:
        for _what, beats in eng.CHOP_PATTERN:
            pos += beats * beat_n
            ends.append(min(pos, want))
            if pos >= want:
                break
    worst = max(float(np.abs(out[e - 1]).max()) for e in ends)
    assert worst < 0.02 * peak, f"a piece ends at {worst / peak:.0%} of peak"
