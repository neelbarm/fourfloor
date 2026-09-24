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
