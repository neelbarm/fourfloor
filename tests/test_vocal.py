"""Playing a voice straight through, or cutting it into pieces that land.

A listener called one of these renders amazing and the other "terrible with
overlaps". The plans were structurally identical; what differed was the voice.
CAN'T SAY sits on a straight sixteenth lattice and can simply be played. *Body*
is a triplet flow -- three syllables to the beat -- which lands a third of a
beat from anything on that lattice and will not lock to four-on-the-floor
however it is warped. These tests pin down the measurement that tells them
apart and the chopping that rescues the second case.
"""

from __future__ import annotations

import numpy as np
import pytest

from fourfloor.analysis import alignment as A
from fourfloor.house.vocal import choose_vocal

SR = 44100
BPM = 128.0
BEAT = 60.0 / BPM

STRAIGHT = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.25, 2.5, 3, 3.5, 3.75]
TRIPLET = [0, 1 / 3, 2 / 3, 1, 4 / 3, 5 / 3, 2, 7 / 3, 8 / 3, 3, 10 / 3, 11 / 3]
#: A sparse triplet flow: one syllable a beat, every one of them off the beat.
#: Nothing in it is on a beat to begin with, so anything that ends up on one
#: got there by being placed there.
SPARSE_TRIPLET = [1 / 3, 1 + 1 / 3, 2 + 1 / 3, 3 + 1 / 3]


def voice(positions, seconds: float = 32.0, hz: float = 220.0,
          sr: int = SR) -> np.ndarray:
    """Syllables at bar-relative beat positions, repeated every bar."""
    n = int(seconds * sr)
    x = np.zeros(n, dtype=np.float32)
    ln = int(0.13 * sr)
    t = np.arange(ln) / sr
    syl = (np.sin(2 * np.pi * hz * t) * (1 + 0.5 * np.sin(2 * np.pi * 6 * t))
           * np.exp(-t / 0.05)).astype(np.float32)
    bar = 0
    while bar * 4 * BEAT * sr < n:
        for p in positions:
            a = int(round((bar * 4 * BEAT + p * BEAT) * sr))
            b = min(n, a + ln)
            if b > a:
                x[a:b] += syl[: b - a]
        bar += 1
    return x


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

def test_a_straight_flow_measures_as_straight() -> None:
    f = A.vocal_fit(voice(STRAIGHT), SR, BPM)
    assert f["straight"] > 0.9, f
    assert f["advantage"] < 0.0, f
    assert f["duty"] > 0.5


def test_a_triplet_flow_measures_as_triplet() -> None:
    f = A.vocal_fit(voice(TRIPLET), SR, BPM)
    assert f["straight"] < 0.5, f
    assert f["triplet"] > 0.9, f
    assert f["advantage"] > 0.3, f
    assert f["scatter_ms"] > 20.0, "triplet syllables are far from a sixteenth"


def test_silence_has_no_opinion() -> None:
    f = A.vocal_fit(np.zeros(SR * 10, dtype=np.float32), SR, BPM)
    assert f["duty"] == 0.0
    mode, _m, why = choose_vocal(np.zeros(SR * 10, dtype=np.float32), SR, BPM)
    assert mode == "flow" and "barely a vocal" in why


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------

def test_a_voice_that_locks_is_played_as_it_was_sung() -> None:
    """The CAN'T SAY case, and the one a listener liked."""
    mode, m, why = choose_vocal(voice(STRAIGHT), SR, BPM)
    assert mode == "flow", (m, why)
    assert "lands on the grid" in why


def test_a_voice_that_will_not_lock_is_chopped() -> None:
    """The Body case, and the one he called terrible."""
    mode, m, why = choose_vocal(voice(TRIPLET), SR, BPM)
    assert mode == "chop", (m, why)
    assert "triplet" in why


# ---------------------------------------------------------------------------
# the chopping
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine(fixture_analysis):
    from fourfloor.arrange import plan
    from fourfloor.house.engine import Engine, Stems

    p = plan(fixture_analysis, BPM, 2.0, length=60.0)
    n = int(p.total_bars * p.bar_dur * SR) + SR
    zero = np.zeros((n, 2), dtype=np.float32)
    return Engine(sr=SR, plan=p, stems=Stems(harmonic=zero, percussive=zero),
                  chords=fixture_analysis.chords, beat_multiple=2.0,
                  vocal_mode="chop")


def _on_beat(x: np.ndarray, seconds: float, tol: float = 0.010) -> float:
    """Share of onset energy within ``tol`` of a beat."""
    onsets, strength = A.onset_times(x, SR)
    if not len(onsets):
        return 0.0
    beats = A.grid_times(BPM, 0.0, seconds, division=1)
    err = np.abs(A.phase_errors(onsets, beats))
    w = np.asarray(strength, dtype=float)
    return float(w[err <= tol].sum() / max(w.sum(), 1e-12))


def test_slices_are_placed_on_beats(engine) -> None:
    """The whole point: a syllable begins each slice, and it lands on a beat.

    The stimulus has a syllable on every beat *except* the beat -- one a bar
    apart, each a third of a beat late -- so nothing in it starts on a beat and
    anything that ends up on one got there by being put there. A slice is cut
    *at* a vocal onset and placed *at* a grid position, so its first and loudest
    transient is on the beat by construction; nothing is stretched.
    """
    want = int(16 * engine.bar_dur * SR)
    voc = np.stack([voice(SPARSE_TRIPLET, seconds=want / SR + 4)] * 2, axis=1)
    out = engine._chop_vocal(voc[: want + SR], want)
    assert out.shape[0] == want

    before = _on_beat(voc[:want], want / SR)
    after = _on_beat(out, want / SR)
    assert before < 0.05, f"the stimulus was already {before:.0%} on the beat"
    assert after > 0.25, f"only {after:.0%} of the chop lands on a beat"

    # and the ones that do land, land tightly
    onsets, strength = A.onset_times(out, SR)
    beats = A.grid_times(BPM, 0.0, want / SR, division=1)
    err = np.abs(A.phase_errors(onsets, beats))
    near = err[err <= 0.030]
    assert len(near) > 8
    assert float(np.median(near)) < 0.010, \
        f"slice starts sit {np.median(near) * 1000:.1f} ms off the beat"


def test_chopping_repeats_and_leaves_gaps(engine) -> None:
    """The device a listener named: splits, repeats and room to answer into."""
    want = int(16 * engine.bar_dur * SR)
    voc = np.stack([voice(SPARSE_TRIPLET, seconds=want / SR + 4)] * 2, axis=1)
    out = engine._chop_vocal(voc[: want + SR], want)
    def quiet_fraction(x: np.ndarray) -> float:
        mono = x.mean(axis=1) if x.ndim == 2 else x
        mono = mono[: len(mono) // 1024 * 1024]
        rms = np.sqrt(np.mean(mono.reshape(-1, 1024) ** 2, axis=1))
        return float(np.mean(rms < 0.05 * max(rms.max(), 1e-9)))

    # Measured against the take, not against silence: this stimulus is sparse
    # to begin with, so most of it is quiet either way. What the chop adds is
    # the rests in the pattern, and they have to be there without the whole
    # thing turning into a gate.
    before = quiet_fraction(voc[:want])
    after = quiet_fraction(out)
    assert after > before, f"the chop left no gaps ({after:.0%} vs {before:.0%})"
    assert after < before + 0.35, f"the chop is mostly silence ({after:.0%})"


def test_a_vocal_too_short_to_chop_is_left_alone(engine) -> None:
    quiet = np.zeros((SR * 4, 2), dtype=np.float32)
    out = engine._chop_vocal(quiet, SR * 2)
    assert len(out) == SR * 2
