"""Two rhythms at once: measuring which layer is playing the other one.

The phase gate answers "is this in time". It is perfectly possible for every
layer of a remix to pass it and for the remix to still sound like two records
playing at once, because a part can be exactly on the sixteenth lattice and
still be playing a pattern that fights four-on-the-floor from end to end. That
is what a trap 808 does under a house kick: an 808 is a kick with a pitch, so
laying one under a four-on-the-floor kit gives you two kick drums.

These tests pin down the measurement that tells them apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from fourfloor.analysis import alignment as A

SR = 44100
BPM = 128.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT


def _notes(positions: list[float], hz: float = 55.0, length: float = 0.35,
           seconds: float = 32.0, sr: int = SR, decay: float | None = None
           ) -> np.ndarray:
    """Sine notes at a list of bar-relative positions, repeated every bar."""
    n = int(seconds * sr)
    x = np.zeros(n, dtype=np.float32)
    ln = int(length * sr)
    t = np.arange(ln) / sr
    env = np.exp(-t / (decay if decay else length * 0.5)).astype(np.float32)
    env[: int(0.005 * sr)] *= np.linspace(0.0, 1.0, int(0.005 * sr))
    note = (np.sin(2 * np.pi * hz * t).astype(np.float32) * env)
    bar = 0
    while bar * BAR * sr < n:
        for p in positions:
            a = int(round((bar * BAR + p * BAR) * sr))
            b = min(n, a + ln)
            if 0 <= a < n and b > a:
                x[a:b] += note[: b - a]
        bar += 1
    return x


def house_bass(seconds: float = 32.0) -> np.ndarray:
    """Offbeat eighths: the rolling bass that dodges a four-on-the-floor kick."""
    return _notes([1 / 8, 3 / 8, 5 / 8, 7 / 8], seconds=seconds, length=0.20)


def sustained_bass(seconds: float = 32.0) -> np.ndarray:
    """One note a bar, held: on the grid and not rhythmic at all."""
    return _notes([0.0], seconds=seconds, length=BAR * 0.95, decay=BAR)


def trap_808(seconds: float = 32.0) -> np.ndarray:
    """A syncopated pattern that also lands squarely on every beat."""
    return _notes([0.0, 3 / 16, 1 / 4, 1 / 2, 10 / 16, 3 / 4, 14 / 16],
                  seconds=seconds, length=0.22)


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

def test_house_positions_are_beats_and_offbeat_eighths() -> None:
    g = A.house_positions(BPM, 0.0, 2.0)
    assert np.allclose(np.diff(g), BEAT / 2)
    assert g[0] == pytest.approx(0.0, abs=1e-9)


def test_a_part_on_the_house_grid_reads_as_on_it() -> None:
    r = A.rhythm_report(house_bass(), SR, BPM)
    assert r["onsets_per_bar"] == pytest.approx(4.0, abs=0.6)
    assert r["off_house"] < 0.05, r


def test_a_part_across_the_grid_reads_as_across_it() -> None:
    r = A.rhythm_report(trap_808(), SR, BPM)
    assert r["onsets_per_bar"] > 6.0, r
    assert r["off_house"] > 0.12, r
    assert r["off_house"] > 4 * A.rhythm_report(house_bass(), SR, BPM)["off_house"]


def test_a_held_note_is_not_a_rhythm() -> None:
    r = A.rhythm_report(sustained_bass(), SR, BPM)
    assert r["onsets_per_bar"] < 2.0, r


def test_silence_reports_as_silent_not_as_a_rhythm() -> None:
    quiet = np.zeros((SR * 8, 2), dtype=np.float32)
    rep = A.overlap_report({"quiet": quiet, "loud": np.stack([house_bass()] * 2, 1)},
                           SR, BPM)
    assert rep["layers"]["quiet"]["silent"]
    assert rep["layers"]["quiet"]["whole"]["onsets_per_bar"] == 0.0
    assert not rep["layers"]["loud"]["silent"]


def test_overlap_reports_section_by_section() -> None:
    x = np.concatenate([house_bass(16.0), trap_808(16.0)])
    rep = A.overlap_report({"bass": x}, SR, BPM, 0.0,
                           [("first", 0.0, 16.0), ("second", 16.0, 32.0)])
    secs = rep["layers"]["bass"]["sections"]
    assert secs["first"]["off_house"] < secs["second"]["off_house"]
    assert secs["first"]["onsets_per_bar"] < secs["second"]["onsets_per_bar"]


# ---------------------------------------------------------------------------
# the decision it feeds
# ---------------------------------------------------------------------------

def test_an_808_is_measured_as_hitting_where_the_kick_goes() -> None:
    m = A.bass_collision(trap_808(), SR, BPM)
    assert m["per_bar"] > 4.0, m
    assert m["on_beat"] > 0.35, m


def test_a_rolling_house_bass_is_measured_as_dodging_the_kick() -> None:
    m = A.bass_collision(house_bass(), SR, BPM)
    assert m["on_beat"] < 0.2, m


def test_a_held_note_is_measured_as_barely_rhythmic() -> None:
    m = A.bass_collision(sustained_bass(), SR, BPM)
    assert m["per_bar"] < 3.0, m
