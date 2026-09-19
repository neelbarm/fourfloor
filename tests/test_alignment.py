"""The beat-alignment gate: does a finished remix sit on its own grid?

The complaint this exists for was "some of it is off beat", and the reason it
took so long to find is that nothing in the project ever looked at a render and
asked where its content was relative to the grid it was built on. These tests
pin the ruler down first -- a signal placed exactly on a grid has to measure as
exactly on it, and a signal moved by a known amount has to measure as moved by
that amount -- and then run the finished gate over real renders.
"""

from __future__ import annotations

import numpy as np
import pytest

from fourfloor.analysis import alignment as A
from fourfloor.arrange import plan, validate
from fourfloor.house.engine import _loop_to
from fourfloor.remix import RemixOptions, remix
from fourfloor.warp import WarpMap

from conftest import house_track

BPM = 128.0
BEAT = 60.0 / BPM


def shifted(x: np.ndarray, seconds: float, sr: int) -> np.ndarray:
    """The same audio, moved later by ``seconds`` (earlier if negative)."""
    n = int(round(seconds * sr))
    if n == 0:
        return x
    if n > 0:
        return np.concatenate([np.zeros(n, dtype=x.dtype), x])[: len(x)]
    return np.concatenate([x[-n:], np.zeros(-n, dtype=x.dtype)])


# ---------------------------------------------------------------------------
# the ruler
# ---------------------------------------------------------------------------

def test_a_grid_is_anchored_on_bars_whatever_the_division() -> None:
    """``grid[0]`` must be a beat one, or ``bar_phase`` measures nonsense."""
    for division in (1, 2, 4):
        g = A.grid_times(BPM, 0.3, 4.0, division=division)
        assert g[0] < 0.3
        bars = (0.3 - g[0]) / (4 * BEAT)
        assert abs(bars - round(bars)) < 1e-9, "the lead-in is not a whole bar"
        assert np.allclose(np.diff(g), BEAT / division)


def test_material_exactly_on_the_grid_measures_as_exactly_on_it(sr: int) -> None:
    x = house_track(BPM, seconds=30.0, sr=sr)
    r = A._layer_report(x, sr, BPM, 0.0)
    assert r["median_ms"] < 5.0, r
    assert r["p90_ms"] < 15.0, r
    assert r["within_20ms"] > 0.95, r
    assert r["beat_alignment"] == "grid"
    assert abs(r["comb_offset_ms"]) < 6.0


def test_a_known_offset_measures_as_that_offset(sr: int) -> None:
    """A 40 ms shift has to read as 40 ms, or the numbers mean nothing."""
    x = shifted(house_track(BPM, seconds=30.0, sr=sr), 0.040, sr)
    r = A._layer_report(x, sr, BPM, 0.0)
    assert 34.0 < r["median_ms"] < 46.0, r
    assert 30.0 < r["comb_offset_ms"] < 50.0, r


def test_a_half_beat_out_is_reported_as_half_a_beat_out(sr: int) -> None:
    x = shifted(house_track(BPM, seconds=30.0, sr=sr), BEAT / 2.0, sr)
    r = A._layer_report(x, sr, BPM, 0.0)
    assert r["beat_alignment"] == "half-beat", r
    assert r["half_beat_ratio"] > A.HALF_BEAT_RATIO


def test_a_bar_out_of_phase_is_reported_as_a_bar_phase(sr: int) -> None:
    """Every hit on the lattice and bar one on beat three is still wrong."""
    x = shifted(house_track(BPM, seconds=40.0, sr=sr), 2 * BEAT, sr)
    r = A._layer_report(x, sr, BPM, 0.0)
    assert r["median_ms"] < 6.0, "the hits are still on the lattice"
    assert r["bar_phase"] == 2, r


def test_the_gate_reports_the_problem_it_found(sr: int) -> None:
    x = shifted(house_track(BPM, seconds=30.0, sr=sr), 0.055, sr)
    rep = A.alignment_report(x, sr, BPM, 0.0)
    assert not rep["ok"]
    assert any("phase error" in p for p in rep["problems"]), rep["problems"]


def test_overlapping_spans_are_counted() -> None:
    assert A.span_overlap([(0, 100), (100, 200)], 200)["max_concurrent"] == 1
    bad = A.span_overlap([(0, 150), (100, 200)], 200)
    assert bad["max_concurrent"] == 2
    assert bad["overlaps"]
    assert A.span_overlap([(0, 50), (100, 200)], 200)["gaps"]


# ---------------------------------------------------------------------------
# the things the gate found
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("downbeat_index", [0, 1, 2, 3])
@pytest.mark.parametrize("beat_multiple", [0.5, 1.0, 2.0])
def test_the_warp_puts_source_downbeats_on_bar_lines(downbeat_index, beat_multiple):
    """Whatever the parity, whatever the half/double-time reading.

    This is the off-beat bug in one assertion. The old warp started the source
    one target beat into the buffer and the arranger assumed it started at zero,
    so a source downbeat landed wherever it landed.
    """
    beats = np.arange(0, 60.0, 60.0 / 146.0) + 0.046
    wm = WarpMap.from_grid(beats, downbeat_index, 128.0, beat_multiple, 60.0)
    lines = wm.bar_downbeats
    assert len(lines) > 8
    assert wm.out_times[0] >= 0.0, "the lead-in does not fit"
    bars = lines / wm.bar_dur
    assert np.allclose(bars, np.round(bars)), "a source downbeat is mid-bar"
    # and snapping any moment in the song lands on one of them
    for t in (0.0, 5.0, 17.3, 41.9):
        assert float(np.min(np.abs(lines - wm.snap(t)))) < 1e-9


def test_a_loop_keeps_its_length_and_its_period(sr: int) -> None:
    """Joining repeats with a crossfade lost 24 ms every time round.

    Eight bars at 128 BPM is a 15-second period, so a drop three repeats long
    finished most of a sixteenth note ahead of the grid, and the error grew all
    the way through the track.
    """
    period = int(2.0 * sr)
    src = np.sin(2 * np.pi * 50.0 * np.arange(6 * sr) / sr).astype(np.float32)
    src = np.stack([src, src], axis=1)
    want = int(7.0 * sr)
    out = _loop_to(src, 0, want, period, sr)
    assert len(out) == want
    fade = int(0.024 * sr)                 # skip the seam itself
    for k in range(1, 3):
        a = k * period + fade
        b = a + period // 2
        assert np.allclose(out[a:b], out[fade:fade + period // 2], atol=2e-3), \
            f"repeat {k} has drifted"


def test_every_slot_starts_on_a_source_bar_line(fixture_analysis) -> None:
    p = plan(fixture_analysis, 124.0, 2.0, length=180.0)
    assert validate(p) == []
    for slot in p.slots:
        bars = slot.source_start / p.bar_dur
        assert abs(bars - round(bars)) < 1e-4, slot


def test_the_arrangement_walks_forward_and_the_breakdown_is_elsewhere(
        fixture_analysis) -> None:
    p = plan(fixture_analysis, 124.0, 2.0, length=270.0)
    drops = [s for s in p.slots if s.kind == "drop"]
    breakdown = next(s for s in p.slots if s.kind == "breakdown")
    assert len(drops) >= 2
    assert drops[1].source_start >= drops[0].source_start
    assert breakdown.source_bars >= 8, "a breakdown needs real material"
    for d in drops:
        same = (abs(breakdown.source_start - d.source_start) < p.bar_dur)
        assert not same, "the breakdown is replaying a drop"


# ---------------------------------------------------------------------------
# whole renders
# ---------------------------------------------------------------------------

def _gate(res) -> dict:
    return A.alignment_report(
        res.audio, res.sr, res.plan.target_bpm, 0.0,
        source_stem=res.layers.get("source_perc"),
        kit_layer=res.layers.get("kit"), spans=res.source_spans)


@pytest.fixture(scope="module")
def fixture_remix(fixture_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("gate") / "fixture.house.mp3"
    return remix(fixture_path, out, RemixOptions(target_bpm=124.0, length="2:00",
                                                 wav=False, keep_layers=True))


@pytest.fixture(scope="module")
def trap_remix(trap_clip, tmp_path_factory):
    out = tmp_path_factory.mktemp("gate") / "trap.house.mp3"
    return remix(trap_clip, out, RemixOptions(target_bpm=128.0, length="2:00",
                                              wav=False, keep_layers=True))


def test_a_render_sits_on_its_own_grid(trap_remix) -> None:
    """The acceptance gate, on a source with the shape of a real record.

    Half-time hip-hop at 146 BPM of hi-hats over a 73 BPM feel, with rolls, a
    bassline and a pad: the tempo octave, the downbeat parity and the warp all
    have to be right at once or this fails.
    """
    rep = _gate(trap_remix)
    assert rep["problems"] == [], rep
    assert rep["source"]["median_ms"] < A.MAX_MEDIAN_MS
    assert rep["source"]["p90_ms"] < A.MAX_P90_MS
    assert rep["source"]["within_20ms"] > A.MIN_ON_GRID
    assert rep["source"]["bar_phase"] == 0
    assert rep["spans"]["max_concurrent"] == 1


def test_the_fixture_render_is_tighter_than_the_fixture(fixture_remix,
                                                        fixture_analysis) -> None:
    """The fixture gets a relative gate, and it is the honest one for it.

    ``lofi-7`` is swung lo-fi at 80 BPM, read half-time onto a 124 BPM grid.
    Swung sixteenths do not live on a sixteenth lattice by definition, so no
    absolute threshold against that lattice describes this track: measured
    against its own detected grid, before anything touches it, the source
    already reads 30 ms median with 13% of its onsets within 20 ms. What a
    remix engine owes it is not to make that worse -- and in fact warping every
    beat onto an exactly periodic grid makes it better, which is the assertion
    below.

    The structural promises still hold absolutely: bar one on bar one, one slot
    of source audio at a time.
    """
    from fourfloor.stems import separate

    a = fixture_analysis
    before = A._layer_report(separate(a.clip.samples, a.sr, "hpss").percussive,
                             a.sr, a.grid.bpm, a.grid.first_downbeat)
    rep = _gate(fixture_remix)
    after = rep["source"]
    assert after["median_ms"] <= before["median_ms"], (before, after)
    assert after["within_20ms"] >= before["within_20ms"], (before, after)
    # Not the comb offset: on a swung source the comb has no single answer to
    # give -- its sharpness here is 0.70 and the two nearest peaks are a swung
    # sixteenth apart -- so comparing the two numbers compares two different
    # arbitrary choices. What it can still say is that nothing ended up on the
    # offbeat, and that is asserted.
    assert after["beat_alignment"] != "half-beat"
    assert after["bar_phase"] == 0
    assert rep["spans"]["max_concurrent"] == 1
    assert not rep["spans"]["overlaps"]


@pytest.mark.parametrize("which", ["fixture_remix", "trap_remix"])
def test_the_kit_is_on_the_grid_by_construction(which, request) -> None:
    """The control. If the kit reads off, the grid handed to the gate is wrong."""
    rep = _gate(request.getfixturevalue(which))
    assert rep["kit"]["median_ms"] < 3.0, rep["kit"]
    assert rep["kit"]["within_20ms"] > 0.95, rep["kit"]


def test_a_half_time_source_is_read_as_half_time(trap_remix) -> None:
    """146 BPM of hi-hats over a 73 BPM feel, laid on a 128 BPM grid.

    The tempo estimate has to choose an octave and the warp has to choose a
    parity; getting either wrong is a remix that is off by a beat from bar one.
    """
    assert trap_remix.tempo_plan.beat_multiple == 2.0
    assert abs(trap_remix.analysis.grid.bpm - 73.0) < 1.5
    assert trap_remix.warp is not None
    lines = trap_remix.warp.bar_downbeats
    bars = lines / trap_remix.plan.bar_dur
    assert np.allclose(bars, np.round(bars))
