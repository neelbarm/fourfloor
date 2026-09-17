"""Phrase-aware cut points, phrase-multiple spans and transition descriptors.

The source here is synthetic and its phrase gaps are placed by hand, so
"the cut landed in a gap" is a fact rather than an opinion. It is built to be
analysable end to end: a metronome gives the beat tracker a grid, a band-limited
"vocal" gives the phrase finder something to phrase, and a loud/quiet contour
gives the segmenter section boundaries to find.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fourfloor import arrange
from fourfloor.analysis import analyze
from fourfloor.analysis import structure as S
from fourfloor.arrange import (BEATS_PER_BAR, TRANSITION_KINDS, FORMS,
                               apply_transitions, plan, validate)

from test_structure_phrasing import BAR, SR, band_noise, click

SRC_BPM = 120.0
TARGET_BPM = 124.0


def phrased_song(bars: int = 96) -> np.ndarray:
    """A synthetic song: 8-bar phrases that rest in their last bar.

    Loud blocks and quiet blocks alternate every 16 bars so the segmenter has
    real boundaries to find, and the vocal rests for a whole bar at the end of
    every 8-bar phrase -- which is where a human editor would cut.
    """
    out = []
    for b in range(bars):
        n = int(BAR * SR)
        singing = (b % 8) != 7
        block_gain = 1.0 if (b // 16) % 2 == 0 else 0.45
        seg = np.zeros(n, dtype=np.float32)
        if singing:
            v = band_noise(BAR, sr=SR, seed=b % 13 + 1)[:n]
            ramp = int(0.02 * SR)
            env = np.ones(n, dtype=np.float32)
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
            seg += v * env * block_gain
        out.append(seg)
    voc = np.concatenate(out)
    mix = voc + click(len(voc) / SR) * 0.4
    mix = mix / max(float(np.max(np.abs(mix))), 1e-9)
    return np.stack([mix, mix], axis=1).astype(np.float32)


def rest_bars(bars: int = 96) -> set[int]:
    """Bar indices where the synthetic vocal is silent."""
    return {b for b in range(bars) if b % 8 == 7}


@pytest.fixture(scope="module")
def song(tmp_path_factory):
    import soundfile as sf
    path = tmp_path_factory.mktemp("phrased") / "phrased.wav"
    sf.write(str(path), phrased_song(), SR, subtype="PCM_16")
    return path


@pytest.fixture(scope="module")
def song_analysis(song):
    return analyze(song)


@pytest.fixture(scope="module")
def song_plan(song_analysis):
    beat_multiple = TARGET_BPM / SRC_BPM
    return plan(song_analysis, TARGET_BPM, beat_multiple, form_name="club",
                length=270.0)


# --------------------------------------------------------------------------
# Cut points
# --------------------------------------------------------------------------


def _source_spans(p, a):
    """Each slot's source span back in *source* seconds."""
    beat_multiple = TARGET_BPM / SRC_BPM
    scale = a.bar_dur / (beat_multiple * p.bar_dur)
    return [(s, s.source_start * scale, s.source_end * scale) for s in p.slots]


def test_the_synthetic_source_is_actually_phraseable(song_analysis) -> None:
    """If this fails the rest of the file is testing nothing."""
    vm = arrange.source_vocal_map(song_analysis)
    assert vm.usable, vm.to_dict()
    for bar in sorted(rest_bars())[:8]:
        assert vm.in_gap(bar * BAR + BAR / 2), f"rest at bar {bar} not found"


def test_every_cut_lands_on_a_downbeat(song_plan, song_analysis) -> None:
    downbeats = np.asarray(song_analysis.grid.downbeats)
    tol = 0.05 * song_analysis.bar_dur
    for slot, entry, exit_ in _source_spans(song_plan, song_analysis):
        assert float(np.min(np.abs(downbeats - entry))) <= tol, \
            f"{slot.kind} entry at {entry:.2f}s is not on a downbeat"
        # the exit is entry + a whole number of bars, therefore also a downbeat
        bars = (exit_ - entry) / song_analysis.bar_dur
        assert abs(bars - round(bars)) < 0.02, \
            f"{slot.kind} span is {bars:.3f} bars, not a whole number"


def test_every_cut_lands_in_a_phrase_gap(song_plan, song_analysis) -> None:
    """The source rests every eighth bar, so there is no excuse to tear a word."""
    vm = arrange.source_vocal_map(song_analysis)
    torn = []
    for slot, entry, exit_ in _source_spans(song_plan, song_analysis):
        if slot.cut_in_mid_phrase:
            torn.append((slot.kind, "entry", round(entry, 2)))
        if slot.cut_out_mid_phrase:
            torn.append((slot.kind, "exit", round(exit_, 2)))
        assert vm.in_gap(entry, pad=0.08), \
            f"{slot.kind} entry at {entry:.2f}s is mid-phrase"
    assert torn == [], f"cuts tore a word: {torn}"


def test_cuts_stay_inside_the_search_window(song_plan) -> None:
    for s in song_plan.slots:
        assert abs(s.cut_moved_bars) <= S.SNAP_WINDOW_BARS + 1e-6, \
            f"{s.kind} moved {s.cut_moved_bars} bars, window is {S.SNAP_WINDOW_BARS}"


def test_every_cut_records_why_it_was_chosen(song_plan) -> None:
    for s in song_plan.slots:
        assert s.cut_in_reason and s.cut_out_reason, f"{s.kind} gave no reason"
        assert "downbeat" in s.cut_in_reason


def test_spans_are_whole_phrases(song_plan) -> None:
    """The engine loops `source_bars`; a non-phrase period drifts across the bar."""
    for s in song_plan.slots:
        assert s.source_bars % 4 == 0, \
            f"{s.kind} reads {s.source_bars} source bars, not a 4-bar multiple"
        if s.kind == "drop":
            assert s.source_bars % 8 == 0, \
                f"a drop reads {s.source_bars} source bars, not an 8-bar multiple"
        assert s.source_end > s.source_start


def test_drops_walk_the_source_forward(song_plan) -> None:
    """Two drops must not be the same 24 bars played twice."""
    drops = [s for s in song_plan.slots if s.kind == "drop"]
    assert len(drops) == 2
    assert drops[1].source_start >= drops[0].source_start, "the second drop rewound"
    assert drops[1].source_start != drops[0].source_start, \
        "both drops read the same span; the source is not being walked"
    # the continuation is snapped, so it may pull back by up to the search
    # window; what it may not do is land back inside the middle of drop 1
    beat_multiple = TARGET_BPM / SRC_BPM
    slack = S.SNAP_WINDOW_BARS * beat_multiple * song_plan.bar_dur
    assert drops[1].source_start >= drops[0].source_end - slack, \
        "the second drop starts well inside the first"
    advance = drops[1].source_start - drops[0].source_start
    assert advance >= 0.5 * (drops[0].source_end - drops[0].source_start), \
        f"the second drop only advanced {advance:.1f}s through the source"


def test_a_build_previews_the_drop_it_runs_into(song_plan) -> None:
    slots = song_plan.slots
    for i, s in enumerate(slots):
        if s.kind == "build" and i + 1 < len(slots) and slots[i + 1].kind == "drop":
            assert s.source_start == pytest.approx(slots[i + 1].source_start), \
                "the build previews different material from the drop it leads to"


def test_the_breakdown_starts_on_a_phrase_start(song_plan, song_analysis) -> None:
    vm = arrange.source_vocal_map(song_analysis)
    bd = next(s for s in song_plan.slots if s.kind == "breakdown")
    beat_multiple = TARGET_BPM / SRC_BPM
    entry = bd.source_start * song_analysis.bar_dur / (beat_multiple * song_plan.bar_dur)
    assert not bd.cut_in_mid_phrase
    assert vm.in_gap(entry, pad=0.08)


def test_an_instrumental_falls_back_to_downbeats_and_says_so(fixture_analysis) -> None:
    """No vocal to phrase against is a fact to report, not a reason to guess."""
    p = plan(fixture_analysis, 124.0, 2.0, length=270.0)
    assert validate(p) == []
    assert p.source["phrasing"]["usable"] is False
    assert all(not s.cut_in_mid_phrase for s in p.slots)
    assert "no vocal contrast" in p.slots[0].cut_in_reason
    # the phrase-multiple guarantee does not depend on the vocal map
    for s in p.slots:
        assert s.source_bars % 4 == 0


# --------------------------------------------------------------------------
# Transition descriptors
# --------------------------------------------------------------------------


def test_every_boundary_has_a_transition_descriptor(song_plan) -> None:
    slots = song_plan.slots
    for i in range(1, len(slots)):
        assert slots[i - 1].transition_out or slots[i].transition_in, \
            f"boundary {slots[i - 1].kind} -> {slots[i].kind} at bar " \
            f"{slots[i].start_bar} has nothing to cover it"
    assert not slots[0].transition_in, "the first slot has nothing to come from"
    assert not slots[-1].transition_out, "the last slot has nothing to go to"


@pytest.mark.parametrize("form", list(FORMS))
def test_every_form_covers_every_boundary(fixture_analysis, form: str) -> None:
    p = plan(fixture_analysis, 124.0, 2.0, form_name=form, length=270.0)
    assert validate(p) == []
    assert len(p.transitions()) == len(p.slots) - 1


def test_descriptors_are_well_formed(song_plan) -> None:
    total_beats = song_plan.total_bars * BEATS_PER_BAR
    for s in song_plan.slots:
        for where, descs in (("in", s.transition_in), ("out", s.transition_out)):
            for d in descs:
                assert d["kind"] in TRANSITION_KINDS
                assert isinstance(d["beats"], int) and d["beats"] >= 1
                assert 0 <= d["beat"] <= total_beats
                assert 0.0 <= d["strength"] <= 1.0
                assert d["why"]
            # an `in` descriptor starts on the slot's downbeat; an `out` one
            # lands inside the slot's final bars
            for d in descs:
                if where == "in":
                    assert d["beat"] == s.start_bar * BEATS_PER_BAR
                else:
                    assert s.start_bar * BEATS_PER_BAR <= d["beat"] < s.end_bar * BEATS_PER_BAR
                    assert d["beat"] + d["beats"] <= s.end_bar * BEATS_PER_BAR


def test_the_kit_lands_on_the_downbeat_of_every_drop(song_plan) -> None:
    """Measured build->drop kick offset was 0.0 beats in every reference case."""
    for i, s in enumerate(song_plan.slots):
        if s.kind == "drop" and i:
            drop_in = [d for d in s.transition_in if d["kind"] == "drop_in"]
            assert len(drop_in) == 1, f"drop at bar {s.start_bar} has no drop_in"
            assert drop_in[0]["beat"] == s.start_bar * BEATS_PER_BAR
            assert drop_in[0]["beats"] == 1


def test_the_kick_leaves_on_the_downbeat_of_the_breakdown(song_plan) -> None:
    bd = next(s for s in song_plan.slots if s.kind == "breakdown")
    out = [d for d in bd.transition_in if d["kind"] == "drums_out"]
    assert len(out) == 1
    assert out[0]["beat"] == bd.start_bar * BEATS_PER_BAR
    assert out[0]["beats"] == 2, "the reference covers the hole with two beats of sweep"


def test_risers_are_short_and_sit_before_the_drop(song_plan) -> None:
    """Reference ramps into a drop are short; a 32-beat climb is not the shape."""
    slots = song_plan.slots
    for i, s in enumerate(slots[:-1]):
        if slots[i + 1].kind != "drop":
            continue
        sweeps = [d for d in s.transition_out if d["kind"] == "sweep_up"]
        assert len(sweeps) == 1, f"no riser into the drop at bar {slots[i + 1].start_bar}"
        d = sweeps[0]
        assert d["beats"] <= arrange.RISER_BEATS
        assert d["beat"] + d["beats"] == s.end_bar * BEATS_PER_BAR, \
            "the riser must land exactly on the drop"


def test_only_the_last_drop_gets_a_fill(song_plan) -> None:
    """Only ~1 in 5 reference into-drop boundaries had a fill at all."""
    slots = song_plan.slots
    last_drop = max(i for i, s in enumerate(slots) if s.kind == "drop")
    for i, s in enumerate(slots[:-1]):
        fills = [d for d in s.transition_out if d["kind"] == "fill"]
        if slots[i + 1].kind == "drop" and i + 1 == last_drop:
            assert len(fills) == 1
            assert fills[0]["beats"] == BEATS_PER_BAR, "a fill is the last bar"
            assert fills[0]["beat"] == (s.end_bar - 1) * BEATS_PER_BAR
        else:
            assert not fills, f"{s.kind} -> {slots[i + 1].kind} got an unearned fill"


def test_no_silence_before_a_drop_unless_asked_for(song_plan) -> None:
    """Nine reference drops in ten put the kick on the one with no gap."""
    assert arrange.SILENCE_BEAT_BEFORE_LAST_DROP is False
    assert not any(d["kind"] == "silence_beat"
                   for s in song_plan.slots for d in s.transition_out)

    slots = song_plan.slots
    apply_transitions(slots, silence_before_last_drop=True)
    try:
        gaps = [d for s in slots for d in s.transition_out
                if d["kind"] == "silence_beat"]
        assert len(gaps) == 1, "the switch should open exactly one hole"
        assert gaps[0]["beats"] == 1
        last_drop = next(s for s in reversed(slots) if s.kind == "drop")
        assert gaps[0]["beat"] == last_drop.start_bar * BEATS_PER_BAR - 1, \
            "the hole must be the beat immediately before the drop"
    finally:
        apply_transitions(slots)            # leave the shared fixture as found


def test_a_drop_exits_with_its_last_bar_intact(song_plan) -> None:
    """The reference exits a drop at only -2.9 dB, kick playing to the end."""
    slots = song_plan.slots
    for i, s in enumerate(slots[:-1]):
        if s.kind == "drop" and slots[i + 1].kind == "outro":
            assert s.transition_out == [], \
                "nothing should interrupt the drop's last bar before the outro"
            assert any(d["kind"] == "drums_out" and d["strength"] < 0.5
                       for d in slots[i + 1].transition_in)


# --------------------------------------------------------------------------
# The plan file
# --------------------------------------------------------------------------


OLD_SLOT_KEYS = {
    "kind", "index", "start_bar", "bars", "source_start", "source_bars",
    "source_label", "drum_pattern", "source_gain", "highpass", "lowpass",
    "sidechain", "use_bass", "use_stabs", "riser", "impact", "chops",
    "reverb_throw", "fill", "percussive_gain", "note",
}


def test_the_plan_schema_only_grew(song_plan) -> None:
    """Fields may be added. Nothing a reader already depends on may move."""
    d = song_plan.to_dict()
    for key in ("target_bpm", "bar_duration", "total_bars", "duration", "form",
                "creative_note", "source", "slots"):
        assert key in d
    for slot in d["slots"]:
        assert OLD_SLOT_KEYS <= set(slot), \
            f"missing: {sorted(OLD_SLOT_KEYS - set(slot))}"


def test_the_plan_round_trips_through_json(song_plan) -> None:
    text = json.dumps(song_plan.to_dict(), indent=2)
    back = json.loads(text)
    assert len(back["transitions"]) == len(song_plan.slots) - 1
    for t in back["transitions"]:
        assert t["out"] or t["in"]
    assert back["source"]["cut_quality"]["cuts"] == 2 * len(song_plan.slots)
    assert back["source"]["phrasing"]["usable"] is True


def test_cut_quality_is_reported_honestly(song_plan) -> None:
    q = song_plan.source["cut_quality"]
    mid = sum(s.cut_in_mid_phrase + s.cut_out_mid_phrase for s in song_plan.slots)
    assert q["mid_phrase"] == mid
    assert q["clean_fraction"] == pytest.approx(1.0 - mid / q["cuts"], abs=1e-3)


def test_validate_catches_a_missing_transition(song_plan) -> None:
    import copy
    p = copy.deepcopy(song_plan)
    p.slots[2].transition_in = []
    p.slots[1].transition_out = []
    problems = validate(p)
    assert any("transition descriptor" in m for m in problems)


def test_validate_catches_an_invented_transition_kind(song_plan) -> None:
    import copy
    p = copy.deepcopy(song_plan)
    p.slots[2].transition_in = [{"kind": "tape_stop", "beats": 4, "beat": 0}]
    assert any("unknown kind" in m for m in validate(p))
