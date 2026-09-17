"""Arrangement math and the session-file schema."""

from __future__ import annotations

import numpy as np
import pytest

from fourfloor import session
from fourfloor.arrange import (FORMS, bars_to_samples, fmt_time, parse_length,
                               plan, scale_form, validate)
from fourfloor.analysis import analyze


@pytest.fixture(scope="module")
def fixture_analysis(fixture_path):
    return analyze(fixture_path)


def test_bars_to_samples_is_exact() -> None:
    bar = 4 * 60.0 / 124.0
    assert bars_to_samples(8, bar, 44100) == int(round(8 * bar * 44100))
    # 128 bars at 124 BPM is 247.74 s
    assert bars_to_samples(128, bar, 44100) == 10925419


@pytest.mark.parametrize("form", list(FORMS))
def test_forms_are_whole_phrases(form: str) -> None:
    total = sum(b for _, b in FORMS[form])
    assert total % 8 == 0
    for _, bars in FORMS[form]:
        assert bars % 8 == 0 and bars > 0


@pytest.mark.parametrize("target", [64, 96, 128, 136, 160, 200])
def test_scale_form_hits_the_target_exactly(target: int) -> None:
    out = scale_form(FORMS["club"], target)
    assert sum(b for _, b in out) == target
    assert [k for k, _ in out] == [k for k, _ in FORMS["club"]]
    for _, bars in out:
        assert bars >= 8 and bars % 8 == 0


def test_parse_length() -> None:
    assert parse_length("4:30") == 270.0
    assert parse_length("270") == 270.0
    assert parse_length("3:05") == 185.0
    assert fmt_time(270.0) == "4:30"
    assert fmt_time(185.0) == "3:05"


@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("length", [180.0, 270.0, 360.0])
def test_plan_is_contiguous_and_on_downbeats(fixture_analysis, form: str,
                                             length: float) -> None:
    p = plan(fixture_analysis, 124.0, 2.0, form_name=form, length=length)
    assert validate(p) == []
    # every slot boundary is a whole number of bars, therefore a downbeat
    bar = 0
    for s in p.slots:
        assert s.start_bar == bar
        assert s.start_bar % 1 == 0
        bar += s.bars
    assert bar == p.total_bars
    assert p.total_bars % 8 == 0
    # the realised length is within one 8-bar phrase of what was asked for
    assert abs(p.duration - length) <= 8 * p.bar_dur


def test_plan_has_two_drops_and_a_breakdown(fixture_analysis) -> None:
    p = plan(fixture_analysis, 124.0, 2.0, form_name="club", length=270.0)
    kinds = [s.kind for s in p.slots]
    assert kinds.count("drop") == 2
    assert "breakdown" in kinds
    assert kinds[0] == "intro" and kinds[-1] == "outro"
    drops = [s for s in p.slots if s.kind == "drop"]
    assert all(s.use_bass for s in drops)
    assert drops[1].use_stabs, "the second drop should add a layer"


def test_validate_catches_a_broken_plan(fixture_analysis) -> None:
    p = plan(fixture_analysis, 124.0, 2.0, length=270.0)
    p.slots[2].start_bar += 3
    assert validate(p)


def test_session_schema(fixture_analysis) -> None:
    p = plan(fixture_analysis, 124.0, 2.0, length=270.0)
    n = int(p.total_bars * p.bar_dur * 44100)
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal((n, 2)) * 0.1).astype(np.float32)
    s = session.build(p, audio, 44100, "out.mp3", "Am", "8A", -2,
                      source={"file": "in.mp3", "bpm": 80.0},
                      tempo_plan={"stretch_ratio": 0.775})
    assert session.validate(s) == []
    assert s["bpm"] == 124.0
    assert s["first_downbeat_sec"] == 0.0
    assert s["semitone_shift"] == -2
    assert len(s["energy_per_bar"]) == s["bars"]
    assert [c["name"] for c in s["cues"]].count("drop 1") == 1
    assert s["cues"][-1]["kind"] == "end"
    assert s["cues"] == sorted(s["cues"], key=lambda c: c["time"])


def test_session_validate_rejects_bad_input(fixture_analysis) -> None:
    p = plan(fixture_analysis, 124.0, 2.0, length=270.0)
    n = int(p.total_bars * p.bar_dur * 44100)
    audio = np.zeros((n, 2), dtype=np.float32)
    s = session.build(p, audio, 44100, "out.mp3", "Am", "8A", 0, {}, {})
    assert session.validate({k: v for k, v in s.items() if k != "cues"})
    bad = dict(s, bpm=400.0)
    assert any("bpm" in m for m in session.validate(bad))
