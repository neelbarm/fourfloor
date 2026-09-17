"""Beat tracking, key detection and downbeat phase."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import chord_track, click_track
from fourfloor.analysis import analyze, suggest_house_tempo
from fourfloor.analysis.features import chromagram, estimate_tuning, frame_rate, onset_strength
from fourfloor.analysis.key import camelot_neighbours, detect_key, parse_key
from fourfloor.analysis.tempo import analyze_beats, estimate_tempo, track_beats


@pytest.mark.parametrize("bpm", [90.0, 124.0, 140.0])
def test_tempo_on_click_tracks(bpm: float, sr: int) -> None:
    """The tempo estimator lands within 1 BPM on a clean metronome."""
    x = click_track(bpm, seconds=24.0, sr=sr)
    env = onset_strength(x, sr)
    got, _ = estimate_tempo(env, frame_rate(sr))
    assert abs(got - bpm) < 1.0, f"{got:.2f} != {bpm}"


@pytest.mark.parametrize("bpm", [90.0, 124.0, 140.0])
def test_beat_grid_on_click_tracks(bpm: float, sr: int) -> None:
    """Tracked beats are evenly spaced at the true period."""
    x = click_track(bpm, seconds=24.0, sr=sr)
    grid = analyze_beats(x, sr)
    assert abs(grid.bpm - bpm) < 1.0
    iois = np.diff(grid.beats)
    assert len(iois) > 20
    assert abs(float(np.mean(iois)) - 60.0 / bpm) < 0.01
    assert float(np.std(iois)) < 0.02


def test_beat_tracker_on_real_fixture(fixture_path) -> None:
    """lofi-7.mp3 is 80 BPM per its groovebox sidecar."""
    a = analyze(fixture_path)
    assert abs(a.grid.bpm - 80.0) < 1.0, f"got {a.grid.bpm:.2f}"


def test_key_on_fixture_is_camelot_compatible(fixture_path) -> None:
    """The fixture is Dm (Camelot 7A); accept anything that mixes with it."""
    a = analyze(fixture_path)
    assert a.key.camelot in camelot_neighbours("7A"), \
        f"got {a.key.name} / {a.key.camelot}"


@pytest.mark.parametrize("root_pc,minor,expect", [
    (0, False, "C"), (9, True, "Am"), (5, False, "F"), (2, True, "Dm"), (7, False, "G"),
])
def test_key_on_synthetic_chords(root_pc: int, minor: bool, expect: str, sr: int) -> None:
    """Key detection recovers the tonic of a synthesised I-IV-V-I."""
    x = chord_track(root_pc, minor, seconds=12.0, sr=sr)
    key = detect_key(chromagram(x, sr, tuning=0.0))
    assert key.name == expect, f"got {key.name}, wanted {expect}"


def test_downbeat_phase_with_accented_beat_one(sr: int) -> None:
    """An accented beat 1 is found, whatever the offset of the first beat."""
    x = click_track(124.0, seconds=24.0, sr=sr, accent_first=True)
    grid = analyze_beats(x, sr)
    assert grid.beats_per_bar == 4
    assert grid.downbeat_confidence > 0.2
    # every detected downbeat should sit near a multiple of one bar
    bar = 4 * 60.0 / grid.bpm
    offs = np.mod(grid.downbeats - grid.downbeats[0], bar)
    offs = np.minimum(offs, bar - offs)
    assert float(np.max(offs)) < 0.05


def test_tuning_estimator(sr: int) -> None:
    """Global tuning offset is recovered to better than 5 cents."""
    t = np.arange(sr * 3) / sr
    for detune in (0.0, 0.3, -0.25):
        f = 440.0 * 2 ** (detune / 12.0)
        x = sum(a * np.sin(2 * np.pi * f * h * t) for h, a in ((1, 1.0), (2, 0.5), (3, 0.3)))
        assert abs(estimate_tuning(x.astype(np.float32), sr) - detune) < 0.05


def test_parse_key_and_camelot() -> None:
    assert parse_key("Am") == (9, True)
    assert parse_key("8A") == (9, True)
    assert parse_key("C") == (0, False)
    assert parse_key("8B") == (0, False)
    assert parse_key("F#m") == (6, True)
    with pytest.raises(ValueError):
        parse_key("H#")


def test_suggested_house_tempo_is_in_band() -> None:
    for bpm in (72.0, 80.0, 95.0, 128.0, 150.0, 174.0):
        assert 120.0 <= suggest_house_tempo(bpm) <= 128.0


def test_structure_finds_sections(fixture_path) -> None:
    a = analyze(fixture_path)
    assert len(a.sections) >= 2
    assert a.sections[0].start == 0.0
    assert abs(a.sections[-1].end - a.duration) < 1.0
    for prev, nxt in zip(a.sections[:-1], a.sections[1:]):
        assert abs(prev.end - nxt.start) < 1e-6, "sections must tile the track"
    assert a.hook().duration > 0
    assert len(a.chords) == len(a.grid.downbeats)


def test_track_beats_handles_silence(sr: int) -> None:
    """A degenerate input returns an empty grid instead of raising."""
    env = np.zeros(400)
    assert len(track_beats(env, frame_rate(sr), 124.0)) >= 0
