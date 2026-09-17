"""The whole pipeline: remix the fixture and measure the result."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fourfloor import session
from fourfloor.analysis import analyze
from fourfloor.audio import decode, rms_db
from fourfloor.preview import write_preview
from fourfloor.remix import RemixOptions, remix

TARGET_BPM = 124.0


@pytest.fixture(scope="module")
def remixed(fixture_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("remix") / "fixture.house.mp3"
    opts = RemixOptions(target_bpm=TARGET_BPM, length="3:00", stems="hpss")
    return remix(fixture_path, out, opts)


def test_outputs_exist(remixed) -> None:
    for key in ("mp3", "wav", "session", "plan"):
        assert remixed.paths[key].is_file(), f"missing {key}"
    assert remixed.paths["mp3"].stat().st_size > 50_000


def test_duration_matches_the_plan(remixed) -> None:
    clip = decode(remixed.paths["wav"])
    assert abs(clip.duration - remixed.plan.duration) < 0.2
    assert abs(clip.duration - 180.0) < 8 * remixed.plan.bar_dur


def test_levels_are_mastered_not_clipped(remixed) -> None:
    clip = decode(remixed.paths["wav"])
    peak_db = 20 * np.log10(float(np.max(np.abs(clip.samples))))
    assert peak_db <= -0.9, f"peak {peak_db:.2f} dB is too hot"
    assert peak_db > -3.0, f"peak {peak_db:.2f} dB is too quiet"
    assert not np.any(np.abs(clip.samples) >= 1.0)
    assert -12.0 < rms_db(clip.samples) < -6.0


def test_output_beat_tracks_at_the_target_tempo(remixed) -> None:
    """Re-analysing the remix must read back the tempo we synthesised it at."""
    clip = decode(remixed.paths["wav"])
    a = analyze(remixed.paths["wav"], clip=clip)
    assert abs(a.grid.bpm - TARGET_BPM) < 1.0, f"read back {a.grid.bpm:.2f}"


def test_kick_dominates_the_low_band_in_drops(remixed) -> None:
    """40-80 Hz should carry a large share of the power where the kick plays."""
    clip = decode(remixed.paths["wav"])
    mono, sr = clip.mono, clip.sr
    drops = [s for s in remixed.session["sections"] if s["kind"] == "drop"]
    assert drops
    for d in drops:
        seg = mono[int(d["start"] * sr): int(d["end"] * sr)]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        low = spec[(freqs >= 40) & (freqs < 80)].sum() / spec.sum()
        assert low > 0.2, f"only {low:.1%} of drop power in 40-80 Hz"


def test_sidechain_pumping_is_visible_at_the_beat_period(remixed) -> None:
    """The RMS envelope of a drop correlates with itself at one beat."""
    from fourfloor.analysis.features import rms_envelope

    clip = decode(remixed.paths["wav"])
    sr = clip.sr
    drop = next(s for s in remixed.session["sections"] if s["kind"] == "drop")
    seg = clip.mono[int(drop["start"] * sr): int(drop["end"] * sr)]
    # A 1024-sample window is 23 ms, which is one cycle of a 43 Hz sub: the
    # envelope it produces wobbles at the sub's own frequency hard enough to
    # bury the thing being measured. Now that the low end is the song's own
    # bass rather than a bright synthesised saw, the window has to be long
    # enough to average a few cycles of it.
    env = rms_envelope(seg, hop=256, win=4096)
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac /= ac[0]
    lag = int(round(remixed.session["beat_duration_sec"] * sr / 256))
    assert lag < len(ac) // 2
    assert ac[lag] > 0.5, f"no pumping at the beat period (r={ac[lag]:.2f})"
    assert ac[lag] > ac[lag // 2], "the period should be a beat, not half a beat"


def test_session_file_is_valid_and_matches_the_audio(remixed) -> None:
    data = json.loads(remixed.paths["session"].read_text())
    assert session.validate(data) == []
    assert data["bpm"] == pytest.approx(TARGET_BPM, abs=0.01)
    assert data["bars"] == remixed.plan.total_bars
    assert data["cues"][0]["time"] == 0.0
    assert data["source"]["file"].endswith(".mp3")


def test_plan_file_records_every_decision(remixed) -> None:
    data = json.loads(remixed.paths["plan"].read_text())
    assert data["total_bars"] == remixed.plan.total_bars
    assert len(data["slots"]) == len(remixed.plan.slots)
    for slot in data["slots"]:
        for key in ("kind", "bars", "source_start", "drum_pattern", "note"):
            assert key in slot
        assert slot["note"], "every slot should explain itself"


def test_preview_html_is_self_contained(remixed, tmp_path) -> None:
    out = write_preview(remixed.paths["mp3"], remixed.session, out=tmp_path / "preview.html")
    html = out.read_text()
    assert out.stat().st_size > 20_000
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert "cdn" not in html.lower()
    assert remixed.paths["mp3"].name in html
    assert "SESSION = {" in html and "WAVE = [" in html


def test_key_shift_is_applied(fixture_path, tmp_path) -> None:
    """--key moves the tonic and the session file reports the shift."""
    src = analyze(fixture_path, keep_audio=False)
    want_pc = (src.key.tonic + 3) % 12
    from fourfloor.analysis.key import PITCH_NAMES
    target = PITCH_NAMES[want_pc] + ("m" if src.key.is_minor else "")
    out = tmp_path / "shift.mp3"
    res = remix(fixture_path, out, RemixOptions(target_bpm=124.0, key=target,
                                                length="1:30"))
    assert res.semitones == 3
    assert res.target_key.name == target
    assert res.session["semitone_shift"] == 3
