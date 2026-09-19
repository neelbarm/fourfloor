"""The low end: the song's own bass, and the sub that is only added if missing."""

from __future__ import annotations

import numpy as np
import pytest

from fourfloor.analysis import analyze
from fourfloor.audio import rms_db
from fourfloor.dsp import filters as FL
from fourfloor.dsp.pitch import f0_autocorr
from fourfloor.remix import RemixOptions, remix
from fourfloor.stems import separate


@pytest.mark.parametrize("hz", [41.2, 55.0, 73.4, 98.0, 146.8])
def test_pitch_tracking_finds_a_bass_note(hz: float, sr: int) -> None:
    t = np.arange(int(0.35 * sr)) / sr
    x = (np.sin(2 * np.pi * hz * t) + 0.4 * np.sin(2 * np.pi * 2 * hz * t)
         + 0.2 * np.sin(2 * np.pi * 3 * hz * t)).astype(np.float32)
    got = f0_autocorr(x, sr)
    cents = 1200 * np.log2(got / hz)
    assert abs(cents) < 20.0, f"{got:.2f} Hz for {hz:.2f} ({cents:+.0f} cents)"


def test_pitch_tracking_says_nothing_rather_than_guessing(sr: int) -> None:
    """Silence and noise must come back as 0.0, or the sub invents a bassline."""
    rng = np.random.default_rng(0)
    assert f0_autocorr(np.zeros(int(0.35 * sr), dtype=np.float32), sr) == 0.0
    noise = rng.standard_normal(int(0.35 * sr)).astype(np.float32)
    assert f0_autocorr(FL.apply(noise, "lowpass", sr, 200.0), sr, clarity=0.6) == 0.0


def test_hpss_hands_back_a_low_band_as_the_bass(trap_clip) -> None:
    a = analyze(trap_clip)
    stems = separate(a.clip.samples, a.sr, "hpss")
    assert stems.bass is not None
    assert stems.bass_name == "hpss low band"
    # it is the bottom of the mix, and the harmonic bed no longer holds it
    low = rms_db(FL.apply(stems.bass.mean(axis=1), "lowpass", a.sr, 180.0))
    high = rms_db(FL.apply(stems.bass.mean(axis=1), "highpass", a.sr, 400.0))
    assert low - high > 12.0, f"{low:.1f} vs {high:.1f} dB"


def test_asking_for_a_synth_bass_separates_without_one(trap_clip) -> None:
    a = analyze(trap_clip)
    stems = separate(a.clip.samples, a.sr, "hpss", want_bass=False)
    assert stems.bass is None
    assert stems.bass_name == "synth"


def test_the_default_bass_is_the_song_and_it_tracks_the_song(trap_clip,
                                                             tmp_path) -> None:
    """The bass layer has to *be* the record's bassline, not a version of it.

    The synthetic source plays 55, 55, 65.4 and 73.4 Hz, two bars each. A
    synthesised bass would play whatever the chord estimate said; the point of
    this change is that the remix plays what the record plays.
    """
    res = remix(trap_clip, tmp_path / "bass.mp3",
                RemixOptions(target_bpm=128.0, length="1:30", wav=False,
                             kit="none", keep_layers=True))
    assert res.bass_source == "hpss low band"
    bass = res.layers["bass"].mean(axis=1)
    sr = res.sr
    drop = next(s for s in res.session["sections"] if s["kind"] == "drop")
    seg = bass[int(drop["start"] * sr):int(drop["end"] * sr)]
    assert rms_db(seg) > -40.0, "there is no bass in the drop"

    beat = 60.0 / 128.0
    notes = []
    for i in range(0, int(len(seg) / sr / beat) - 1):
        f = f0_autocorr(seg[int(i * beat * sr):int((i + 1) * beat * sr)], sr)
        if f > 0:
            notes.append(f)
    assert len(notes) > 8, "the bass layer is not pitched"
    wanted = np.array([55.0, 65.41, 73.42, 110.0, 130.81, 146.83])
    off = [np.min(np.abs(1200 * np.log2(np.array(n) / wanted))) for n in notes]
    assert float(np.median(off)) < 60.0, \
        f"the bass is playing notes the song does not: median {np.median(off):.0f} cents"


def test_the_synth_bass_is_still_available(trap_clip, tmp_path) -> None:
    res = remix(trap_clip, tmp_path / "synthbass.mp3",
                RemixOptions(target_bpm=128.0, length="1:30", wav=False,
                             kit="none", bass="synth", keep_layers=True))
    assert res.bass_source == "synth"
    assert rms_db(res.layers["bass"]) > -45.0


def test_a_record_with_its_own_sub_gets_no_synthetic_one(sr: int) -> None:
    """The sub is a repair, not a feature; a record that has one is left alone."""
    from fourfloor.arrange import plan
    from fourfloor.house.engine import Engine, Stems

    a = analyze("fixtures/lofi-7.mp3")
    p = plan(a, 124.0, 2.0, length=60.0)
    n = int(p.total_bars * p.bar_dur * a.sr) + a.sr
    t = np.arange(n) / a.sr
    deep = np.stack([np.sin(2 * np.pi * 45.0 * t)] * 2, axis=1).astype(np.float32)
    stems = Stems(harmonic=np.zeros((n, 2), dtype=np.float32),
                  percussive=np.zeros((n, 2), dtype=np.float32),
                  bass=deep, bass_name="demucs bass")
    e = Engine(sr=a.sr, plan=p, stems=stems, chords=a.chords, beat_multiple=2.0)
    assert e._sub_under(deep) is None
