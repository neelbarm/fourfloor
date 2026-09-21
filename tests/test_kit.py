"""Sampling a drum kit out of a record, and playing it back on the grid."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fourfloor import kit as K
from fourfloor.analysis import alignment as A
from fourfloor.analysis import analyze
from fourfloor.audio import decode, rms_db
from fourfloor.remix import RemixOptions, remix

from conftest import house_track

BPM = 128.0


@pytest.fixture(scope="module")
def built_kit(house_clip, monkeypatch_module):
    """A kit built from the synthetic house record.

    Demucs is stubbed out: the synthetic record is already nothing but drums, so
    running a source-separation model over it would cost forty seconds to hand
    back what went in. Everything after the separation -- finding the loop,
    warping it, trimming, naming, storing -- is the real code path.
    """
    from fourfloor.house.engine import Stems

    def fake(x, sr=44100, **kw):
        return Stems(harmonic=np.zeros_like(x), percussive=x, source_name="stub")

    monkeypatch_module.setattr("fourfloor.stems.separate_demucs", fake)
    return K.build(house_clip, name="testkit")


def test_the_loop_is_exactly_eight_bars_at_the_canonical_tempo(built_kit) -> None:
    want = int(round(built_kit.bars * 4 * 60.0 / K.CANONICAL_BPM * built_kit.sr))
    assert built_kit.bars == 8
    assert len(built_kit.loop) == want
    assert abs(built_kit.source_bpm - BPM) < 1.5
    assert float(np.max(np.abs(built_kit.loop))) == pytest.approx(0.89, abs=0.02)


def test_the_loop_sits_on_its_own_grid(built_kit) -> None:
    """A kit that is off its own grid puts every remix built on it off theirs."""
    r = A._layer_report(built_kit.loop, built_kit.sr, K.CANONICAL_BPM, 0.0)
    assert r["median_ms"] < 8.0, r
    assert r["within_20ms"] > 0.9, r
    assert r["beat_alignment"] == "grid", r


def test_a_kick_was_found_on_every_beat(built_kit) -> None:
    assert len(built_kit.kick_beats) == built_kit.bars * 4
    spacing = np.diff(built_kit.kick_beats)
    assert np.allclose(spacing, 60.0 / K.CANONICAL_BPM, atol=0.02)


def test_it_stretches_to_a_target_tempo_without_losing_a_sample(built_kit) -> None:
    for bpm in (120.0, 124.0, 128.0, 132.0):
        loop, kicks = built_kit.at_bpm(bpm)
        want = int(round(built_kit.bars * 4 * 60.0 / bpm * built_kit.sr))
        assert len(loop) == want, f"{bpm}: {len(loop)} != {want}"
        assert np.allclose(np.diff(kicks), 60.0 / bpm, atol=0.03)


def test_it_round_trips_through_the_store(built_kit) -> None:
    folder = K.kits_home() / "testkit"
    assert (folder / "loop.wav").is_file()
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["bars"] == 8 and meta["canonical_bpm"] == K.CANONICAL_BPM
    again = K.load("testkit")
    assert again.bars == built_kit.bars
    assert len(again.loop) == len(built_kit.loop)
    assert again.kick_beats == pytest.approx(built_kit.kick_beats, abs=1e-3)


def test_resolve_picks_the_newest_a_name_or_nothing(built_kit) -> None:
    assert K.resolve(None).name == "testkit"
    assert K.resolve("testkit").name == "testkit"
    assert K.resolve("none") is None
    with pytest.raises(FileNotFoundError):
        K.resolve("not-a-kit")


def test_a_bad_name_is_refused(house_clip, built_kit) -> None:
    with pytest.raises(ValueError):
        K.build(house_clip, name="../../etc/passwd")


def test_a_remix_plays_the_kit_and_stays_on_the_grid(built_kit, trap_clip,
                                                     tmp_path) -> None:
    """The point of all of it: a real loop, still exactly on the grid."""
    res = remix(trap_clip, tmp_path / "with-kit.mp3",
                RemixOptions(target_bpm=BPM, length="1:30", wav=False,
                             kit="testkit", keep_layers=True))
    assert res.kit_name == "testkit"
    rep = A.alignment_report(res.audio, res.sr, BPM, 0.0,
                             source_stem=res.layers["source_perc"],
                             kit_layer=res.layers["kit"], spans=res.source_spans)
    assert rep["kit"]["median_ms"] < 10.0, rep["kit"]
    # The source's own drums have to be *gone*, not merely turned down: a
    # second drummer playing the source's rhythm under a record's is the
    # other thing a listener hears as two rhythms at once. Measured, because
    # reading the arrangement code is not evidence.
    over = A.overlap_report({"source_drums": res.layers["source_drums"]},
                            res.sr, BPM)
    assert over["layers"]["source_drums"]["silent"], over
    assert rep["kit"]["beat_alignment"] == "grid", rep["kit"]
    assert rep["source"]["bar_phase"] == 0
    assert rep["spans"]["max_concurrent"] == 1


def test_the_synth_kit_is_still_there_when_no_kit_is_wanted(trap_clip,
                                                            tmp_path) -> None:
    res = remix(trap_clip, tmp_path / "synth-kit.mp3",
                RemixOptions(target_bpm=BPM, length="1:30", wav=False, kit="none"))
    assert res.kit_name is None
    assert res.metrics["kick_count"] > 100


def test_drums_db_moves_the_drum_bus_and_nothing_else(built_kit, trap_clip,
                                                      tmp_path) -> None:
    """"Drums a little quieter" has to mean the drums and only the drums.

    Two renders of the same source differing by one flag: the drum bus moves by
    what was asked for, and every other bus comes back sample for sample the
    same. A trim that also moved the vocal would be a master fader with a
    misleading name.
    """
    def render(db: float):
        return remix(trap_clip, tmp_path / f"db{db}.mp3",
                     RemixOptions(target_bpm=BPM, length="1:00", wav=False,
                                  kit="testkit", bass="source", drums_db=db,
                                  keep_layers=True))

    flat, quiet = render(0.0), render(-6.0)
    from fourfloor.audio import rms_db

    moved = rms_db(flat.layers["kit"]) - rms_db(quiet.layers["kit"])
    assert moved == pytest.approx(6.0, abs=0.15), f"drum bus moved {moved:.2f} dB"
    for layer in ("source", "harmonic", "bass", "source_perc"):
        assert np.allclose(flat.layers[layer], quiet.layers[layer], atol=1e-6), \
            f"{layer} moved too"


def test_a_sampled_loop_sits_below_its_own_reinforcement(built_kit, trap_clip,
                                                         tmp_path) -> None:
    """The trim comes off the loop, not off the sub kick underneath it.

    What was too loud on the records Neel listened to was the sampled record's
    mids and highs against the vocal, not the weight below them. Taking the
    whole bus down would have removed the part that was right, so the bus RMS
    -- which the sub kick dominates -- moves much less than the loop does.
    """
    from fourfloor.house.engine import LOOP_TRIM_DB

    assert -3.0 < LOOP_TRIM_DB < -1.0
    res = remix(trap_clip, tmp_path / "trim.mp3",
                RemixOptions(target_bpm=BPM, length="1:00", wav=False,
                             kit="testkit", bass="none", keep_layers=True))
    bus = rms_db(res.layers["kit"])
    assert -30.0 < bus < -5.0, bus
