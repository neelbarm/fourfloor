"""Verification: is this candidate the same song as that remix?

Everything here is synthetic and nothing here touches Demucs or the network. A
"song" is a melody of harmonic tones; its "remix" is that melody run through
this project's own phase vocoder at a different tempo, resampled to a different
pitch, with drums laid on top -- which is what a remixer does, minus the taste.
An unrelated song is a different melody in a different key.

The claim being pinned is the one the whole pipeline rests on: the remix scores
far above the accept threshold against its own original, the unrelated one
scores below the reject threshold, and the tempo ratio and semitone shift that
come back are the ones that were applied.
"""

from __future__ import annotations

import numpy as np
import pytest

from fourfloor.refs import verify as V

SR = 44100


# ---------------------------------------------------------------------------
# synthetic music
# ---------------------------------------------------------------------------

def tones(midi: list[int], beat: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    """A melody: one harmonic tone per note, with an attack and a release."""
    rng = np.random.default_rng(seed)
    out = []
    for m in midi:
        n = int(beat * sr)
        t = np.arange(n) / sr
        f = 440.0 * 2 ** ((m - 69) / 12.0)
        sig = np.zeros(n)
        for h, a in enumerate((1.0, 0.6, 0.35, 0.2, 0.12), start=1):
            if f * h < sr / 2:
                sig += a * np.sin(2 * np.pi * f * h * t + 0.05 * np.sin(2 * np.pi * 5.2 * t))
        env = np.clip(np.minimum(t / 0.02, (beat - t) / 0.06), 0.0, 1.0)
        out.append(sig * env)
    x = np.concatenate(out) + 0.002 * rng.standard_normal(sum(int(beat * sr) for _ in midi))
    return (x / max(float(np.max(np.abs(x))), 1e-9) * 0.7).astype(np.float32)


def drums(seconds: float, bpm: float, sr: int = SR, seed: int = 1) -> np.ndarray:
    """Four on the floor with an offbeat hat: what a house remix adds."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = np.zeros(n)
    beat = 60.0 / bpm
    t = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * 55.0 * np.exp(-t * 8) * t * 6) * np.exp(-t / 0.05)
    hat = rng.standard_normal(int(0.05 * sr)) * np.exp(-np.arange(int(0.05 * sr)) / sr / 0.01)
    i = 0
    while int(i * beat * sr) + len(kick) < n:
        p = int(i * beat * sr)
        x[p:p + len(kick)] += kick * 0.9
        h = int((i + 0.5) * beat * sr)
        if h + len(hat) < n:
            x[h:h + len(hat)] += hat * 0.3
        i += 1
    return x.astype(np.float32)


def remix_of(song: np.ndarray, tempo: float, semitones: int,
             sr: int = SR) -> np.ndarray:
    """``song`` stretched, repitched and played over drums.

    The stretch is the project's own phase vocoder; the pitch shift is a
    resample, and the vocoder rate is divided by it so the *net* tempo change is
    exactly ``tempo``. That is the same pair of operations a remixer's DAW does.
    """
    from fourfloor.dsp import phasevocoder as PV

    ratio = 2 ** (semitones / 12.0)
    y = np.asarray(PV.time_stretch(song.astype(np.float32), tempo / ratio),
                   dtype=np.float64)
    idx = np.arange(0, len(y) - 1, ratio)
    z = np.interp(idx, np.arange(len(y)), y)
    z = z / max(float(np.max(np.abs(z))), 1e-9) * 0.7
    beat = drums(len(z) / sr + 1.0, 124.0)
    n = min(len(z), len(beat))
    return (z[:n] + beat[:n] * 0.8).astype(np.float32)


HOOK = [60, 62, 64, 67, 64, 62, 60, 57, 60, 62, 64, 65, 64, 60, 59, 60]
#: A different record: different notes, different order, different rhythm --
#: not this melody transposed, which would be the same song in another key.
OTHER = [71, 66, 69, 61, 63, 68, 70, 62, 66, 73, 61, 64, 69, 62, 71, 63]


@pytest.fixture(scope="module")
def pieces():
    """An original, a remix of it, and a record that has nothing to do with it."""
    song = tones(HOOK * 3, 0.42)
    remix = remix_of(song[int(2.0 * SR):int(14.0 * SR)], tempo=1.15, semitones=2)
    other = tones(OTHER * 3, 0.36, seed=7)
    return song, remix, other


def fp(x: np.ndarray, bpm: float, kind: str = "remix",
       source: str = "vocals") -> V.Fingerprint:
    f = V.Fingerprint(kind=kind, bpm=bpm, source=source, seconds=len(x) / SR)
    f.chroma = V.chroma_sequence(x, SR)
    return f


# ---------------------------------------------------------------------------
# the features
# ---------------------------------------------------------------------------

def test_a_chroma_sequence_is_centred_and_unit_length() -> None:
    seq = V.chroma_sequence(tones(HOOK, 0.4), SR)
    assert seq.shape[0] == 12
    assert seq.shape[1] > 10
    # centred before it is normalised, so the mean is near zero rather than
    # exactly zero -- which is the point: unrelated material now scores near 0
    # instead of at the 0.7 two non-negative vectors share for free
    assert abs(float(seq.mean())) < 0.05
    assert np.allclose(np.linalg.norm(seq, axis=0), 1.0, atol=1e-6)


def test_a_transposed_melody_is_the_same_sequence_rotated() -> None:
    """Which is why a pitch shift is a search over twelve rotations."""
    up = [m + 3 for m in HOOK]
    a = V.chroma_sequence(tones(HOOK, 0.4), SR)
    b = V.chroma_sequence(tones(up, 0.4), SR)
    n = min(a.shape[1], b.shape[1])
    aligned = float(np.mean(np.sum(np.roll(b[:, :n], -3, axis=0) * a[:, :n], axis=0)))
    plain = float(np.mean(np.sum(b[:, :n] * a[:, :n], axis=0)))
    assert aligned > 0.7
    assert aligned > plain + 0.4


def test_the_window_picker_finds_the_part_with_singing_in_it() -> None:
    quiet = drums(20.0, 124.0)
    voice = tones(HOOK, 0.5)
    x = np.concatenate([quiet, voice + drums(len(voice) / SR, 124.0)[:len(voice)]])
    start = V.pick_window(x, SR, seconds=6.0)
    assert start > 15.0


def test_phrases_are_runs_of_singing_with_the_breaths_bridged() -> None:
    beat = np.concatenate([tones([60], 1.0), np.zeros(int(0.1 * SR), dtype=np.float32),
                           tones([62], 1.0), np.zeros(int(1.5 * SR), dtype=np.float32),
                           tones([64], 0.8)])
    duty, found = V.phrases(beat, SR)
    assert 0.4 < duty < 0.95
    assert len(found) == 2                     # the 100 ms gap is a breath, not a stop
    assert found[0] > 1.5 and found[1] < 1.2


# ---------------------------------------------------------------------------
# the alignment
# ---------------------------------------------------------------------------

def test_dtw_finds_a_free_ended_subsequence() -> None:
    cost = np.ones((4, 12))                    # a free diagonal at columns 5..8
    cost[0, 5] = cost[1, 6] = cost[2, 7] = cost[3, 8] = 0.0
    mean, end, cells = V.subsequence_dtw(cost)
    assert mean == pytest.approx(0.0, abs=1e-9)
    assert end == 8
    assert cells == 4


def test_dtw_on_a_flat_cost_matrix_is_that_cost() -> None:
    mean, _end, _cells = V.subsequence_dtw(np.full((6, 20), 0.4))
    assert mean == pytest.approx(0.4, abs=1e-9)


def test_a_remix_matches_the_song_it_was_made_from(pieces) -> None:
    song, remix, _other = pieces
    m = V.compare(fp(song, 120.0, "original"), fp(remix, 138.0))
    assert m.score > V.ACCEPT
    assert m.margin > V.MARGIN
    assert m.verdict == "match"
    assert m.semitones == 2                    # the two semitones it was shifted by
    assert m.beat_relation == 1.0


def test_an_unrelated_record_does_not_match(pieces) -> None:
    song, _remix, other = pieces
    m = V.compare(fp(song, 120.0, "original"), fp(other, 138.0))
    assert m.verdict == "reject"
    assert m.score < V.ACCEPT


def test_the_thresholds_leave_room_between_a_true_and_a_false_pair(pieces) -> None:
    """The gap is the whole design, so it is worth asserting it exists."""
    song, remix, other = pieces
    true = V.compare(fp(song, 120.0, "original"), fp(remix, 138.0))
    false = V.compare(fp(song, 120.0, "original"), fp(other, 138.0))
    assert true.score - false.score > 0.15
    assert true.margin > false.margin + 0.1


def test_a_half_time_reading_has_to_beat_the_straight_one_by_a_margin() -> None:
    assert V.RELATION_COST > 0.0
    assert V.REVIEW < V.ACCEPT
    assert V.REVIEW_MARGIN < V.MARGIN
    assert V.ACCEPT_CHROMA > V.ACCEPT       # an instrumental match claims less


def test_an_instrumental_match_is_marked_lower_confidence(pieces) -> None:
    song, remix, _other = pieces
    m = V.compare(fp(song, 120.0, "original", source="mix"),
                  fp(remix, 138.0, source="mix"))
    assert m.method == "chroma"
    assert m.confidence == "low"
    assert "weaker evidence" in m.note


def test_comparing_something_with_no_features_says_so() -> None:
    empty = V.Fingerprint(bpm=120.0)
    with pytest.raises(V.VerifyError):
        V.compare(empty, fp(tones(HOOK, 0.4), 124.0))


# ---------------------------------------------------------------------------
# fingerprinting, with demucs stood in for
# ---------------------------------------------------------------------------

@pytest.fixture
def no_demucs(monkeypatch):
    """Demucs replaced by "the vocals are the whole file", which is true here."""
    import fourfloor.stems as stems_mod
    from fourfloor.house.engine import Stems

    calls = []

    def fake(x, sr=SR, *_a, **_k):
        calls.append(len(x))
        return Stems(vocals=x, harmonic=x, percussive=np.zeros_like(x),
                     source_name="fake", bass=None, bass_name="none")

    monkeypatch.setattr(stems_mod, "separate_demucs", fake)
    return calls


def test_a_fingerprint_measures_the_vocal_and_caches_it(tmp_path, no_demucs) -> None:
    from fourfloor.audio import write_wav

    path = write_wav(tmp_path / "song.wav", tones(HOOK * 2, 0.45), SR)
    first = V.fingerprint(path, kind="remix", cache_dir=tmp_path / "cache")
    assert first.source == "vocals"
    assert first.chroma.shape[0] == 12
    assert first.duty > 0.5
    assert first.phrase_median > 0.0
    assert len(no_demucs) == 1

    again = V.fingerprint(path, kind="remix", cache_dir=tmp_path / "cache")
    assert len(no_demucs) == 1                 # the model did not run twice
    assert np.allclose(again.chroma, first.chroma)
    assert again.bpm == first.bpm


def test_a_silent_vocal_stem_falls_back_to_the_mix(tmp_path, monkeypatch) -> None:
    import fourfloor.stems as stems_mod
    from fourfloor.audio import write_wav
    from fourfloor.house.engine import Stems

    monkeypatch.setattr(stems_mod, "separate_demucs",
                        lambda x, sr=SR, *a, **k: Stems(
                            vocals=np.zeros_like(x), harmonic=x,
                            percussive=np.zeros_like(x), source_name="fake",
                            bass=None, bass_name="none"))
    path = write_wav(tmp_path / "instrumental.wav", tones(HOOK * 2, 0.45), SR)
    got = V.fingerprint(path, kind="remix")
    assert got.source == "mix"
    assert got.chroma.shape[1] > 10


def test_without_demucs_nothing_is_separated_at_all(tmp_path, no_demucs) -> None:
    from fourfloor.audio import write_wav

    path = write_wav(tmp_path / "song.wav", tones(HOOK * 2, 0.45), SR)
    got = V.fingerprint(path, kind="remix", demucs=False)
    assert got.source == "mix"
    assert not no_demucs
