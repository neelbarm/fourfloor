"""Playing a voice straight through, or cutting it into pieces that land.

A listener called one of these renders amazing and the other "terrible with
overlaps". The plans were structurally identical; what differed was the voice.
CAN'T SAY is played as it was sung. A triplet flow -- three syllables to the
beat -- lands a third of a beat from anything on a sixteenth lattice and will
not lock to four-on-the-floor however it is warped. These tests pin down the
measurement that tells the two apart (and that a voice with no rhythm at all
is *not* a triplet flow, which the first measure got wrong: it chopped CAN'T
SAY), and the chopping that rescues the second case.
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
    assert "played as it was sung" in why


def test_a_voice_that_will_not_lock_is_chopped() -> None:
    """A triplet flow, the case chopping exists for."""
    mode, m, why = choose_vocal(voice(TRIPLET), SR, BPM)
    assert mode == "chop", (m, why)
    assert "triplet" in why


def voice_at(times, seconds: float, hz: float = 220.0, sr: int = SR) -> np.ndarray:
    """Syllables at arbitrary times, in seconds."""
    n = int(seconds * sr)
    x = np.zeros(n, dtype=np.float32)
    ln = int(0.13 * sr)
    t = np.arange(ln) / sr
    syl = (np.sin(2 * np.pi * hz * t) * (1 + 0.5 * np.sin(2 * np.pi * 6 * t))
           * np.exp(-t / 0.05)).astype(np.float32)
    for s in times:
        a = int(round(s * sr))
        b = min(n, a + ln)
        if 0 <= a < b:
            x[a:b] += syl[: b - a]
    return x


def flow(positions_per_beat, seconds=180.0, lag=0.0, jitter=0.0, keep=1.0, seed=0):
    """A voice on the given in-beat positions, ``lag`` seconds late, each
    syllable jittered by up to ``jitter``. ``positions_per_beat`` maps each
    position to the probability a syllable is sung there."""
    rng = np.random.default_rng(seed)
    beats = np.arange(0.0, seconds - 1.0, BEAT)
    times = [b + p * BEAT for b in beats for p, prob in positions_per_beat.items()
             if rng.random() < prob * keep]
    times = np.sort(np.asarray(times) + lag + rng.uniform(-jitter, jitter, len(times)))
    return voice_at(times, seconds)


#: A sung or rapped straight line leans on the beat and the "and"; the
#: sixteenths between are the ornaments.
STRAIGHT_LINE = {0.0: 0.9, 0.25: 0.35, 0.5: 0.8, 0.75: 0.35}
TRIPLET_LINE = {0.0: 0.8, 1 / 3: 0.7, 2 / 3: 0.7}


def test_a_voice_with_no_rhythm_is_not_called_a_triplet_flow() -> None:
    """The regression that chopped CAN'T SAY.

    The old rule compared a 16th lattice with a denser six-per-beat one at the
    same +/-30 ms tolerance, so syllables placed at random scored about +0.26
    "triplet advantage" -- four times the edge -- and every real voice, rap or
    sung, was chopped. Random syllables must be played as sung.
    """
    chopped = 0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        x = voice_at(np.sort(rng.uniform(0.0, 179.0, 1000)), 180.0)
        f = A.vocal_fit(x, SR, BPM)
        assert f["advantage"] > 0.1, "the old measure still reads luck as triplet"
        mode, m, why = choose_vocal(x, SR, BPM)
        chopped += mode == "chop"
        assert 0.4 < m["triplet_share"] < 0.6, m
    assert chopped == 0, f"{chopped} of 6 rhythmless voices were chopped"


def test_a_singer_behind_the_beat_is_not_a_triplet_flow() -> None:
    """Sung pop sits 40-50 ms behind the grid (Never Be Like You read +50 ms,
    The Sweet Escape +45). That puts its sixteenths on the triplet zone of a
    fixed ruler; the lag is measured and taken out first."""
    x = flow(STRAIGHT_LINE, lag=0.047, jitter=0.012)
    f = A.vocal_fit(x, SR, BPM)
    assert f["advantage"] > 0.06, "the old rule would have chopped this"
    mode, m, why = choose_vocal(x, SR, BPM)
    assert mode == "flow", (m, why)
    assert 35.0 < m["lag_ms"] < 60.0, m
    assert m["triplet_share"] < 0.3, m


def test_a_loose_straight_flow_is_played() -> None:
    """The CAN'T SAY shape: a straight flow, a little late and a little loose."""
    x = flow(STRAIGHT_LINE, lag=0.012, jitter=0.022, seed=1)
    mode, m, why = choose_vocal(x, SR, BPM)
    assert mode == "flow", (m, why)
    assert "not a triplet flow" in why


def test_a_loose_triplet_flow_is_chopped() -> None:
    """A real triplet flow: triplet syllables, late and loose, some dropped."""
    x = flow(TRIPLET_LINE, lag=0.015, jitter=0.015, seed=2)
    mode, m, why = choose_vocal(x, SR, BPM)
    assert mode == "chop", (m, why)
    assert m["triplet_share"] > 0.8 and m["triplet_windows"] > 0.8, m


@pytest.mark.parametrize("bpm", [124.0, 126.0, 128.0, 130.0, 132.0])
def test_auto_keeps_the_loved_flow_and_chops_a_triplet_at_every_set_tempo(bpm) -> None:
    """The dress-rehearsal regression: --vocal auto chopped CAN'T SAY, whose
    flowing vocal is the render Neel loved. A CAN'T SAY-shaped line (straight,
    12 ms late, loose) must flow and a real triplet flow must chop, whatever
    tempo the set is at."""
    beat = 60.0 / bpm

    def line(positions, lag, jitter, seed):
        rng = np.random.default_rng(seed)
        times = [b + p * beat for b in np.arange(0.0, 179.0, beat)
                 for p, prob in positions.items() if rng.random() < prob]
        times = np.sort(np.asarray(times) + lag + rng.uniform(-jitter, jitter, len(times)))
        return voice_at(times, 180.0)

    mode, m, why = choose_vocal(line(STRAIGHT_LINE, 0.012, 0.022, 1), SR, bpm)
    assert mode == "flow", (bpm, m, why)
    mode, m, why = choose_vocal(line(TRIPLET_LINE, 0.015, 0.015, 2), SR, bpm)
    assert mode == "chop", (bpm, m, why)


def test_the_measure_does_not_depend_on_the_tempo() -> None:
    """Chance is one half at any tempo: a random voice reads about the same at
    100 and 150 BPM, where the old fits' chance levels were 0.40 and 0.60."""
    rng = np.random.default_rng(9)
    x = voice_at(np.sort(rng.uniform(0.0, 179.0, 1000)), 180.0)
    for bpm in (100.0, 150.0):
        mode, m, _why = choose_vocal(x, SR, bpm)
        assert mode == "flow" and 0.4 < m["triplet_share"] < 0.6, (bpm, m)


def test_real_stem_calibration_is_what_the_thresholds_split() -> None:
    """The numbers measured on the real separated stems (house/vocal.py's
    table): none of them is a triplet flow by more than luck gives, so all of
    them -- CAN'T SAY first -- are played as sung."""
    from fourfloor.house import vocal as V

    measured = {"CAN'T SAY": (0.472, 0.308), "Body": (0.546, 0.636),
                "You Belong With Me": (0.509, 0.444),
                "Never Be Like You": (0.485, 0.423), "The Sweet Escape": (0.385, 0.296),
                "E85": (0.438, 0.316), "Cold Shoulder": (0.468, 0.400)}
    for name, (share, windows) in measured.items():
        assert not (share >= V.TRIPLET_SHARE and windows >= V.TRIPLET_WINDOWS), name


#: The same stems separated again at other set tempos (demucs --shifts 0,
#: warped by the real pipeline, analysis only): (triplet_share, windows).
SWEEP = {
    "CAN'T SAY": {124: (0.444, 0.346), 126: (0.473, 0.385), 130: (0.492, 0.538),
                  132: (0.450, 0.192)},
    "Body": {124: (0.536, 0.545), 126: (0.545, 0.524), 130: (0.565, 0.545),
             132: (0.530, 0.591)},
    "Never Be Like You": {124: (0.541, 0.615), 126: (0.497, 0.519),
                          130: (0.506, 0.462), 132: (0.515, 0.444)},
    "Cold Shoulder": {124: (0.457, 0.483), 126: (0.465, 0.400), 130: (0.471, 0.414),
                      132: (0.486, 0.533)},
    "E85": {124: (0.405, 0.211), 126: (0.427, 0.278), 130: (0.370, 0.389),
            132: (0.392, 0.222)},
    "The Sweet Escape": {124: (0.383, 0.357), 126: (0.373, 0.214),
                         130: (0.353, 0.214), 132: (0.383, 0.250)},
}


def test_cant_say_flows_at_every_set_tempo() -> None:
    """The loved recipe's vocal is played as sung whatever tempo the set is at,
    with room to spare on the share (luck alone reaches 0.60)."""
    from fourfloor.house import vocal as V

    for bpm, (share, windows) in SWEEP["CAN'T SAY"].items():
        assert share < V.TRIPLET_SHARE - 0.1, bpm
        assert not (share >= V.TRIPLET_SHARE and windows >= V.TRIPLET_WINDOWS), bpm


def test_body_is_not_separable_from_sung_pop_on_this_evidence() -> None:
    """Why --vocal auto does not chop *Body*: at 124 BPM Flume's sung vocal
    (which its remixer played continuously) reads as triplet as Body does at
    128, so any threshold that chopped Body would chop it too. Body is chopped
    on request (--vocal chop); auto plays it as sung."""
    from fourfloor.house import vocal as V

    body = SWEEP["Body"]
    flume = SWEEP["Never Be Like You"][124]
    assert min(s for s, _w in body.values()) <= flume[0] + 0.01
    assert min(w for _s, w in body.values()) <= flume[1]
    for name, by_tempo in SWEEP.items():
        for bpm, (share, windows) in by_tempo.items():
            assert not (share >= V.TRIPLET_SHARE and windows >= V.TRIPLET_WINDOWS), (name, bpm)


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
