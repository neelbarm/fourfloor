"""The critic, driven with signals whose right answer is known by construction.

Real audio cannot anchor a detector test: if the critic said a render was
off beat there would be no way, here, to check. So every unit test below
builds a signal where the answer is arithmetic -- a click track is on the
grid because it was placed on the grid, a spliced sine is discontinuous
because a sample was cut out of it -- and asserts the detector agrees.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from fourfloor.critic import features as F
from fourfloor.critic import score as S

REFS_DIR = os.environ.get("FOURFLOOR_CRITIC_REFS")

SR = 44100
BPM = 128.0
BEAT = 60.0 / BPM


# ---------------------------------------------------------------------------
# synthetic material
# ---------------------------------------------------------------------------

ATTACK = 0.002
"""Seconds of attack ramp on a synthetic hit.

Not cosmetic. A burst that begins at full amplitude is a step
discontinuity, and the click detector is right to call it one -- an
instant-attack click track fires it 125 times a minute. Real drums, and
real renders, reach full level over a millisecond or two. Two
milliseconds is the difference between a fixture that models a drum and
one that models a splice.
"""


def _hit(n: int, decay: float, seed: int, tone: float = 0.0) -> np.ndarray:
    """A short percussive burst: noise (and optional tone) under an envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    env = np.exp(-t / decay)
    ramp = int(ATTACK * SR)
    env[:ramp] *= np.linspace(0.0, 1.0, ramp)
    body = rng.standard_normal(n)
    if tone:
        body = 0.4 * body + np.sin(2 * np.pi * tone * t)
    return (body * env).astype(np.float64)


def click_track(seconds: float = 24.0, bpm: float = BPM, jitter: float = 0.0,
                jitter_fraction: float = 0.0, seed: int = 7,
                per_beat: int = 2) -> np.ndarray:
    """Kick on the beat, hat between, every hit exactly on the grid.

    ``jitter`` moves ``jitter_fraction`` of the hits off the grid by that
    many seconds, which is the only difference between the "tight" and
    "off beat" fixtures. ``per_beat`` sets the subdivision. The result is
    normalised, so it never clips and the loudness fixtures are honest.
    """
    beat = 60.0 / bpm
    x = np.zeros(int(seconds * SR))
    kick = _hit(int(0.12 * SR), 0.035, seed, tone=55.0)
    hat = 0.35 * _hit(int(0.04 * SR), 0.010, seed + 1)
    rng = np.random.default_rng(seed + 2)
    step = beat / per_beat
    for i, t in enumerate(np.arange(0.0, seconds - 0.5, step)):
        if jitter_fraction and rng.random() < jitter_fraction:
            t += jitter
        start = int(t * SR)
        src = kick if i % per_beat == 0 else hat
        end = min(len(x), start + len(src))
        if end > start:
            x[start:end] += src[: end - start]
    return x * (0.8 / max(float(np.max(np.abs(x))), 1e-9))


def _measure(x: np.ndarray, bpm: float | None = BPM) -> F.Measured:
    m = F.Measured(duration=len(x) / SR)
    sp = F.spectral(F.resample_to(x, SR, F.ANALYSIS_SR))
    F.measure_groove(sp, bpm, m)
    F.measure_mush(sp, m)
    F.measure_loudness(x, m)
    return m


# ---------------------------------------------------------------------------
# groove
# ---------------------------------------------------------------------------

def test_regular_click_track_scores_high_on_groove():
    m = _measure(click_track())
    assert m.on_grid > 0.85, f"on_grid={m.on_grid:.3f}"
    assert S.score_groove(m).score > 70.0


def test_shifting_30_percent_of_the_hits_by_60ms_collapses_groove():
    tight = _measure(click_track())
    loose = _measure(click_track(jitter=0.060, jitter_fraction=0.30))
    assert loose.on_grid < tight.on_grid - 0.15
    tight_score = S.score_groove(tight).score
    loose_score = S.score_groove(loose).score
    assert loose_score < tight_score - 15.0, f"{loose_score:.1f} vs {tight_score:.1f}"


def test_tempo_is_recovered_without_a_hint():
    m = _measure(click_track(), bpm=None)
    assert m.bpm == pytest.approx(BPM, rel=0.04)


def test_grid_fraction_is_phase_free():
    """A grid offset by half a step is still a grid."""
    times = np.arange(0, 40) * 0.25 + 0.137
    frac, _phase = F.grid_fraction(times, np.ones(40), 0.25)
    assert frac == pytest.approx(1.0)


def test_grid_fraction_notices_scatter():
    rng = np.random.default_rng(3)
    times = np.sort(rng.uniform(0, 10, 400))
    frac, _ = F.grid_fraction(times, np.ones(400), 0.25)
    assert frac < 0.35


def test_windowed_grid_survives_tempo_drift_that_defeats_a_global_fold():
    """The reason the grid score is windowed at all.

    These onsets are perfectly quantised to a grid that speeds up by
    0.4 % over two minutes -- audibly tight, the way a live edit or a
    slightly imperfect stretch is. Folded against one fixed period they
    smear across the whole cycle; measured locally they are what they are.
    """
    step = 0.25
    times, t = [], 0.0
    while t < 120.0:
        times.append(t)
        t += step
        step *= 0.99999
    times = np.array(times)
    w = np.ones(len(times))
    globally, _ = F.grid_fraction(times, w, 0.25)
    locally = F.windowed_grid(times, w, 0.25)
    assert locally > 0.95
    assert globally < 0.6


# ---------------------------------------------------------------------------
# clicks
# ---------------------------------------------------------------------------

def _spliced_sine(seconds: float = 20.0, cuts: int = 9) -> np.ndarray:
    """A clean 220 Hz tone with the phase yanked every two seconds."""
    t = np.arange(int(seconds * SR)) / SR
    x = 0.5 * np.sin(2 * np.pi * 220 * t)
    for k in range(1, cuts + 1):
        i = k * 2 * SR
        if i >= len(x):
            break
        x[i:] = 0.5 * np.sin(2 * np.pi * 220 * t[: len(x) - i] + 1.9 * k)
    return x


def test_a_splice_trips_the_click_detector():
    m = F.Measured(duration=20.0)
    F.measure_clicks(_spliced_sine(), SR, m)
    assert m.extra["n_clicks"] >= 5
    assert m.click_rate > 10.0
    assert S.score_clicks(m).score < 85.0


def test_a_clean_tone_trips_nothing():
    t = np.arange(20 * SR) / SR
    m = F.Measured(duration=20.0)
    F.measure_clicks(0.5 * np.sin(2 * np.pi * 220 * t), SR, m)
    assert m.extra.get("n_clicks", 0) == 0
    assert S.score_clicks(m).score == pytest.approx(100.0)


def test_a_kick_attack_is_not_a_click():
    """The false positive the detector exists to avoid."""
    m = F.Measured(duration=24.0)
    F.measure_clicks(click_track(), SR, m)
    assert m.click_rate < 8.0


def test_clicks_at_section_cues_count_triple():
    x = _spliced_sine()
    plain, at_cue = F.Measured(duration=20.0), F.Measured(duration=20.0)
    F.measure_clicks(x, SR, plain)
    F.measure_clicks(x, SR, at_cue, cues=[2.0, 4.0, 6.0, 8.0, 10.0])
    assert at_cue.click_rate > plain.click_rate


# ---------------------------------------------------------------------------
# overlap / mush
# ---------------------------------------------------------------------------

def test_two_copies_of_a_loop_offset_by_1_3_beats_trip_the_overlap_detector():
    """The exact failure Neel described: "everything overlapping".

    1.3 beats is deliberately not a musical offset. A copy landing 1 or 2
    beats late would double the hits and still be on the grid; landing
    1.3 beats late puts a second, unrelated set of hits between them,
    which is what a mis-placed source span actually does.
    """
    one = click_track(seconds=24.0, per_beat=4)
    offset = int(1.3 * BEAT * SR)
    both = one.copy()
    both[offset:] += one[: len(one) - offset]
    both *= 0.8 / max(float(np.max(np.abs(both))), 1e-9)

    clean, mush = _measure(one), _measure(both)
    # not 2x: the onset picker merges hits inside 28 ms, and some of the
    # displaced copy lands on top of the original
    assert mush.onsets_per_beat > clean.onsets_per_beat * 1.3
    assert S.score_clarity(mush).score < S.score_clarity(clean).score - 15.0
    # and it reads as off the grid too, because half the hits now are
    assert mush.on_grid < clean.on_grid - 0.1


# ---------------------------------------------------------------------------
# loudness
# ---------------------------------------------------------------------------

def test_clipping_is_detected_and_punished():
    t = np.arange(10 * SR) / SR
    hot = np.clip(3.0 * np.sin(2 * np.pi * 110 * t), -1.0, 1.0)
    m = F.Measured(duration=10.0)
    F.measure_loudness(hot, m)
    assert m.clip_fraction > 0.01
    assert S.score_loudness(m).score < 80.0


def test_a_sane_master_scores_well_on_loudness():
    """A dense master at -11 dBFS with about 11 dB of crest: club-ready."""
    rng = np.random.default_rng(11)
    x = np.convolve(rng.standard_normal(10 * SR), np.ones(8) / 8, mode="same")
    x *= 0.95 / float(np.max(np.abs(x)))            # peak-normalised, no limiting
    m = F.Measured(duration=10.0)
    F.measure_loudness(x, m)
    assert m.clip_fraction == 0.0
    assert -14.0 < m.rms_db < -8.0
    assert 6.0 < m.crest_db < 16.0
    assert S.score_loudness(m).score > 85.0


# ---------------------------------------------------------------------------
# scoring shape
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    assert sum(S.WEIGHTS.values()) == pytest.approx(1.0)


def test_similarity_anchors_put_a_reference_at_ninety():
    from fourfloor.critic.embed import Clap, MfccRhythm

    for backend in (Clap.__dict__["anchors"], MfccRhythm.__dict__["anchors"]):
        lo, hi = backend
        assert S.score_similarity(hi, (lo, hi), "x", [], 4).score == pytest.approx(90.0)
        assert S.score_similarity(lo, (lo, hi), "x", [], 4).score == pytest.approx(0.0)


def test_the_groove_gate_caps_a_well_mastered_arrhythmic_render():
    subs = [S.SubScore("groove", "groove", 10.0, 0.30, ""),
            S.SubScore("loudness", "loudness", 95.0, 0.70, "")]
    total = sum(s.score * s.weight for s in subs)
    cap = S.GROOVE_GATE[0] + S.GROOVE_GATE[1] * 10.0
    assert total > cap and cap < 45.0


def test_verdict_names_the_worst_sub_score():
    subs = [S.SubScore("groove", "groove", 12.0, 0.30, "only 40% on grid"),
            S.SubScore("loudness", "loudness", 95.0, 0.10, "fine")]
    assert "timing" in S.verdict_for(30.0, subs, gated=True)
    assert "capped" in S.verdict_for(30.0, subs, gated=True)


# ---------------------------------------------------------------------------
# integration: the real fixture, end to end
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "fixtures" / "lofi-7.mp3"
RENDER = REPO / "examples" / "lofi-7.house.mp3"


@pytest.mark.skipif(not RENDER.exists(), reason="run `make demo` to build the example")
def test_critic_rates_the_house_render_above_its_own_source():
    """The one end-to-end claim: turning a song into house should show up.

    No references here, so the learned-similarity sub-score sits out and
    the other five are reweighted -- which keeps this test runnable
    without CLAP installed or any private audio on disk.
    """
    from fourfloor.critic import critique

    render = critique(RENDER)
    source = critique(SOURCE)

    assert render.backend == "none"
    assert render.notes, "a refs-less critique should say the similarity score was skipped"
    assert render.score > source.score
    assert render.measured.bpm == pytest.approx(124.0, rel=0.03)
    assert render.sub("groove").score > source.sub("groove").score
    assert 0.0 <= render.score <= 100.0
    assert render.verdict

    payload = render.to_dict()
    assert set(payload["sub_scores"]) == {"groove", "clarity", "clicks", "vocal", "loudness"}
    assert payload["score"] == pytest.approx(round(render.score, 1))
    json.dumps(payload)          # the --json path must be serialisable


@pytest.mark.skipif(not RENDER.exists(), reason="run `make demo` to build the example")
def test_critic_reads_the_session_cues_beside_a_render():
    from fourfloor.critic import session_for

    session = session_for(RENDER)
    assert session and session["bpm"] == pytest.approx(124.0)
    assert any(c["kind"] == "drop" for c in session["cues"])


@pytest.mark.skipif(not (REFS_DIR and Path(REFS_DIR).is_dir()),
                    reason="set FOURFLOOR_CRITIC_REFS to a folder of house remixes")
def test_similarity_runs_when_references_are_available():
    from fourfloor.critic import critique

    crit = critique(RENDER, refs=REFS_DIR)
    sim = crit.sub("similarity")
    assert sim is not None and 0.0 <= sim.score <= 100.0
    assert crit.backend in ("laion-clap", "mfcc-rhythm")
