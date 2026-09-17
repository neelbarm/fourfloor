"""Vocal-activity envelopes, phrase gaps and phrase-aware cut points.

Everything here runs on synthetic sources with gaps we placed ourselves, so an
assertion that "the cut landed in a gap" is checkable rather than a matter of
taste.
"""

from __future__ import annotations

import numpy as np
import pytest

from fourfloor.analysis import structure as S

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT            # 2.0 s


def band_noise(seconds: float, lo: float = 300.0, hi: float = 3000.0,
               sr: int = SR, seed: int = 7) -> np.ndarray:
    """Band-limited noise: a stand-in for a voice.

    Noise confined to a band reads as *tonal* to the flatness measure the
    envelope uses -- the bins outside the band are empty, so the geometric mean
    over the whole vocal range collapses -- which is exactly how a sung note
    reads and exactly how a broadband click does not.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec[(freqs < lo) | (freqs > hi)] = 0.0
    y = np.fft.irfft(spec, n).astype(np.float32)
    return y / max(float(np.max(np.abs(y))), 1e-9)


def phrased_vocal(pattern: list[tuple[float, bool]], sr: int = SR) -> np.ndarray:
    """A vocal track from ``(seconds, singing?)`` chunks, with soft edges.

    The 20 ms ramps matter: a hard gate would put a broadband click at every
    phrase edge, which is the one artefact the envelope is meant to find.
    """
    out = []
    for seconds, singing in pattern:
        n = int(seconds * sr)
        if not singing:
            out.append(np.zeros(n, dtype=np.float32))
            continue
        seg = band_noise(seconds, sr=sr, seed=len(out) + 1)[:n]
        ramp = max(1, int(0.02 * sr))
        env = np.ones(n, dtype=np.float32)
        env[:ramp] = np.linspace(0, 1, ramp)
        env[-ramp:] = np.linspace(1, 0, ramp)
        out.append(seg * env)
    return np.concatenate(out).astype(np.float32)


def sing_3_rest_1(bars: int = 32, sr: int = SR) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Three bars of singing, one bar of silence, repeated.

    Returns the audio and the spans where the silence sits, so a test can ask
    whether a cut landed in one without re-deriving them from the signal.
    """
    pattern, gaps, t = [], [], 0.0
    for b in range(bars):
        singing = (b % 4) != 3
        pattern.append((BAR, singing))
        if not singing:
            gaps.append((t, t + BAR))
        t += BAR
    return phrased_vocal(pattern, sr), gaps


def click(seconds: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    """A metronome at ``BPM`` -- broadband, so the envelope should ignore it."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = np.zeros(n, dtype=np.float32)
    t = np.arange(int(0.05 * sr)) / sr
    tick = rng.standard_normal(len(t)).astype(np.float32) * np.exp(-t / 0.004)
    i = 0
    while int(i * BEAT * sr) + len(tick) < n:
        x[int(i * BEAT * sr):int(i * BEAT * sr) + len(tick)] += tick * 0.6
        i += 1
    return x


def stereo(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x], axis=1).astype(np.float32)


@pytest.fixture(scope="module")
def phrased():
    """A 64 s source: metronome plus a vocal that rests every fourth bar."""
    voc, gaps = sing_3_rest_1()
    mix = voc + click(len(voc) / SR) * 0.35
    return stereo(mix / max(float(np.max(np.abs(mix))), 1e-9)), gaps


def test_envelope_tracks_the_vocal_not_the_click(phrased) -> None:
    mix, gaps = phrased
    env, fps = S.vocal_envelope(mix, SR)
    sing = np.mean([env[int((g[0] - BAR) * fps):int(g[0] * fps)].mean() for g in gaps])
    rest = np.mean([env[int((g[0] + 0.3) * fps):int((g[1] - 0.3) * fps)].mean() for g in gaps])
    assert sing > 4 * rest, f"singing {sing:.3f} vs rest {rest:.3f}: the click leaks through"


def test_gaps_are_found_where_we_put_them(phrased) -> None:
    mix, gaps = phrased
    vm = S.vocal_map(mix, SR)
    assert vm.usable
    for a, b in gaps:
        mid = 0.5 * (a + b)
        assert vm.in_gap(mid), f"no gap found over the silent bar at {a:.1f}-{b:.1f}s"
    # and nothing spurious: every detected gap overlaps one we placed
    for g in vm.gaps:
        assert any(g.start < b and a < g.end for a, b in gaps), \
            f"phantom gap at {g.start:.2f}-{g.end:.2f}s"


def test_short_breaths_are_not_phrase_gaps() -> None:
    """A 150 ms stop is a consonant, not a place to cut."""
    voc = phrased_vocal([(4.0, True), (0.15, False), (4.0, True),
                         (0.6, False), (4.0, True)])
    vm = S.vocal_map(stereo(voc), SR)
    assert not vm.in_gap(4.08), "a 150 ms stop was taken for a phrase end"
    assert vm.in_gap(8.45), "a 600 ms rest was missed"


def test_a_click_track_has_no_usable_phrasing() -> None:
    """An instrumental must report itself unusable rather than invent gaps."""
    vm = S.vocal_map(stereo(click(30.0)), SR)
    assert not vm.usable
    # and snapping still works -- it just falls back to the nearest downbeat
    downbeats = np.arange(0, 30.0, BAR)
    cut = S.snap_cut(5.3, downbeats, vm, BAR, role="entry")
    assert cut.time in set(downbeats)
    assert not cut.mid_phrase
    assert "no vocal contrast" in cut.reason


def test_snap_moves_a_cut_off_a_word_and_onto_a_downbeat(phrased) -> None:
    mix, gaps = phrased
    vm = S.vocal_map(mix, SR)
    downbeats = np.arange(0, 64.0, BAR)
    # bar 12 is sung; bar 11 is one of the rests, one bar away
    boundary = 12 * BAR
    cut = S.snap_cut(boundary, downbeats, vm, BAR, role="entry")
    assert cut.time in set(downbeats), "a cut left the downbeat grid"
    assert abs(cut.moved_bars) <= S.SNAP_WINDOW_BARS
    assert vm.in_gap(cut.time), f"entry at {cut.time:.2f}s is not in a phrase gap"
    assert not cut.mid_phrase
    assert "gap" in cut.reason


@pytest.mark.parametrize("role", ["entry", "exit"])
def test_both_ends_are_snapped_clear_of_a_word(phrased, role: str) -> None:
    """Every boundary is within two bars of a rest here, so none may tear a word.

    "Clear of a word" is the assertion rather than "inside a gap": landing
    exactly on a phrase start is *better* than landing in the middle of the
    silence before it, and the snapper is allowed to prefer it.
    """
    mix, _ = phrased
    vm = S.vocal_map(mix, SR)
    downbeats = set(np.arange(0, 64.0, BAR))
    torn = []
    for bar in range(4, 28):
        cut = S.snap_cut(bar * BAR, downbeats and np.asarray(sorted(downbeats)),
                         vm, BAR, role=role)
        assert cut.time in downbeats, f"bar {bar} left the downbeat grid"
        assert abs(cut.moved_bars) <= S.SNAP_WINDOW_BARS
        if cut.mid_phrase:
            torn.append((bar, round(cut.time, 2)))
        # in a gap, or on the edge of one -- never in the middle of a sung bar
        assert vm.in_gap(cut.time, pad=0.06), \
            f"bar {bar} snapped to {cut.time:.2f}s, which is mid-phrase"
    assert torn == [], f"{role} cuts tore a word at {torn}"


def test_a_metronome_is_not_a_singer() -> None:
    """Huge contrast and sixty evenly spaced "gaps" -- and still not a voice.

    Normalising an envelope to itself makes any bursty signal look articulate,
    so contrast and gap count alone cannot tell a click track from a vocal. The
    median voiced run can: a sung phrase sustains for about a second, a click
    for tens of milliseconds.
    """
    vm = S.vocal_map(stereo(click(30.0)), SR)
    assert len(vm.gaps) > 10 and vm.contrast_db > 40, "premise of the test changed"
    assert vm.voiced_run < S.MIN_VOICED_RUN
    assert not vm.usable


def test_a_voice_sustains_long_enough_to_be_believed(phrased) -> None:
    mix, _ = phrased
    vm = S.vocal_map(mix, SR)
    assert vm.voiced_run >= S.MIN_VOICED_RUN
    assert vm.usable


def test_mid_phrase_is_about_continuity_across_the_splice(phrased) -> None:
    """Sound on both sides is a torn word; sound on one side is a phrase edge."""
    mix, gaps = phrased
    vm = S.vocal_map(mix, SR)
    a, b = gaps[1]
    inside_a_word = a - BAR          # one bar before the rest: mid-phrase
    at_phrase_end = a + 0.05         # the voice has just stopped
    at_phrase_start = b - 0.05       # the voice is about to start
    for role in ("entry", "exit"):
        assert S.snap_cut(inside_a_word, np.asarray([inside_a_word]), vm, BAR,
                          role=role).mid_phrase
        assert not S.snap_cut(at_phrase_end, np.asarray([at_phrase_end]), vm, BAR,
                              role=role).mid_phrase
        assert not S.snap_cut(at_phrase_start, np.asarray([at_phrase_start]), vm, BAR,
                              role=role).mid_phrase


def test_hook_score_prefers_the_voice_over_the_volume() -> None:
    """A loud instrumental section must not outrank a repeated sung one."""
    voc, _ = sing_3_rest_1(bars=16)
    vm = S.vocal_map(stereo(voc), SR)
    sung = S.Section(0.0, 16.0, "hook", 0, energy=0.70, rms_db=-10.0, repeats=3)
    loud = S.Section(20.0, 36.0, "section", 1, energy=1.00, rms_db=-8.0, repeats=1)
    hook = S.section_hook_score(sung, vm, max_repeats=3, bar_dur=BAR)
    other = S.section_hook_score(loud, vm, max_repeats=3, bar_dur=BAR)
    assert hook > other, f"loudness won: sung {hook:.3f} vs loud {other:.3f}"


def test_hook_score_still_discounts_a_section_too_short_to_loop() -> None:
    voc, _ = sing_3_rest_1(bars=16)
    vm = S.vocal_map(stereo(voc), SR)
    long = S.Section(0.0, 16.0, "hook", 0, energy=0.8, rms_db=-10.0, repeats=2)
    short = S.Section(0.0, 4.0, "hook", 0, energy=0.8, rms_db=-10.0, repeats=2)
    assert S.section_hook_score(long, vm, 2, BAR) > S.section_hook_score(short, vm, 2, BAR)


def test_vocal_map_survives_degenerate_input() -> None:
    for buf in (np.zeros((10, 2), dtype=np.float32),
                np.zeros((SR, 2), dtype=np.float32)):
        vm = S.vocal_map(buf, SR)
        assert not vm.usable
        assert vm.activity(0.0) == pytest.approx(0.0, abs=1e-6)
        assert S.snap_cut(1.0, np.asarray([0.0, BAR]), vm, BAR).time in (0.0, BAR)
