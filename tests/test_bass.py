"""The house bass: pattern, pitch, envelope, level and measurement.

Every threshold here comes from ``bass.REFERENCE`` -- numbers measured off the
Demucs bass stems of six commercial house remixes. If a default in ``bass.py``
drifts away from what real records do, one of these fails and says by how much.
"""

from __future__ import annotations

import numpy as np
import pytest

from fourfloor.dsp import dynamics as DY
from fourfloor.house import bass as BA

BPM = 126.0
BAR = 4 * 60.0 / BPM
# Am - F - C - G, one chord per bar: four distinct roots, so a line that just
# sat on the tonic would be obvious.
CHORDS = [(9, True), (9, True), (5, False), (5, False),
          (0, False), (0, False), (7, False), (7, False)]
ROOTS = {pc for pc, _ in CHORDS}
FIFTHS = {(pc + 7) % 12 for pc, _ in CHORDS}


def kick_times(bars: int, bar_dur: float = BAR) -> list[float]:
    """Four-to-the-floor, the grid the bass has to dodge."""
    return [(b + k / 4.0) * bar_dur for b in range(bars) for k in range(4)]


@pytest.fixture(scope="module")
def drop_line():
    """An 8-bar drop bassline with the engine's sidechain over a kick grid."""
    sr = 44100
    bars = 8
    raw = BA.render_bassline(sr, BAR, CHORDS, bars=bars, style="house",
                             seed=7, stereo=False)
    kt = kick_times(bars)
    env = DY.sidechain_envelope(len(raw), sr, np.asarray(kt), depth=0.85,
                                release=0.20)
    return DY.apply_sidechain(raw, env), sr, kt, bars


# -- pitch -------------------------------------------------------------------

@pytest.mark.parametrize("pc", range(12))
def test_picked_octave_keeps_the_fundamental_in_the_club_window(pc: int) -> None:
    """Every pitch class lands inside the measured window, near the 50 Hz centre."""
    f = BA.note_hz(pc, BA.pick_bass_octave(pc))
    lo, hi = BA.TARGET_F0_HZ
    assert lo <= f <= hi, f"pitch class {pc} lands at {f:.1f} Hz"
    off = abs(12 * np.log2(f / BA.BASS_CENTRE_HZ))
    assert off <= 6.0, f"pitch class {pc} is {off:.1f} semitones off centre"


def test_octave_choice_is_nearest_to_centre_not_first_above_a_floor() -> None:
    """The old rule put C an octave above B. The centred rule does not."""
    b = BA.note_hz(11, BA.pick_bass_octave(11))
    c = BA.note_hz(0, BA.pick_bass_octave(0))
    assert abs(12 * np.log2(c / b)) <= 2.0, (
        f"B at {b:.1f} Hz and C at {c:.1f} Hz are not neighbours")


def test_notes_are_chord_roots_or_fifths() -> None:
    """The line follows the per-bar chords rather than droning on one note."""
    events = BA.bass_pattern(CHORDS, bars=8, seed=3)
    for ev in events:
        if ev.role == "passing":
            continue
        bar_root = CHORDS[ev.bar % len(CHORDS)][0]
        allowed = {bar_root, (bar_root + 7) % 12}
        assert ev.pitch_class in allowed, (
            f"bar {ev.bar} step {ev.step}: {ev.pitch_class} is neither the root "
            f"{bar_root} nor its fifth")


@pytest.mark.parametrize("seed", [0, 5, 11])
def test_every_note_stays_in_the_target_window(seed: int) -> None:
    """No note drops under the window -- headroom spent below audibility."""
    lo, hi = BA.TARGET_F0_HZ
    for ev in BA.bass_pattern(CHORDS, bars=16, seed=seed):
        assert lo <= ev.freq <= hi, (
            f"{ev.role} note at {ev.freq:.1f} Hz is outside {lo}-{hi} Hz")


@pytest.mark.parametrize("chords", [
    CHORDS,
    [(2, True), (2, True), (9, True), (9, True),
     (7, False), (7, False), (4, True), (4, True)],       # Dm-Am-G-Em
    [(1, True), (6, False), (8, True), (11, False)],      # C#m-F#-G#m-B
    [(11, True), (4, True), (9, False), (2, False)],      # Bm-Em-A-D
])
def test_the_line_stays_in_one_register(chords) -> None:
    """The references live inside ~6.7 semitones. No octave leaps out of it."""
    events = BA.bass_pattern(chords, bars=16, seed=7)
    span = BA.span_semitones([ev.freq for ev in events])
    assert span <= BA.BASS_SPAN_SEMITONES, (
        f"line spans {span:.1f} semitones, budget {BA.BASS_SPAN_SEMITONES}")


def test_the_line_moves_with_the_chords() -> None:
    """More than one root appears, and no single note dominates the bar."""
    events = BA.bass_pattern(CHORDS, bars=8, seed=5)
    pcs = [ev.pitch_class for ev in events]
    assert len(set(pcs)) >= 4, f"only {len(set(pcs))} distinct pitch classes"
    top = max(pcs.count(p) for p in set(pcs)) / len(pcs)
    assert top < 0.6, f"one pitch class is {top:.0%} of the line"


def test_chord_changes_slide() -> None:
    """The first note of a bar with a new root glides in from the old one."""
    events = BA.bass_pattern(CHORDS, bars=8, seed=3)
    slides = [ev for ev in events if ev.slide_from_hz is not None]
    assert slides, "no slides at all across four chord changes"
    for ev in slides:
        prev_root = CHORDS[(ev.bar - 1) % len(CHORDS)][0]
        assert CHORDS[ev.bar % len(CHORDS)][0] != prev_root
        assert ev.slide_from_hz > 0


def test_phrase_ends_take_a_passing_note() -> None:
    events = BA.bass_pattern(CHORDS, bars=16, seed=3, phrase=8)
    passing = [ev for ev in events if ev.role == "passing"]
    assert passing, "no passing notes at the phrase ends"
    assert all(ev.bar % 8 == 7 for ev in passing)


# -- pattern -----------------------------------------------------------------

def test_the_bass_rests_where_the_kick_hits() -> None:
    """Steps 0, 4, 8 and 12 are the kick's. The references leave them empty."""
    events = BA.bass_pattern(CHORDS, bars=8, seed=1)
    on_kick = [ev for ev in events if ev.step % 4 == 0]
    assert not on_kick, f"{len(on_kick)} notes land on top of the kick"


def test_note_density_matches_the_references() -> None:
    events = BA.bass_pattern(CHORDS, bars=8, seed=1)
    per_bar = len(events) / 8
    lo, hi = BA.REFERENCE["onsets_per_bar_range"]
    assert lo <= per_bar <= hi + 1.0, f"{per_bar} notes per bar"


def test_the_default_never_bounces_the_octave() -> None:
    """Measured octave-jump rate across the corpus is 0.000. So is ours.

    This is the regression that matters most: the previous bass lifted an
    octave on step 5 of every single bar, and that tic is the thing that made
    fourfloor's drops sound synthetic rather than like a record.
    """
    events = BA.bass_pattern(CHORDS, bars=32, seed=2)
    assert not [ev for ev in events if ev.role == "octave"]

    midi = [round(12 * np.log2(ev.freq / 440.0) + 69) for ev in events]
    jumps = np.abs(np.diff(midi))
    share = float(np.mean((jumps >= 11) & (jumps <= 13)))
    assert share <= BA.REFERENCE["octave_jump_share_max"], (
        f"{share:.1%} of note-to-note moves are octave jumps")


@pytest.mark.parametrize("style", sorted(BA.STYLES))
def test_no_style_leans_on_octave_bounces(style: str) -> None:
    events = BA.bass_pattern(CHORDS, bars=32, seed=2, style=style)
    share = sum(ev.role == "octave" for ev in events) / len(events)
    assert share <= 0.05, f"{style} uses octave lifts on {share:.0%} of notes"


@pytest.mark.parametrize("style", sorted(BA.STYLES))
def test_every_style_renders_clean_audio(style: str) -> None:
    sr = 44100
    x = BA.render_bassline(sr, BAR, CHORDS, bars=4, style=style, seed=4,
                           stereo=False)
    rep = BA.bass_report(x, sr, bar_dur=BAR)
    assert not rep["has_nan"]
    assert not rep["clipping"]
    assert rep["peak"] > 0.05, f"{style} rendered near-silence"


# -- envelope and level ------------------------------------------------------

def test_envelope_dips_after_every_kick(drop_line) -> None:
    """The bass ducks under the kick as deep as the references do."""
    x, sr, kt, _ = drop_line
    rep = BA.bass_report(x, sr, kick_times=kt, bar_dur=BAR)
    assert rep["kicks_measured"] >= len(kt) // 2
    lo, hi = BA.REFERENCE["sidechain_dip_db_range"]
    assert lo - 5.0 <= rep["sidechain_dip_db"] <= hi + 4.0, (
        f"dip {rep['sidechain_dip_db']} dB is outside the target {lo}..{hi} dB")
    assert rep["dip_in_reference_range"]


def test_the_line_ducks_rather_than_gates(drop_line) -> None:
    """Past about -20 dB a sidechain stops pumping and starts sounding gated.

    The old line went properly silent between kicks and measured near -30 dB
    here; notes that sustain across the beat are what put this back in range.
    """
    x, sr, kt, _ = drop_line
    rep = BA.bass_report(x, sr, kick_times=kt, bar_dur=BAR)
    assert rep["sidechain_dip_db"] >= -23.0, (
        f"{rep['sidechain_dip_db']} dB reads as gating, not pumping")


def test_the_duck_recovers_before_the_next_kick(drop_line) -> None:
    x, sr, kt, _ = drop_line
    rep = BA.bass_report(x, sr, kick_times=kt, bar_dur=BAR)
    beat = 60.0 / BPM
    assert 0.08 <= rep["sidechain_recovery_s"] <= beat, (
        f"recovery {rep['sidechain_recovery_s']}s against a {beat:.3f}s beat")


def test_harmonic_content_is_sub_led_not_saw_led(drop_line) -> None:
    """Energy above 200 Hz sits where the reference stems put it."""
    x, sr, _, _ = drop_line
    rep = BA.bass_report(x, sr, bar_dur=BAR)
    lo, hi = BA.REFERENCE["above_200_rel_db_range"]
    assert lo - 2.0 <= rep["above_200_rel_db"] <= hi + 2.0, (
        f"{rep['above_200_rel_db']} dB above 200 Hz, references say {lo}..{hi}")
    assert rep["above_200_in_reference_range"]


def test_fundamentals_land_in_the_target_window(drop_line) -> None:
    x, sr, _, _ = drop_line
    rep = BA.bass_report(x, sr, bar_dur=BAR)
    assert rep["f0_in_target_share"] >= 0.9, rep["f0_in_target_share"]
    assert rep["f0_p10_hz"] >= BA.TARGET_F0_HZ[0] - 1.0
    assert rep["f0_p90_hz"] <= BA.TARGET_F0_HZ[1] + 1.0


def test_the_rendered_line_measures_as_one_register(drop_line) -> None:
    """Not just the note list -- the audio itself has to stay in the register."""
    x, sr, _, _ = drop_line
    rep = BA.bass_report(x, sr, bar_dur=BAR)
    assert rep["span_in_reference_range"], (
        f"rendered line spans {rep['span_semitones']} semitones")


def kick_bus(n: int, kt, sr: int, gain: float = 0.72) -> np.ndarray:
    """A four-to-the-floor kick at engine.DRUM_GAIN, to measure the bass against."""
    from fourfloor.audio import add_at
    from fourfloor.house import drums as DR

    out = np.zeros(n, dtype=np.float32)
    sample = DR.kick(sr)
    for t in kt:
        add_at(out, sample, int(round(t * sr)), 1.0)
    return out * gain


def test_level_against_the_kick_hits_the_published_target(drop_line) -> None:
    """The number the mix stage reads is the number the render delivers."""
    x, sr, kt, _ = drop_line
    # bus gains exactly as engine.py applies them
    rep = BA.bass_report(x * 0.55, sr, kick=kick_bus(len(x), kt, sr), bar_dur=BAR)
    assert rep["level_target_db"] == BA.SUB_LEVEL_VS_KICK_DB
    assert abs(rep["level_error_db"]) <= BA.SUB_LEVEL_TOLERANCE_DB, (
        f"bass sits {rep['bass_minus_kick_db']} dB against the kick, target "
        f"{BA.SUB_LEVEL_VS_KICK_DB} dB")
    assert rep["level_on_target"]


def test_the_engines_own_note_loop_is_also_on_level() -> None:
    """engine.render_bass still calls bass_note per note; that path must land too.

    Until the engine adopts ``render_bassline`` this is the level a listener
    actually hears, so it is the one worth pinning.
    """
    from fourfloor.audio import add_at

    sr, bars = 44100, 8
    n = int(bars * BAR * sr) + sr // 2
    raw = np.zeros(n, dtype=np.float32)
    for b in range(bars):
        root = CHORDS[b % len(CHORDS)][0]
        octave = BA.pick_bass_octave(root)
        for k in (1, 3, 5, 7):
            f = BA.note_hz(root, octave + (1 if k == 5 else 0))
            add_at(raw, BA.bass_note(sr, f, BAR / 8.0 * 0.95),
                   int(round((b + k / 8.0) * BAR * sr)), 0.9)

    kt = kick_times(bars)
    env = DY.sidechain_envelope(n, sr, np.asarray(kt), depth=0.85, release=0.20)
    x = DY.apply_sidechain(raw, env) * 0.55
    rep = BA.bass_report(x, sr, kick=kick_bus(n, kt, sr), bar_dur=BAR)
    assert abs(rep["level_error_db"]) <= BA.SUB_LEVEL_TOLERANCE_DB, (
        f"engine note loop sits {rep['bass_minus_kick_db']} dB against the kick")


def test_match_kick_level_locks_the_balance(drop_line) -> None:
    """The helper holds the target even when the bass arrives badly wrong."""
    x, sr, kt, _ = drop_line
    kick = kick_bus(len(x), kt, sr)
    for offset_db in (-8.0, -3.0, 6.0):
        wrong = x * 10.0 ** (offset_db / 20.0)
        fixed, applied = BA.match_kick_level(wrong, kick, sr)
        got = BA.sub_level_db(fixed, sr) - BA.sub_level_db(kick, sr)
        assert got == pytest.approx(BA.SUB_LEVEL_VS_KICK_DB, abs=0.2), got
        assert -24.0 <= applied <= 12.0


def test_match_kick_level_refuses_to_shovel_gain(drop_line) -> None:
    """A near-empty bar must not be dragged up to kick level, noise floor and all.

    Attenuation is uncapped by comparison: turning a bass down never hurts.
    """
    x, sr, kt, _ = drop_line
    kick = kick_bus(len(x), kt, sr)
    _, applied = BA.match_kick_level(x * 10.0 ** (-40.0 / 20.0), kick, sr)
    assert applied == pytest.approx(12.0, abs=1e-6)


def test_match_kick_level_leaves_silence_alone() -> None:
    sr = 44100
    quiet = np.zeros(sr, dtype=np.float32)
    out, g = BA.match_kick_level(quiet, quiet, sr)
    assert g == 0.0 and np.array_equal(out, quiet)


def test_styles_normalise_to_the_same_sub_level() -> None:
    """Crest factor swings between styles; the delivered sub level must not."""
    sr = 44100
    levels = {s: BA.sub_level_db(
        BA.render_bassline(sr, BAR, CHORDS, bars=4, style=s, seed=4, stereo=False), sr)
        for s in BA.STYLES}
    spread = max(levels.values()) - min(levels.values())
    assert spread <= 1.5, f"styles differ by {spread:.1f} dB: {levels}"


def test_no_nan_and_no_clipping(drop_line) -> None:
    x, sr, kt, _ = drop_line
    assert np.isfinite(x).all()
    rep = BA.bass_report(x, sr, kick_times=kt, bar_dur=BAR)
    assert not rep["has_nan"]
    assert not rep["clipping"]
    assert rep["peak"] <= 1.0


# -- determinism -------------------------------------------------------------

def test_same_seed_renders_the_same_audio() -> None:
    sr = 44100
    a = BA.render_bassline(sr, BAR, CHORDS, bars=4, seed=99, stereo=False)
    b = BA.render_bassline(sr, BAR, CHORDS, bars=4, seed=99, stereo=False)
    assert np.array_equal(a, b)


def test_different_seeds_render_different_lines() -> None:
    a = BA.bass_pattern(CHORDS, bars=16, seed=1)
    b = BA.bass_pattern(CHORDS, bars=16, seed=2)
    assert a != b, "the seed does not reach the pattern"


def test_pattern_is_deterministic() -> None:
    assert BA.bass_pattern(CHORDS, bars=8, seed=5) == \
        BA.bass_pattern(CHORDS, bars=8, seed=5)


# -- the API the engine already calls ---------------------------------------

def test_engine_entry_points_still_work() -> None:
    """engine.render_bass calls these exactly like this; keep them working."""
    sr = 44100
    octave = BA.pick_bass_octave(9)
    f = BA.note_hz(9, octave)
    note = BA.bass_note(sr, f, BAR / 8.0 * 0.95)
    assert note.ndim == 1 and note.dtype == np.float32
    assert np.isfinite(note).all() and 0.0 < float(np.max(np.abs(note))) <= 1.0

    s = BA.stab(sr, 9, True, BAR / 8.0 * 0.8)
    assert s.ndim == 1 and np.isfinite(s).all()


def test_bass_note_glides() -> None:
    """A slide starts near the old pitch and ends on the new one."""
    sr = 44100
    y = BA.bass_note(sr, 55.0, 0.4, glide_from=41.2, glide_time=0.12)
    assert np.isfinite(y).all()

    def peak_hz(seg):
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 16))
        fr = np.fft.rfftfreq(1 << 16, 1.0 / sr)
        m = (fr > 30) & (fr < 200)
        return float(fr[m][int(np.argmax(spec[m]))])

    assert peak_hz(y[int(0.20 * sr):]) == pytest.approx(55.0, abs=2.0)


def test_chords_accepted_as_analysis_dicts() -> None:
    """key.chords_per_bar hands out dicts; take them without a shim."""
    dicts = [{"root_pc": 9, "quality": "min"}, {"root_pc": 5, "quality": "maj"}]
    assert BA.normalise_chords(dicts) == [(9, True), (5, False)]
    events = BA.bass_pattern(dicts, bars=2, seed=0)
    assert {ev.bar for ev in events} == {0, 1}


def test_reference_targets_are_published() -> None:
    """The mix stage reads these; they must exist and be sane."""
    assert BA.SUB_BAND == (40.0, 120.0)
    assert -6.0 <= BA.SUB_LEVEL_VS_KICK_DB <= 6.0
    # Level *with* the kick, not tucked under it.
    assert BA.SUB_LEVEL_RANGE_DB[0] < BA.SUB_LEVEL_VS_KICK_DB < BA.SUB_LEVEL_RANGE_DB[1]
    assert BA.TARGET_F0_HZ[0] < BA.BASS_CENTRE_HZ < BA.TARGET_F0_HZ[1] <= 100.0
    assert 35.0 <= BA.KICK_FUNDAMENTAL_HZ <= 50.0
    for key in ("bass_minus_kick_db", "above_200_rel_db", "sidechain_dip_db",
                "sidechain_recovery_s", "f0_median_hz", "onsets_per_bar",
                "span_semitones", "octave_jump_share_max", "kick_f0_hz"):
        assert key in BA.REFERENCE
    # The raw stem figure is kept, but it must not be what anything targets.
    assert BA.REFERENCE["sidechain_dip_db_raw"] < BA.REFERENCE["sidechain_dip_db"]


def test_bass_report_survives_silence_and_garbage() -> None:
    sr = 44100
    quiet = BA.bass_report(np.zeros(sr, dtype=np.float32), sr)
    assert quiet["peak"] == 0.0 and not quiet["has_nan"]

    bad = np.full(sr, np.nan, dtype=np.float32)
    assert BA.bass_report(bad, sr)["has_nan"]

    loud = BA.bass_report(np.full(sr, 2.0, dtype=np.float32), sr)
    assert loud["clipping"]
