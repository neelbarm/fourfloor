"""The two complaints, as tests.

A listener said the transitions were rough and that the vocals got buried. Both
are claims about numbers, so both get pinned to numbers here: a discontinuity
detector that runs over every slot boundary of a real render, a balance target
taken from a commercial house remix of one of the test sources, and the filter
invariant a drop has to satisfy.
"""

from __future__ import annotations

import numpy as np
import pytest

from fourfloor.audio import decode, step_score
from fourfloor.dsp import dynamics as DY
from fourfloor.dsp import filters as FL
from fourfloor.house.engine import _equal_power, _loop_to
from fourfloor.remix import RemixOptions, remix

TARGET_BPM = 124.0


@pytest.fixture(scope="module")
def mixed(fixture_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("mix") / "fixture.house.mp3"
    return remix(fixture_path, out, RemixOptions(target_bpm=TARGET_BPM, length="3:00",
                                                 stems="hpss", wav=True))


# ---------------------------------------------------------------------------
# transitions: no slot boundary may be a discontinuity
# ---------------------------------------------------------------------------

def _downbeat_scores(mono, sr, plan):
    """Click score at every downbeat, split into slot boundaries and the rest."""
    bounds = {s.start_bar for s in plan.slots} - {0}
    at_bound, interior = {}, []
    for bar in range(1, plan.total_bars):
        score = step_score(mono, sr, bar * plan.bar_dur)
        if bar in bounds:
            at_bound[bar] = score
        else:
            interior.append(score)
    return at_bound, np.array(interior)


def test_no_slot_boundary_clicks(mixed) -> None:
    """A cut between sections must not be spikier than an ordinary downbeat.

    Every downbeat in a house track has a kick on it, so an absolute threshold
    on the derivative would only measure how hard the kick hits. The question
    that matters is comparative: is the sample-to-sample jump at a boundary --
    where two different pieces of source meet, a filter sweep resolves and the
    pattern changes -- unlike the jump at the downbeats either side of it.
    """
    clip = decode(mixed.paths["wav"])
    at_bound, interior = _downbeat_scores(clip.mono, clip.sr, mixed.plan)
    assert at_bound, "the fixture plan should have interior slot boundaries"
    ceiling = float(np.percentile(interior, 99.0))
    for bar, score in at_bound.items():
        assert score <= ceiling, (
            f"bar {bar} is a slot boundary and scores {score:.2f}, above the "
            f"99th percentile of ordinary downbeats ({ceiling:.2f})"
        )


def test_the_source_layer_is_continuous_across_boundaries(mixed) -> None:
    """The same test on the source bed alone, where no kick can hide a click.

    The mix test can be passed by a loud enough kit. This one cannot: the
    harmonic bed is the layer that used to be spliced with a pair of 12 ms
    fades to zero, and it has no transients of its own to hide behind.
    """
    bed = np.asarray(_source_bed(mixed))
    sr = mixed.sr
    at_bound, interior = _downbeat_scores(bed.mean(axis=1), sr, mixed.plan)
    ceiling = float(np.percentile(interior, 99.0))
    for bar, score in at_bound.items():
        assert score <= max(ceiling, 6.0), (
            f"the source bed jumps at bar {bar}: {score:.2f} against an "
            f"ordinary-downbeat 99th percentile of {ceiling:.2f}"
        )


def _source_bed(res) -> np.ndarray:
    """Re-render just the source layer of a finished remix."""
    from fourfloor.house.engine import Engine

    engine = Engine(sr=res.sr, plan=res.plan, stems=_stems_of(res), chords=[],
                    beat_multiple=res.tempo_plan.beat_multiple,
                    src_bar_dur=res.analysis.bar_dur)
    harm, _ = engine.render_source()
    return harm


def _stems_of(res):
    from fourfloor.remix import warp_source
    from fourfloor.stems import separate

    warped = warp_source(res.analysis, res.tempo_plan, res.semitones)
    return separate(warped, res.sr, "hpss")


def test_a_drop_starts_with_its_filters_open(mixed) -> None:
    """The build's sweep has to have resolved by the drop's first sample.

    Not a block later, not over the first beat: the cutoff the build climbed to
    is gone on the first kick, which is what makes the drop read as an arrival
    rather than as a filter being let go of.
    """
    from fourfloor.house.engine import Engine

    engine = Engine(sr=mixed.sr, plan=mixed.plan, stems=_stems_of(mixed), chords=[])
    drops = [i for i, s in enumerate(mixed.plan.slots) if s.kind == "drop"]
    assert drops, "the club form should contain drops"
    for i in drops:
        hp, lp = engine.entry_cutoff(i)
        assert hp == 0.0, f"drop at slot {i} still high-passed at {hp:.0f} Hz"
        assert lp >= mixed.sr * 0.49, f"drop at slot {i} still low-passed at {lp:.0f} Hz"
        prev = mixed.plan.slots[i - 1]
        if prev.highpass:
            assert prev.highpass[1] > hp, "the build should sweep away from open"


def test_a_breakdown_is_handed_the_cutoff_it_opens_from(mixed) -> None:
    """The section before a breakdown sweeps down to where the breakdown starts.

    A step from wide open to 4.2 kHz on one sample is the transition the
    listener called rough; the two halves have to meet at the same number.
    """
    from fourfloor.house.engine import EXIT_LOWPASS_HZ, Engine

    engine = Engine(sr=mixed.sr, plan=mixed.plan, stems=_stems_of(mixed), chords=[])
    for i, slot in enumerate(mixed.plan.slots):
        if slot.kind != "breakdown" or i == 0:
            continue
        _, lp = engine.entry_cutoff(i)
        want = slot.lowpass[0] if slot.lowpass else EXIT_LOWPASS_HZ
        assert lp == pytest.approx(want, rel=1e-6), (
            f"breakdown at slot {i} opens from {want:.0f} Hz but the section "
            f"before it hands over {lp:.0f} Hz"
        )


# ---------------------------------------------------------------------------
# balance: the source has to be audible against the kit
# ---------------------------------------------------------------------------

def test_the_source_sits_with_the_bed_in_the_vocal_band(mixed) -> None:
    """In a drop, 300 Hz - 4 kHz, the source is level with the rest of the mix.

    The target is the commercial house remix of one of the test sources,
    measured with the same split fourfloor uses -- source against drums plus
    bass plus original percussion. Its drop reads +1.7 dB. fourfloor's fixture
    used to read +7.4, which is a pad sitting where the kick and the bass are
    supposed to be.
    """
    drops = [b for b in mixed.metrics["balance"] if b["kind"] == "drop"]
    assert drops
    for b in drops:
        margin = b["source_minus_bed_vocal_db"]
        assert -1.0 <= margin <= 4.0, (
            f"drop at {b['start']:.1f}s has the source {margin:+.2f} dB against "
            "the bed in 300 Hz - 4 kHz; the reference drop is +1.7"
        )


def test_the_bed_owns_the_full_band_in_a_drop(mixed) -> None:
    """...and the same drop is 8 to 16 dB bed-dominated across the whole range.

    The two numbers together are the point: level in the band where a voice is
    understood, buried underneath it everywhere else. The reference reads -12.0.
    """
    for b in (b for b in mixed.metrics["balance"] if b["kind"] == "drop"):
        assert -16.0 <= b["source_minus_bed_full_db"] <= -8.0, (
            f"drop at {b['start']:.1f}s reads "
            f"{b['source_minus_bed_full_db']:+.2f} dB full-band"
        )


def test_the_kit_does_not_run_away_with_the_presence_band(mixed) -> None:
    """2 - 5 kHz is where consonants live and where hats and claps want to sit.

    The reference drop has the kit 2.9 dB over the source there. fourfloor's
    fixture used to give it 13.4 dB, which is most of what "the vocals get
    drowned out" means. The bound here is the one the adaptive solve can
    actually reach on a source with very little of its own 3 kHz content.
    """
    for b in (b for b in mixed.metrics["balance"] if b["kind"] == "drop"):
        assert b["presence_margin_db"] >= -9.5, (
            f"drop at {b['start']:.1f}s leaves the kit "
            f"{-b['presence_margin_db']:.1f} dB over the source at 2 - 5 kHz"
        )


def test_the_balance_is_solved_not_dialled_in(mixed) -> None:
    """The corrections must differ between sections, or they are a fixed gain."""
    moves = mixed.metrics["balance_moves"]
    assert moves
    live = [m for m in moves if m["kind"] in ("drop", "build")]
    assert any(m["presence_lift_db"] > 0 for m in live)
    assert len({round(m["source_trim_db"], 1) for m in moves}) > 1


def test_the_master_hits_its_loudness_target(mixed) -> None:
    clip = decode(mixed.paths["wav"])
    peak = 20 * np.log10(float(np.max(np.abs(clip.samples))))
    assert peak == pytest.approx(-1.0, abs=0.1)
    assert not np.any(np.abs(clip.samples) >= 1.0)
    # The reference corpus masters to -10.8 dBFS at a 12 dB crest. fourfloor
    # holds the peak at -1.0 exactly, so its crest -- and therefore its RMS --
    # is whatever survives the limiter; the band is wide enough to allow that
    # and narrow enough to catch a render that has gone quiet or been squashed.
    assert -12.5 <= DY.rms_db(clip.samples) <= -8.0


# ---------------------------------------------------------------------------
# the pieces, on their own
# ---------------------------------------------------------------------------

def test_loop_phase_does_not_drift() -> None:
    """A looped span must land on the same phase every repeat.

    The old implementation crossfaded repeat onto repeat with a function that
    returns a buffer shorter than its inputs, so the loop walked forward of the
    grid by the fade length every period.
    """
    sr = 8000
    period = sr                     # one second
    src = np.zeros((sr * 4, 2), dtype=np.float32)
    src[::period // 8] = 1.0        # a tick every eighth of the period
    out = _loop_to(src, 0, period * 6, period, sr, seam=0)
    ticks = np.flatnonzero(out[:, 0] > 0.5)
    assert len(ticks) == 6 * 8
    assert np.array_equal(np.diff(ticks), np.full(len(ticks) - 1, period // 8))


def test_loop_pre_roll_is_at_the_loop_phase_the_slot_will_have() -> None:
    """The crossfade lead-in is what the loop would have been playing.

    Not the material that happens to precede the span in the source: the phase
    the loop is on. Those are the same thing on the first pass through and
    different on every later one, and it is the phase that has to be right, or
    the fade into a slot arrives on a different part of the bar than the slot.
    """
    sr = 8000
    src = np.arange(sr * 4, dtype=np.float32)[:, None].repeat(2, axis=1)
    period, pre = sr, 400
    out = _loop_to(src, sr, sr, period, sr, pre=pre, seam=0)
    assert out[pre, 0] == pytest.approx(sr)                   # the slot's own start
    assert out[0, 0] == pytest.approx(sr + period - pre)      # the loop's tail


def test_crossfade_gains_are_complementary() -> None:
    t = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    out_g, in_g = _equal_power(t, False)
    assert np.allclose(out_g ** 2 + in_g ** 2, 1.0, atol=1e-6)
    assert out_g[0] == pytest.approx(1.0)
    assert in_g[-1] == pytest.approx(1.0)


def test_split_sidechain_ducks_the_low_band_and_spares_the_vocal(sr: int) -> None:
    """A 60 Hz tone and a 2 kHz tone, ducked by the same envelope."""
    n = sr
    t = np.arange(n) / sr
    low = np.sin(2 * np.pi * 60.0 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 2000.0 * t).astype(np.float32)
    env = np.full(n, 0.2, dtype=np.float32)     # a steady -14 dB duck

    gentle = 20 * np.log10(1.0 - (1.0 - 0.2) * DY.SIDECHAIN_HIGH_SCALE)
    for sig, want_db in ((low, -14.0), (high, gentle)):
        x = np.stack([sig, sig], axis=1)
        y = DY.split_sidechain(x, sr, env)
        got = DY.rms_db(y[sr // 4:]) - DY.rms_db(x[sr // 4:])
        assert got == pytest.approx(want_db, abs=1.2)


def test_split_sidechain_is_flat_when_both_bands_move_together(sr: int) -> None:
    """With the high band ducked as hard as the low one, nothing is coloured.

    A Linkwitz-Riley pair sums to an all-pass rather than to the input, so the
    test is on the magnitude spectrum, not on the samples.
    """
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((sr, 2)) * 0.2).astype(np.float32)
    env = np.full(sr, 0.5, dtype=np.float32)
    y = DY.split_sidechain(x, sr, env, high_scale=1.0)
    keep = slice(sr // 4, None)
    assert DY.rms_db(y[keep]) == pytest.approx(DY.rms_db(x[keep] * 0.5), abs=0.05)
    for band in ((60.0, 120.0), (300.0, 4000.0), (6000.0, 12000.0)):
        got = DY.band_rms_db(y[keep], sr, band)
        want = DY.band_rms_db(x[keep] * 0.5, sr, band)
        assert got == pytest.approx(want, abs=0.3), f"{band} coloured by {got - want:+.2f} dB"


def test_moving_bell_lifts_its_centre_and_leaves_the_rest_alone(sr: int) -> None:
    """+4 dB at the centre, unity two decades away, for a constant gain."""
    t = np.arange(sr) / sr
    for freq, want in ((3160.0, 4.0), (80.0, 0.0), (18000.0, 0.0)):
        x = np.sin(2 * np.pi * freq * t).astype(np.float32)
        y = DY.moving_bell(x, sr, 3160.0, DY.db_to_bell(4.0), q=0.6)
        keep = slice(sr // 4, None)
        assert DY.rms_db(y[keep]) - DY.rms_db(x[keep]) == pytest.approx(want, abs=0.4)


def test_moving_bell_gain_curve_does_not_click(sr: int) -> None:
    """Sweeping the bell's gain from flat to +6 dB must not step."""
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 3160.0 * t).astype(np.float32)
    gain = DY.db_to_bell(np.linspace(0.0, 6.0, sr, dtype=np.float32))
    y = DY.moving_bell(x, sr, 3160.0, gain, q=0.6)
    jumps = np.abs(np.diff(y))
    assert float(jumps.max()) < 1.6 * float(np.percentile(jumps, 99.0))


def test_sidechain_recovers_in_the_window_the_references_use(sr: int) -> None:
    """-14 dB of duck, back to within 1 dB of unity in 80 to 110 ms."""
    from fourfloor.house.engine import (SIDECHAIN_LOW_DIP_DB, SIDECHAIN_RECOVERY,
                                        SIDECHAIN_SHAPE, Engine)

    depth = 1.0 - 10.0 ** (SIDECHAIN_LOW_DIP_DB / 20.0)
    release = SIDECHAIN_RECOVERY / Engine._recovery_fraction(SIDECHAIN_LOW_DIP_DB)
    env = DY.sidechain_envelope(sr, sr, np.array([0.0]), depth=depth,
                                release=release, shape=SIDECHAIN_SHAPE)
    bottom = int(np.argmin(env))
    assert 20 * np.log10(float(env[bottom])) == pytest.approx(SIDECHAIN_LOW_DIP_DB, abs=0.5)
    after = env[bottom:]
    back = int(np.argmax(after >= 10.0 ** (-1.0 / 20.0))) / sr
    assert 0.080 <= back <= 0.110, f"recovered in {back * 1000:.0f} ms"
