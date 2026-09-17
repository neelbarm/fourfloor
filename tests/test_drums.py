"""The synthesised kit, measured against six commercial house drum stems.

``drums.REFERENCE`` holds ranges measured from the drum stems of six commercial
house remixes (isolated with Demucs, analysed hit by hit and again over whole
16-bar windows). ``drums.kit_report()`` measures the synthesised kit with the
same code paths. These tests assert the second lands inside the first, so a
change to a voice that makes it stop sounding like a record fails here rather
than in someone's ears three renders later.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from fourfloor.house import drums as DR

SR = 44100
ORIGINAL_PATTERNS = ("drop", "drop_var", "intro", "intro_full", "build",
                     "breakdown", "outro")


@pytest.fixture(scope="module")
def report() -> dict:
    return DR.kit_report(sr=SR)


@pytest.fixture(scope="module")
def kit() -> DR.Kit:
    return DR.Kit(sr=SR)


# ---------------------------------------------------------------------------
# the numbers, against the references
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", sorted(DR.REFERENCE))
def test_kit_lands_inside_the_reference_range(report: dict, metric: str) -> None:
    lo, hi = DR.REFERENCE[metric]
    if metric.startswith("band_"):
        value = report["band_balance_db"][metric[len("band_"):-len("_db")]]
    else:
        value = report[metric]
    assert np.isfinite(value), f"{metric} is not finite"
    assert lo <= value <= hi, f"{metric}={value} outside the reference range [{lo}, {hi}]"


def test_kick_fundamental_is_a_house_kick(report: dict) -> None:
    """45-55 Hz is the brief; the references measured 41.7-45.8."""
    assert 41.0 <= report["kick_f0_hz"] <= 55.0


def test_kick_tail_is_short_enough_not_to_muddy_the_track(report: dict) -> None:
    """The regression this module was rebuilt for: the old kick rang 310 ms."""
    assert report["kick_sub_t20_ms"] < 150.0


def test_kick_has_an_audible_click_transient(report: dict) -> None:
    """Some energy above 2 kHz in the first 10 ms, but not a snare."""
    assert -39.0 <= report["kick_click_db"] <= -18.0


def test_closed_hat_is_a_hat_and_not_static(report: dict) -> None:
    """The old hat sat at 13.4 kHz, which reads as noise, not as a hi-hat."""
    assert 7000.0 <= report["hat_centroid_hz"] <= 9200.0


def test_clap_has_a_room_tail(report: dict) -> None:
    assert report["clap_t20_ms"] >= 28.0


def test_open_hat_decays_before_the_next_downbeat(report: dict) -> None:
    """At 128 BPM a beat is 469 ms; an open hat must be gone well inside it."""
    assert 60.0 <= report["ohat_t20_ms"] <= 130.0


def test_swing_offset_matches_the_references(report: dict) -> None:
    lo, hi = DR.REFERENCE["swing"]
    assert lo <= report["swing"] <= hi


def test_the_kit_is_straight_by_default() -> None:
    """All six references quantise their 16ths to within 3 ms. No shuffle."""
    assert DR.Kit(sr=SR).swing == 0.0
    assert abs(DR.kit_report(sr=SR)["swing"]) < 0.02


def test_straight_render_measures_as_straight() -> None:
    stem = DR.render_pattern(SR, "drop", 8, 128.0, swing=0.0)
    assert abs(DR._swing_of(stem, SR, 128.0)) < 0.03


def test_sixteenths_land_within_the_references_quantise_window() -> None:
    """The corpus quantises to +-7 ms; a straight kit must beat that easily."""
    step_ms = 60.0 / 128.0 / 4.0 * 1000.0
    assert abs(DR._swing_of(DR.render_pattern(SR, "drop", 8, 128.0),
                            SR, 128.0)) * step_ms < 7.0


def test_swing_shows_up_in_the_render() -> None:
    swung = DR.render_pattern(SR, "drop", 8, 128.0, swing=0.10)
    assert DR._swing_of(swung, SR, 128.0) == pytest.approx(0.10, abs=0.03)


def test_band_balance_shape_matches_a_real_drum_stem(report: dict) -> None:
    """Sub and low on top, mids scooped, air above the mids: the house shape."""
    b = report["band_balance_db"]
    assert b["sub"] > b["mid"] and b["low"] > b["mid"]
    assert b["air"] > b["himid"]


# ---------------------------------------------------------------------------
# hygiene: finite, unclipped, deterministic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("voice", ["kick", "clap", "snare", "hat", "ohat",
                                   "shaker", "rim", "perc", "tom"])
def test_one_shots_are_finite_and_unclipped(kit: DR.Kit, voice: str) -> None:
    x = kit.samples[voice]
    assert len(x) > 0
    assert np.all(np.isfinite(x)), f"{voice} has NaN or inf"
    assert float(np.max(np.abs(x))) <= 1.0, f"{voice} clips"
    assert float(np.max(np.abs(x))) > 0.05, f"{voice} is silent"


@pytest.mark.parametrize("name", sorted(DR.PATTERNS))
def test_pattern_renders_finite_and_unclipped(name: str) -> None:
    stems = DR.render_pattern(SR, name, bars=8, bpm=128.0, per_voice=True)
    assert isinstance(stems, dict)
    for voice, buf in stems.items():
        assert np.all(np.isfinite(buf)), f"{name}/{voice} has NaN or inf"
        assert float(np.max(np.abs(buf))) <= 1.0, f"{name}/{voice} stem clips"
    summed = DR.render_pattern(SR, name, bars=8, bpm=128.0)
    assert np.all(np.isfinite(summed))
    assert np.allclose(sum(stems.values()), summed, atol=1e-5)


def test_riser_and_impact_are_finite_and_unclipped() -> None:
    for x in (DR.riser_noise(SR, 2.0), DR.impact(SR)):
        assert np.all(np.isfinite(x))
        assert float(np.max(np.abs(x))) <= 1.0


def test_render_is_deterministic_for_a_fixed_seed() -> None:
    a = DR.render_pattern(SR, "drop", 4, 128.0, swing=0.08, seed=7)
    b = DR.render_pattern(SR, "drop", 4, 128.0, swing=0.08, seed=7)
    assert np.array_equal(a, b)


def test_voices_are_deterministic() -> None:
    for name in ("kick", "clap", "snare", "hat", "shaker", "rim", "perc", "tom"):
        fn = getattr(DR, name)
        assert np.array_equal(fn(SR), fn(SR)), f"{name} is not deterministic"


def test_kit_report_is_deterministic() -> None:
    a, b = DR.kit_report(sr=SR), DR.kit_report(sr=SR)
    a.pop("reference"), b.pop("reference")
    assert a == b


def test_a_different_seed_changes_the_render() -> None:
    a = DR.render_pattern(SR, "drop", 4, 128.0, seed=1)
    b = DR.render_pattern(SR, "drop", 4, 128.0, seed=2)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# the patterns
# ---------------------------------------------------------------------------

def test_every_pattern_is_structurally_valid() -> None:
    for name, pattern in DR.PATTERNS.items():
        for voice, steps in pattern.items():
            for step, vel in steps:
                assert 0 <= step < 16, f"{name}/{voice} step {step} off the bar"
                assert 0.0 < vel <= 1.0, f"{name}/{voice} velocity {vel} out of range"
            assert len(steps) == len({s for s, _ in steps}), \
                f"{name}/{voice} hits the same step twice"


def test_every_pattern_voice_has_a_sample(kit: DR.Kit) -> None:
    """engine.render_drums silently drops a voice with no sample -- catch it here."""
    for name, pattern in DR.PATTERNS.items():
        for voice in pattern:
            if voice == "sub":      # handled by the bass renderer, not the kit
                continue
            assert voice in kit.samples, f"{name} uses {voice!r}, which the kit lacks"


def test_kick_is_four_on_the_floor() -> None:
    for name in ("drop", "drop_var", "intro", "intro_full", "build", "outro"):
        assert {s for s, _ in DR.PATTERNS[name]["kick"]} == {0, 4, 8, 12}, name


def test_clap_is_on_two_and_four() -> None:
    for name in ("drop", "drop_var", "build"):
        steps = {s for s, _ in DR.PATTERNS[name]["clap"]}
        assert {4, 12} <= steps, name


def test_snare_layers_the_clap_in_a_drop() -> None:
    clap = {s for s, _ in DR.PATTERNS["drop"]["clap"]}
    snare = {s for s, _ in DR.PATTERNS["drop"]["snare"]}
    assert {4, 12} <= snare
    # the snare is a layer under the clap, not the main hit
    clap_vel = dict(DR.PATTERNS["drop"]["clap"])
    snare_vel = dict(DR.PATTERNS["drop"]["snare"])
    assert all(snare_vel[s] < clap_vel[s] for s in clap & snare)


def test_open_hats_are_on_the_and_of_each_beat() -> None:
    for name in ("drop", "drop_var", "intro_full", "build"):
        assert {s for s, _ in DR.PATTERNS[name]["ohat"]} == {2, 6, 10, 14}, name


def test_closed_hats_duck_under_the_open_hat() -> None:
    """Closed hats play through the "and", but well under the open hat there."""
    for name in ("drop", "drop_var"):
        hat = dict(DR.PATTERNS[name]["hat"])
        ohat = dict(DR.PATTERNS[name]["ohat"])
        for s in (2, 6, 10, 14):
            assert hat[s] < ohat[s], f"{name} slot {s}"


def test_hat_accents_follow_the_reference_slot_occupancy() -> None:
    """Downbeat loudest, then the "a", then the "and", with the "e" quietest."""
    vel = dict(DR.PATTERNS["drop"]["hat"])
    for beat in (0, 4, 8, 12):
        down, e, and_, a = (vel[beat + i] for i in range(4))
        assert down > a > and_ > e, f"beat at slot {beat}"


def test_thinning_a_hat_line_keeps_the_downbeats() -> None:
    thinned = {s for s, _ in DR._hats(0.5, thin=0.25)}
    assert {0, 4, 8, 12} <= thinned
    assert len(thinned) == 12


def test_build_rolls_into_the_drop() -> None:
    snare = DR.PATTERNS["build"]["snare"]
    assert len(snare) >= 5
    steps = [s for s, _ in snare]
    vels = [v for _, v in snare]
    assert steps == sorted(steps)
    assert vels == sorted(vels), "the roll should get louder into the downbeat"
    assert steps[-1] == 15
    # and the hats climb across the bar
    hat = dict(DR.PATTERNS["build"]["hat"])
    assert hat[15] > hat[0]


def test_breakdown_has_no_kick() -> None:
    assert DR.PATTERNS["breakdown"]["kick"] == []
    assert DR.PATTERNS["breakdown"]["shaker"], "a breakdown still needs a pulse"


def test_intro_perc_is_hats_and_percussion_only() -> None:
    p = DR.PATTERNS["intro_perc"]
    assert p["kick"] == [] and p["clap"] == []
    assert p["hat"] and p["perc"]


def test_outro_thins_out_against_the_drop() -> None:
    def hits(name: str) -> int:
        return sum(len(v) for k, v in DR.PATTERNS[name].items() if k != "sub")
    assert hits("outro") < hits("drop")
    assert hits("intro") < hits("drop")


def test_drop_var_is_busier_than_drop() -> None:
    def hits(name: str) -> int:
        return sum(len(v) for k, v in DR.PATTERNS[name].items() if k != "sub")
    assert hits("drop_var") > hits("drop")


# ---------------------------------------------------------------------------
# backward compatibility with engine.py and producer.py
# ---------------------------------------------------------------------------

def test_the_original_pattern_names_all_still_exist() -> None:
    assert set(ORIGINAL_PATTERNS) <= set(DR.PATTERNS)


def test_producer_pattern_whitelist_is_renderable() -> None:
    from fourfloor.producer import VALID_PATTERNS
    assert VALID_PATTERNS <= set(DR.PATTERNS)


@pytest.mark.parametrize("name,args", [
    ("kick", (SR,)), ("clap", (SR,)), ("hat", (SR,)), ("shaker", (SR,)),
    ("tom", (SR,)), ("riser_noise", (SR, 1.0)), ("impact", (SR,)),
])
def test_original_voices_are_callable_with_their_original_arguments(name, args) -> None:
    x = getattr(DR, name)(*args)
    assert isinstance(x, np.ndarray) and x.dtype == np.float32 and len(x) > 0


def test_original_keyword_arguments_are_still_accepted() -> None:
    """engine.py and any saved session may pass these by name."""
    expected = {
        "kick": ["sr", "length", "f_start", "f_end", "pitch_decay", "amp_decay",
                 "click", "drive"],
        "clap": ["sr", "spread", "bursts", "decay", "seed"],
        "hat": ["sr", "open_", "seed"],
        "shaker": ["sr", "seed"],
        "tom": ["sr", "freq", "seed"],
        "riser_noise": ["sr", "seconds", "f0", "f1", "seed"],
        "impact": ["sr", "seed"],
    }
    for name, names in expected.items():
        params = list(inspect.signature(getattr(DR, name)).parameters)
        assert params[:len(names)] == names, name


def test_hat_open_flag_is_positional_as_engine_calls_it() -> None:
    closed, open_ = DR.hat(SR, False), DR.hat(SR, True)
    assert len(open_) > len(closed)


def test_kit_exposes_the_voices_engine_looks_up(kit: DR.Kit) -> None:
    assert {"kick", "clap", "hat", "ohat", "shaker", "tom"} <= set(kit.samples)


def test_step_time_applies_swing_to_the_odd_sixteenths() -> None:
    k = DR.Kit(sr=SR, swing=0.25)
    step = 0.125
    assert k.step_time(0.0, 0, step) == pytest.approx(0.0)
    assert k.step_time(0.0, 1, step) == pytest.approx(step + 0.25 * step)
    assert k.step_time(0.0, 2, step) == pytest.approx(2 * step)
    assert k.step_time(1.0, 4, step) == pytest.approx(1.0 + 4 * step)


def test_a_kit_can_still_be_built_from_supplied_samples() -> None:
    custom = {"kick": np.zeros(16, dtype=np.float32)}
    k = DR.Kit(sr=SR, samples=custom)
    assert k.samples is custom


def test_env_helper_keeps_its_shape() -> None:
    e = DR._env(1000, 10, 0.05, SR)
    assert e.dtype == np.float32 and e[0] == 0.0 and 0.0 < e[10] <= 1.0
    assert e[-1] < e[10]
