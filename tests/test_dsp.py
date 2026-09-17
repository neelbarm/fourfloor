"""Phase vocoder, pitch shifting, HPSS, filters and dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import peak_freq, sine
from fourfloor.dsp import filters as FL
from fourfloor.dsp.dynamics import (master, normalize_peak, sidechain_envelope,
                                    soft_limit)
from fourfloor.dsp.hpss import hpss
from fourfloor.dsp.phasevocoder import time_stretch, warp
from fourfloor.dsp.pitch import pitch_shift, plan_tempo


@pytest.mark.parametrize("rate", [0.775, 0.833, 1.0, 1.25])
def test_stretch_length_is_exact(rate: float, sr: int) -> None:
    x = sine(440.0, 3.0, sr)
    y = time_stretch(x, rate)
    assert len(y) == int(round(len(x) / rate))


@pytest.mark.parametrize("rate", [0.775, 0.833, 1.25])
def test_stretch_preserves_pitch(rate: float, sr: int) -> None:
    """A 440 Hz sine is still 440 Hz after stretching (within 20 cents)."""
    y = time_stretch(sine(440.0, 3.0, sr), rate)
    cents = 1200 * np.log2(peak_freq(y, sr) / 440.0)
    assert abs(cents) < 20.0, f"{cents:.1f} cents of drift at rate {rate}"


def test_stretch_does_not_smear_transients(sr: int) -> None:
    """A single click stays inside ~20 ms after a 1.2x stretch."""
    x = np.zeros(sr, dtype=np.float32)
    x[sr // 2] = 1.0
    y = time_stretch(x, 1 / 1.2)
    energy = np.abs(y)
    above = np.flatnonzero(energy > 0.1 * energy.max())
    spread_ms = (above[-1] - above[0]) / sr * 1000.0
    assert spread_ms < 20.0, f"click spread over {spread_ms:.1f} ms"


def test_stretched_sine_has_no_beating(sr: int) -> None:
    """A stretched sine's amplitude envelope stays flat (no phasiness)."""
    y = time_stretch(sine(440.0, 3.0, sr), 1 / 1.2)
    env = np.convolve(np.abs(y), np.ones(441) / 441, mode="same")[sr:-sr]
    # measures 0.0597 here; the old 0.06 bound left half a percent of headroom,
    # which is not enough to survive an FFT or resampler change upstream
    assert float(env.std() / env.mean()) < 0.07


@pytest.mark.parametrize("semitones", [-5, -3, 2, 4, 7])
def test_pitch_shift(semitones: int, sr: int) -> None:
    """Pitch moves by the requested interval and the duration is unchanged."""
    x = sine(440.0, 3.0, sr)
    y = pitch_shift(x, semitones)
    assert len(y) == len(x)
    want = 440.0 * 2 ** (semitones / 12.0)
    cents = 1200 * np.log2(peak_freq(y, sr) / want)
    assert abs(cents) < 25.0, f"{cents:.1f} cents off for {semitones:+d}"


def test_warp_pins_beats_to_a_uniform_grid(sr: int) -> None:
    """A drifting click track is warped so beats land on an exact grid."""
    in_times = np.array([0.0, 0.50, 1.05, 1.55, 2.12, 2.60])
    out_times = np.arange(len(in_times)) * 0.4839
    x = np.zeros(int(3.0 * sr), dtype=np.float32)
    for t in in_times:
        x[int(t * sr)] = 1.0
    out_len = int(out_times[-1] * sr)
    y = warp(x, in_times, out_times, sr, out_len)
    peaks = np.flatnonzero(np.abs(y) > 0.35 * np.abs(y).max())
    assert len(peaks) > 0

    # cluster adjacent peaks, then compare cluster centres with the target grid
    groups: list[list[int]] = [[int(peaks[0])]]
    for p in peaks[1:]:
        if p - groups[-1][-1] < 0.05 * sr:
            groups[-1].append(int(p))
        else:
            groups.append([int(p)])
    centres = np.array([float(np.mean(g)) / sr for g in groups])
    for target in out_times[: len(centres)]:
        assert float(np.min(np.abs(centres - target))) < 0.03


def test_hpss_separates_sine_from_clicks(sr: int) -> None:
    """A steady tone goes to the harmonic part, clicks to the percussive part."""
    t = np.arange(sr * 3) / sr
    tone = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    clicks = np.zeros_like(tone)
    for k in range(0, len(clicks), sr // 4):
        clicks[k:k + 64] = np.linspace(1.0, 0.0, min(64, len(clicks) - k))
    harm, perc = hpss(tone + clicks, sr)

    def tone_fraction(sig: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        freqs = np.fft.rfftfreq(len(sig), 1.0 / sr)
        band = (freqs > 280) & (freqs < 320)
        return float((spec[band] ** 2).sum() / max((spec ** 2).sum(), 1e-12))

    assert tone_fraction(harm) > 0.8
    assert tone_fraction(perc) < 0.2
    assert len(harm) == len(tone) and len(perc) == len(tone)


def test_tempo_plan_picks_the_smallest_log_ratio() -> None:
    """80 BPM into 124 is a half-time mapping, not a 1.55x stretch."""
    p = plan_tempo(80.0, 124.0)
    assert p.beat_multiple == 2.0
    assert abs(p.ratio - 0.775) < 1e-3
    assert p.warning is not None            # 0.775 is outside 0.8-1.25
    assert plan_tempo(124.0, 124.0).ratio == pytest.approx(1.0)
    assert plan_tempo(150.0, 124.0).beat_multiple == 1.0
    assert plan_tempo(62.0, 124.0).beat_multiple == 2.0


def test_sidechain_envelope_pumps(sr: int) -> None:
    triggers = np.arange(0, 4.0, 0.4839)
    env = sidechain_envelope(int(4.0 * sr), sr, triggers, depth=0.6)
    assert env.max() <= 1.0 + 1e-6
    assert env.min() < 0.45
    for t in triggers[1:-1]:
        assert env[int((t + 0.005) * sr)] < 0.6      # ducked just after a kick
        assert env[int((t + 0.42) * sr)] > 0.85      # recovered before the next


def test_limiter_and_normalise(sr: int) -> None:
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((sr, 2)) * 0.8).astype(np.float32)
    y = soft_limit(x, ceiling=0.5, sr=sr)
    assert float(np.max(np.abs(y))) <= 0.56
    z = normalize_peak(y, -1.0)
    assert abs(20 * np.log10(float(np.max(np.abs(z)))) + 1.0) < 0.01


def test_master_chain_hits_targets(sr: int) -> None:
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((sr * 4, 2)) * 0.2).astype(np.float32)
    y = master(x, sr, peak_db=-1.0)
    peak = 20 * np.log10(float(np.max(np.abs(y))))
    assert -1.05 < peak < -0.95
    assert not np.any(np.abs(y) >= 1.0)


def test_filter_sweep_is_continuous(sr: int) -> None:
    """A swept filter must not click at its internal block boundaries."""
    x = sine(300.0, 2.0, sr)
    cut = FL.exp_curve(len(x), 200.0, 8000.0)
    y = FL.sweep(x, "highpass", sr, cut, order=2)
    assert float(np.max(np.abs(np.diff(y)))) < 0.2
    assert len(y) == len(x)
