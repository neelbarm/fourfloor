"""The chunked spectral features give the whole-signal answers, in bounded memory."""

from __future__ import annotations

import tracemalloc

import numpy as np
from scipy import signal as sps

from fourfloor.analysis import features as F


def _old_stft(x, n_fft, hop):
    """The whole-signal STFT the chunked path replaced, kept as the reference."""
    x = np.pad(np.asarray(x, dtype=np.float64), n_fft // 2, mode="reflect")
    win = sps.get_window("hann", n_fft, fftbins=True)
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[idx] * win[None, :], n=n_fft, axis=1).T


def _signal(seconds: float, sr: int = 44100) -> np.ndarray:
    rng = np.random.default_rng(3)
    n = int(seconds * sr)
    x = 0.1 * rng.standard_normal(n)
    x[:: sr // 2] += 1.0                               # clicks, for the onsets
    return x.astype(np.float32)


def test_stft_matches_the_gathered_reference() -> None:
    x = _signal(3.0)
    np.testing.assert_array_equal(F.stft(x, 2048, 512), _old_stft(x, 2048, 512))


def test_attack_envelope_and_band_energy_are_unchanged_by_chunking() -> None:
    sr = 44100
    x = _signal(12.0)
    mag = np.abs(_old_stft(x, F.ATTACK_FFT, F.ATTACK_HOP))
    fb = F.mel_filterbank(sr, F.ATTACK_FFT, n_mels=64)
    ref_bands = fb @ mag
    got = F.stft_reduce(x, lambda m: fb @ m, F.ATTACK_FFT, F.ATTACK_HOP, block=333)
    np.testing.assert_allclose(got, ref_bands, rtol=1e-12, atol=1e-12)

    mag2 = np.abs(_old_stft(x, F.N_FFT, F.ATTACK_HOP))
    freqs = np.fft.rfftfreq(F.N_FFT, 1.0 / sr)
    sel = (freqs >= 20.0) & (freqs < 120.0)
    ref_low = np.sqrt(np.mean(mag2[sel] ** 2, axis=0))
    np.testing.assert_allclose(F.band_energy(x, sr, 20.0, 120.0, hop=F.ATTACK_HOP),
                               ref_low, rtol=1e-12, atol=1e-12)


def test_rms_envelope_is_unchanged_and_survives_short_input() -> None:
    x = _signal(2.0)
    win, hop = 2048, 64
    n_frames = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    np.testing.assert_array_equal(F.rms_envelope(x, hop=hop, win=win),
                                  np.sqrt(np.mean(x[idx] ** 2, axis=1)))
    assert F.rms_envelope(x[:100]).shape == (1,)


def test_attack_envelope_memory_does_not_grow_with_the_song() -> None:
    """A whole-song hop-64 spectrum peaked at 5-8 GB on a four-minute record."""
    sr = 44100
    x = _signal(60.0)
    tracemalloc.start()
    try:
        F.attack_envelope(x, sr)
        F.band_energy(x, sr, 20.0, 120.0, hop=F.ATTACK_HOP)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # the unchunked version peaked above 1 GB on this minute of audio
    assert peak < 200e6, f"peak {peak / 1e6:.0f} MB"


def test_novelty_is_unchanged_without_the_whole_similarity_matrix() -> None:
    """The Foote novelty used to build an m-by-m matrix (3.8 GB on four
    minutes); the banded version gives the same curve."""
    from fourfloor.analysis import structure as S

    rng = np.random.default_rng(5)
    m = 2600
    chroma = np.abs(rng.standard_normal((12, m)))
    chroma[:, 1300:] += 2.0 * np.arange(12)[:, None] / 12.0     # a boundary
    timbre = rng.standard_normal((13, m))

    def reference(chroma, timbre, kernel=32):
        def norm_rows(a):
            a = a - a.mean(axis=1, keepdims=True)
            return a / np.maximum(a.std(axis=1, keepdims=True), 1e-9)
        n = min(chroma.shape[1], timbre.shape[1])
        feat = S._stack(np.vstack([norm_rows(chroma[:, :n]),
                                   0.5 * norm_rows(timbre[:, :n])]))
        unit = feat / np.maximum(np.linalg.norm(feat, axis=0, keepdims=True), 1e-9)
        ssm = unit.T @ unit
        mm = ssm.shape[0]
        k = min(kernel, max(4, mm // 8))
        kern = S._checkerboard(k)
        nov = np.zeros(mm)
        for i in range(k, mm - k):
            nov[i] = float(np.sum(ssm[i - k:i + k, i - k:i + k] * kern))
        nov = np.maximum(nov, 0.0)
        return nov / nov.max()

    got = S.novelty_curve(chroma, timbre)
    np.testing.assert_allclose(got, reference(chroma, timbre), rtol=1e-9, atol=1e-12)
    assert abs(int(np.argmax(got)) - 1300) < 40
