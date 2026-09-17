"""Harmonic/percussive source separation by median filtering.

Fitzgerald, "Harmonic/percussive separation using median filtering", DAFx 2010.
Harmonic content is horizontal in a spectrogram (steady across time, narrow in
frequency); percussive content is vertical (broadband, brief). Median-filtering
the magnitude along time suppresses transients and leaves the harmonic estimate;
filtering along frequency suppresses tonal ridges and leaves the percussive one.
Soft Wiener-style masks (Driedger et al., DAFx 2014) split the complex STFT
without the artefacts of a hard binary mask.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..analysis.features import HOP, N_FFT, istft, stft


def hpss(x: np.ndarray, sr: int, kernel_time: float = 0.25, kernel_hz: float = 500.0,
         power: float = 2.0, margin: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Split a mono signal into (harmonic, percussive).

    ``kernel_time`` is the horizontal median length in seconds (long enough to
    span a kick but short enough to track a chord change); ``kernel_hz`` is the
    vertical median width in Hz. ``power`` sets the mask sharpness: 1 is a
    magnitude ratio, 2 is a Wiener filter.
    """
    n_time = max(3, int(round(kernel_time * sr / HOP)) | 1)
    n_freq = max(3, int(round(kernel_hz * N_FFT / sr)) | 1)

    spec = stft(x, N_FFT, HOP)
    mag = np.abs(spec)
    harm = ndimage.median_filter(mag, size=(1, n_time), mode="nearest")
    perc = ndimage.median_filter(mag, size=(n_freq, 1), mode="nearest")

    hp, pp = harm ** power, (perc * margin) ** power
    total = hp + pp + 1e-12
    mask_h, mask_p = hp / total, pp / total

    n = len(x)
    return (istft(spec * mask_h, N_FFT, HOP, length=n).astype(np.float32),
            istft(spec * mask_p, N_FFT, HOP, length=n).astype(np.float32))


def hpss_stereo(x: np.ndarray, sr: int, **kw) -> tuple[np.ndarray, np.ndarray]:
    """HPSS on a stereo buffer.

    The mask is computed once on the mid channel and applied to both, so the two
    channels are separated identically and the stereo image of each part is
    preserved (independent per-channel masks smear the image badly).
    """
    if x.ndim == 1:
        return hpss(x, sr, **kw)
    n_time = max(3, int(round(kw.get("kernel_time", 0.25) * sr / HOP)) | 1)
    n_freq = max(3, int(round(kw.get("kernel_hz", 500.0) * N_FFT / sr)) | 1)
    power = kw.get("power", 2.0)

    mid = x.mean(axis=1)
    mag = np.abs(stft(mid, N_FFT, HOP))
    harm = ndimage.median_filter(mag, size=(1, n_time), mode="nearest")
    perc = ndimage.median_filter(mag, size=(n_freq, 1), mode="nearest")
    hp, pp = harm ** power, perc ** power
    total = hp + pp + 1e-12
    mask_h, mask_p = hp / total, pp / total

    n = len(x)
    hs, ps = [], []
    for c in range(x.shape[1]):
        spec = stft(x[:, c], N_FFT, HOP)
        m = min(spec.shape[1], mask_h.shape[1])
        hs.append(istft(spec[:, :m] * mask_h[:, :m], N_FFT, HOP, length=n))
        ps.append(istft(spec[:, :m] * mask_p[:, :m], N_FFT, HOP, length=n))
    return (np.stack(hs, axis=1).astype(np.float32),
            np.stack(ps, axis=1).astype(np.float32))
