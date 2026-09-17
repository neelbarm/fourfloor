"""Low-level spectral features: STFT, mel filterbank, onset strength, chroma.

Pure numpy/scipy. Hop 512 at 44.1 kHz gives a feature rate of 86.13 Hz, which is
plenty for beat tracking (the finest musical event we care about, a 16th at
200 BPM, is 75 ms = 6.5 frames).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

N_FFT = 2048
HOP = 512
CHROMA_FFT = 4096


def frame_rate(sr: int, hop: int = HOP) -> float:
    """Feature frames per second."""
    return sr / float(hop)


def stft(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP, center: bool = True) -> np.ndarray:
    """Short-time Fourier transform → complex array of shape (1 + n_fft//2, n_frames).

    Uses a periodic Hann window, which satisfies COLA at hop = n_fft/4 so the
    same framing is reusable for the phase vocoder's overlap-add.
    """
    x = np.asarray(x, dtype=np.float64)
    win = sps.get_window("hann", n_fft, fftbins=True)
    if center:
        x = np.pad(x, n_fft // 2, mode="reflect")
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * win[None, :]
    return np.fft.rfft(frames, n=n_fft, axis=1).T


def istft(spec: np.ndarray, n_fft: int = N_FFT, hop: int = HOP, length: int | None = None,
          center: bool = True) -> np.ndarray:
    """Inverse STFT with window-squared normalisation (Griffin & Lim 1984 OLA)."""
    win = sps.get_window("hann", n_fft, fftbins=True)
    frames = np.fft.irfft(spec, n=n_fft, axis=0).T * win[None, :]
    n_frames = frames.shape[0]
    out = np.zeros(n_fft + hop * (n_frames - 1))
    wsum = np.zeros_like(out)
    w2 = win ** 2
    for i in range(n_frames):
        s = i * hop
        out[s:s + n_fft] += frames[i]
        wsum[s:s + n_fft] += w2
    out /= np.maximum(wsum, 1e-8)
    if center:
        out = out[n_fft // 2:]
    if length is not None:
        out = out[:length] if len(out) >= length else np.pad(out, (0, length - len(out)))
    return out


def mel_filterbank(sr: int, n_fft: int = N_FFT, n_mels: int = 96,
                   fmin: float = 27.5, fmax: float | None = None) -> np.ndarray:
    """Slaney-style triangular mel filterbank, area-normalised, (n_mels, 1+n_fft/2)."""
    fmax = fmax if fmax is not None else sr / 2.0

    def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        left = (freqs - lo) / max(mid - lo, 1e-9)
        right = (hi - freqs) / max(hi - mid, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(left, right))
        norm = fb[i].sum()
        if norm > 0:
            fb[i] /= norm
    return fb


def log_mel(x: np.ndarray, sr: int, n_mels: int = 96, hop: int = HOP) -> np.ndarray:
    """Log-compressed mel spectrogram, (n_mels, n_frames)."""
    mag = np.abs(stft(x, N_FFT, hop))
    fb = mel_filterbank(sr, N_FFT, n_mels)
    return np.log1p(1000.0 * (fb @ mag))


def onset_strength(x: np.ndarray, sr: int, hop: int = HOP) -> np.ndarray:
    """Spectral-flux onset envelope over a log-mel spectrogram.

    Half-wave-rectified first difference summed across mel bands, then
    mean-removed with a 0.5 s moving average so that quiet and loud passages
    contribute comparably to the tempo autocorrelation (Ellis, "Beat Tracking
    by Dynamic Programming", JNMR 2007, §2).
    """
    lm = log_mel(x, sr, hop=hop)
    flux = np.maximum(0.0, np.diff(lm, axis=1)).sum(axis=0)
    flux = np.concatenate([[0.0], flux])
    win = max(3, int(round(0.5 * sr / hop)) | 1)
    local = sps.convolve(flux, np.ones(win) / win, mode="same")
    env = np.maximum(0.0, flux - local)
    peak = env.max()
    return env / peak if peak > 0 else env


def band_energy(x: np.ndarray, sr: int, lo: float, hi: float, hop: int = HOP) -> np.ndarray:
    """Per-frame RMS energy inside a frequency band, from the magnitude STFT."""
    mag = np.abs(stft(x, N_FFT, hop))
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    sel = (freqs >= lo) & (freqs < hi)
    if not sel.any():
        return np.zeros(mag.shape[1])
    return np.sqrt(np.mean(mag[sel] ** 2, axis=0))


# ---------------------------------------------------------------------------
# chroma
# ---------------------------------------------------------------------------

def estimate_tuning(x: np.ndarray, sr: int, n_fft: int = CHROMA_FFT) -> float:
    """Global tuning deviation in semitones, in [-0.5, 0.5).

    Parabolic-interpolated spectral peaks are converted to a fractional MIDI
    number; the histogram of their deviation from equal temperament peaks at the
    instrument tuning offset (the standard "tuning estimation by peak binning").

    Interpolation is done on log magnitude, which is unbiased for a Gaussian-ish
    (Hann) main lobe, and only peaks above 250 Hz are used -- below that a single
    FFT bin spans more than a fifth of a semitone and the deviation is noise.
    """
    mag = np.abs(stft(x, n_fft, n_fft // 4))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    if mag.shape[1] == 0:
        return 0.0
    logmag = np.log(mag + 1e-10)
    devs: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    band = (freqs > 250.0) & (freqs < 4000.0)
    for t in range(0, mag.shape[1], 4):
        col = mag[:, t]
        ceiling = col.max()
        if ceiling <= 1e-8:
            continue
        # per-frame threshold: only true partials, well above this frame's floor
        peaks, props = sps.find_peaks(col, height=0.06 * ceiling, prominence=0.03 * ceiling)
        if not len(peaks):
            continue
        keep = band[peaks] & (peaks > 1) & (peaks < len(col) - 1)
        peaks = peaks[keep]
        if not len(peaks):
            continue
        lc = logmag[:, t]
        a, b, c = lc[peaks - 1], lc[peaks], lc[peaks + 1]
        denom = a - 2 * b + c
        shift = np.where(np.abs(denom) > 1e-9, 0.5 * (a - c) / np.where(denom == 0, 1e-9, denom), 0.0)
        shift = np.clip(shift, -0.5, 0.5)
        f = (peaks + shift) * sr / n_fft
        midi = 69.0 + 12.0 * np.log2(np.maximum(f, 1e-6) / 440.0)
        devs.append(midi - np.round(midi))
        weights.append(col[peaks])
    if not devs:
        return 0.0
    allv = np.concatenate(devs)
    allw = np.concatenate(weights)
    # circular-mean over the semitone, weighted by partial strength: robust to
    # the wrap at +/-0.5 that a plain histogram argmax handles badly
    ang = 2.0 * np.pi * allv
    mean_ang = np.arctan2(np.sum(allw * np.sin(ang)), np.sum(allw * np.cos(ang)))
    est = float(mean_ang / (2.0 * np.pi))
    return float(np.clip(est, -0.5, 0.5))


def chromagram(x: np.ndarray, sr: int, hop: int = HOP, tuning: float | None = None,
               n_fft: int = CHROMA_FFT) -> np.ndarray:
    """Tuning-corrected chromagram, (12, n_frames), each column L1-normalised.

    Linear-frequency STFT bins between A1 and C7 are folded onto the 12 pitch
    classes weighted by magnitude, with the global tuning offset removed first.
    """
    if tuning is None:
        tuning = estimate_tuning(x, sr, n_fft)
    mag = np.abs(stft(x, n_fft, hop))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    sel = (freqs >= 55.0) & (freqs <= 2093.0)
    midi = 69.0 + 12.0 * np.log2(np.maximum(freqs[sel], 1e-6) / 440.0) - tuning
    pc = np.mod(np.round(midi).astype(int), 12)
    sub = mag[sel] ** 2
    out = np.zeros((12, mag.shape[1]))
    for k in range(12):
        m = pc == k
        if m.any():
            out[k] = sub[m].sum(axis=0)
    out = np.sqrt(out)
    norm = out.sum(axis=0, keepdims=True)
    return out / np.maximum(norm, 1e-9)


def mfcc(x: np.ndarray, sr: int, n_mfcc: int = 13, hop: int = HOP) -> np.ndarray:
    """MFCC timbre features via DCT-II of the log-mel spectrogram."""
    from scipy.fftpack import dct

    lm = log_mel(x, sr, n_mels=40, hop=hop)
    return dct(lm, type=2, axis=0, norm="ortho")[:n_mfcc]


def rms_envelope(x: np.ndarray, hop: int = HOP, win: int = N_FFT) -> np.ndarray:
    """Frame-wise RMS of a time-domain signal at the feature hop."""
    n_frames = 1 + max(0, (len(x) - win)) // hop
    if n_frames <= 0:
        return np.array([float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0])
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.sqrt(np.mean(x[idx] ** 2, axis=1))
