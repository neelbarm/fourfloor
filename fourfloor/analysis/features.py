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


#: Window, hop and flux lag for :func:`attack_envelope`.
ATTACK_FFT = 1024
ATTACK_HOP = 64
ATTACK_LAG = 2

#: Residual latency of :func:`attack_envelope` in seconds, measured on
#: synthesised kicks, hats, claps and a whole kit placed exactly on a grid:
#: every one of them reads 2.8-3.5 ms early once the flux's own group delay is
#: accounted for. What is left is the mel filterbank's asymmetric response to an
#: attack. It is a constant, so it is simply added back.
ATTACK_LATENCY = 0.0032


def attack_envelope(x: np.ndarray, sr: int, hop: int = ATTACK_HOP,
                    n_fft: int = ATTACK_FFT, lag: int = ATTACK_LAG,
                    fmax: float | None = None) -> tuple[np.ndarray, float]:
    """A *when exactly* onset envelope, and its frame rate.

    :func:`onset_strength` answers "how strong is the rhythm here", which is what
    tempo estimation and beat tracking need, and it is deliberately smooth: a
    2048-sample window and a logarithmic magnitude. Both push the flux peak ahead
    of the actual attack, because the log lifts the quiet leading edge of the
    window. Measured on a synthesised kit placed exactly on a grid, that envelope
    reads hats 19 ms early and kicks 45 ms early.

    This one answers "when exactly did it hit". Flux over *linear* mel magnitude
    in a 1024-sample window at a 64-sample hop reads the same kit within 3.5 ms
    with a sub-millisecond spread, because linear magnitude is dominated by the
    attack rather than by what precedes it. The group delay of the ``lag``-frame
    difference and the filterbank's own latency are both removed from the time
    base, so frame ``i`` means "an attack at ``i / fps`` seconds".

    ``fmax`` restricts the filterbank to the bottom of the spectrum. That is how
    you ask "where is the kick", as opposed to "where is anything": summed over
    the whole spectrum, a house open hat outruns the kick it sits between by
    five to one, because the hat's attack is spread over forty mel bands and the
    kick's over two.

    Use it to place things. Use :func:`onset_strength` to find the pulse.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < n_fft:
        return np.zeros(0), sr / float(hop)
    mag = np.abs(stft(x, n_fft, hop))
    n_mels = 64 if fmax is None else 16
    bands = mel_filterbank(sr, n_fft, n_mels=n_mels, fmax=fmax) @ mag
    flux = np.maximum(0.0, bands[:, lag:] - bands[:, :-lag]).sum(axis=0)
    flux = np.concatenate([np.zeros(lag), flux])
    fps = sr / float(hop)
    win = max(3, int(round(0.25 * fps)) | 1)
    env = np.maximum(0.0, flux - sps.convolve(flux, np.ones(win) / win, mode="same"))
    peak = env.max()
    return (env / peak if peak > 0 else env), fps


#: How late :func:`kick_envelope` reads, in seconds, measured on a synthesised
#: kick placed exactly on a 124 and a 128 BPM grid. A kick's bottom two octaves
#: take this long to develop, and the envelope is a rise measured over a 46 ms
#: window on top of that. Real kicks scatter roughly +/-10 ms around it, which
#: is why this envelope is used to decide *which* beat, never exactly where.
KICK_LATENCY = 0.018


def kick_envelope(x: np.ndarray, sr: int, hop: int = ATTACK_HOP, lag: int = 8,
                  lo: float = 30.0, hi: float = 130.0) -> tuple[np.ndarray, float]:
    """Where the *kick* hits: rectified rise of the 30-130 Hz band.

    A kick's attack occupies two mel bands and a house open hat occupies forty,
    so any envelope summed over the spectrum is a hi-hat detector with a kick
    somewhere underneath it. Looking only at the bottom of the spectrum, and at
    the *rise* rather than the level -- the level is mostly the bassline, which
    in house plays the offbeats -- gives an envelope that answers "where is the
    four on the floor" and nothing else.

    The lag is longer than :func:`attack_envelope` uses because a kick takes
    tens of milliseconds to develop. The envelope's whole latency, group delay
    and all, is :data:`KICK_LATENCY`, and it is already removed from the time
    base :func:`attack_times` gives, so the two envelopes share one clock.
    """
    low = band_energy(np.asarray(x, dtype=np.float64), sr, lo, hi, hop=hop)
    if len(low) <= lag:
        return np.zeros(0), sr / float(hop)
    flux = np.concatenate([np.zeros(lag), np.maximum(0.0, low[lag:] - low[:-lag])])
    # Slide the whole envelope earlier by its measured latency, so that reading
    # it on the time base `attack_times` hands out means the same thing it means
    # for `attack_envelope`.
    shift = max(0, int(round(KICK_LATENCY * sr / hop)))
    if shift:
        flux = np.concatenate([flux[shift:], np.zeros(shift)])
    peak = flux.max()
    return (flux / peak if peak > 0 else flux), sr / float(hop)


def attack_times(n: int, fps: float, lag: int = ATTACK_LAG) -> np.ndarray:
    """Time base of an :func:`attack_envelope`, group delay and latency removed."""
    return np.arange(n) / fps - lag / (2.0 * fps) + ATTACK_LATENCY


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
