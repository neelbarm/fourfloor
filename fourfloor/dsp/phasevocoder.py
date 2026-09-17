"""Phase vocoder time-stretch with phase locking and transient reset.

The baseline is the standard analysis/synthesis vocoder (Flanagan & Golden 1966;
Laroche & Dolson, "Improved phase vocoder time-scale modification of audio",
IEEE TSAP 7(3), 1999). Two additions matter for remix work:

* **Identity phase locking** (Laroche & Dolson §IV): the synthesis phase is
  propagated only for spectral peaks, and every bin in a peak's region of
  influence inherits that peak's phase rotation. This keeps the partials of a
  voice vertically coherent instead of letting each bin drift, which is what
  produces the classic "phasiness".
* **Transient phase reset**: frames whose spectral flux spikes are re-seeded
  with the original analysis phase, so a kick or a snare is reproduced with its
  true phase alignment instead of a smeared one.

The warp is driven by an arbitrary array of fractional analysis frame positions,
which lets ``warp_to_grid`` do a piecewise stretch that pins every source beat
onto an exactly periodic target grid.
"""

from __future__ import annotations

import numpy as np

from ..analysis.features import N_FFT, HOP, istft, stft


def _transient_frames(mag: np.ndarray, sensitivity: float = 2.2) -> np.ndarray:
    """Boolean mask of frames that look like onsets (rectified spectral flux)."""
    flux = np.maximum(0.0, np.diff(mag, axis=1)).sum(axis=0)
    flux = np.concatenate([[0.0], flux])
    if flux.max() <= 0:
        return np.zeros(mag.shape[1], dtype=bool)
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-9
    return flux > med + sensitivity * 1.4826 * mad


def _peak_regions(col: np.ndarray) -> np.ndarray:
    """Map every bin to the index of the local magnitude peak that owns it.

    A bin is a peak when it exceeds its two neighbours on each side (Laroche &
    Dolson's 4-neighbour rule); the region of influence runs to the midpoint
    between adjacent peaks.
    """
    n = len(col)
    pad = np.pad(col, 2, mode="constant")
    is_peak = ((pad[2:-2] > pad[:-4]) & (pad[2:-2] > pad[1:-3]) &
               (pad[2:-2] > pad[3:-1]) & (pad[2:-2] > pad[4:]))
    peaks = np.flatnonzero(is_peak)
    owner = np.arange(n)
    if not len(peaks):
        return owner
    mids = np.concatenate([[0], (peaks[:-1] + peaks[1:] + 1) // 2, [n]])
    for i, p in enumerate(peaks):
        owner[mids[i]:mids[i + 1]] = p
    return owner


def stretch_frames(spec: np.ndarray, time_steps: np.ndarray, hop: int = HOP,
                   n_fft: int = N_FFT, phase_lock: bool = True,
                   transients: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Resynthesise ``spec`` at fractional analysis positions ``time_steps``.

    Returns ``(output_spectrogram, accumulated_phase)``. The phase trajectory is
    returned so a stereo pair can share one trajectory (see ``_stereo``) and
    keep its inter-channel phase differences, and therefore its width, instead
    of decorrelating into a wash.
    """
    n_bins, n_frames = spec.shape
    mag_all = np.abs(spec)
    phase_all = np.angle(spec)
    omega = 2.0 * np.pi * hop * np.arange(n_bins) / n_fft   # expected per-hop advance

    out = np.zeros((n_bins, len(time_steps)), dtype=complex)
    advance = np.zeros((n_bins, len(time_steps)))
    acc = phase_all[:, 0].copy()

    for i, step in enumerate(time_steps):
        lo = int(np.floor(step))
        frac = step - lo
        hi = min(lo + 1, n_frames - 1)
        lo = min(lo, n_frames - 1)
        mag = (1.0 - frac) * mag_all[:, lo] + frac * mag_all[:, hi]

        if i > 0:
            dphi = phase_all[:, hi] - phase_all[:, lo] - omega
            dphi -= 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))   # principal argument
            acc = acc + omega + dphi
        if transients is not None and transients[lo]:
            acc = phase_all[:, lo].copy()          # hard reset: keep the attack crisp
        if phase_lock:
            owner = _peak_regions(mag)
            acc = acc[owner] + (phase_all[:, lo] - phase_all[:, lo][owner])
        advance[:, i] = acc
        out[:, i] = mag * np.exp(1j * acc)
    return out, advance


def _time_steps_uniform(n_frames: int, rate: float) -> np.ndarray:
    """Analysis positions for a constant-rate stretch."""
    n_out = max(1, int(np.ceil(n_frames / rate)))
    return np.minimum(np.arange(n_out) * rate, n_frames - 1)


def time_stretch(x: np.ndarray, rate: float, hop: int = HOP, n_fft: int = N_FFT,
                 phase_lock: bool = True, preserve_transients: bool = True) -> np.ndarray:
    """Stretch mono or stereo audio by ``rate`` (>1 = faster/shorter).

    Output length is ``round(len(x) / rate)`` exactly.
    """
    if abs(rate - 1.0) < 1e-9:
        return x.copy()
    target = int(round(len(x) / rate))
    if x.ndim == 2:
        return _stereo(x, lambda n: _time_steps_uniform(n, rate), target, hop, n_fft,
                       phase_lock, preserve_transients)
    return _mono(x, lambda n: _time_steps_uniform(n, rate), target, hop, n_fft,
                 phase_lock, preserve_transients)


def warp(x: np.ndarray, in_times: np.ndarray, out_times: np.ndarray, sr: int,
         out_length: int, hop: int = HOP, n_fft: int = N_FFT,
         phase_lock: bool = True, preserve_transients: bool = True) -> np.ndarray:
    """Piecewise time-warp mapping ``in_times`` (source seconds) to ``out_times``.

    This is how a source beat grid gets pinned to a perfectly periodic target
    grid: pass the detected beat times and the target beat times and every beat
    lands exactly on the click, with the tempo drift between beats absorbed
    smoothly rather than as one global rate.
    """
    fps = sr / float(hop)

    def steps(n_frames: int) -> np.ndarray:
        n_out = max(1, int(np.ceil(out_length / hop)) + 1)
        out_sec = np.arange(n_out) / fps
        in_sec = np.interp(out_sec, out_times, in_times)
        return np.clip(in_sec * fps, 0.0, n_frames - 1)

    fn = _stereo if x.ndim == 2 else _mono
    return fn(x, steps, out_length, hop, n_fft, phase_lock, preserve_transients)


def _mono(x, steps_fn, target, hop, n_fft, phase_lock, preserve_transients):
    spec = stft(x, n_fft, hop)
    tr = _transient_frames(np.abs(spec)) if preserve_transients else None
    out, _ = stretch_frames(spec, steps_fn(spec.shape[1]), hop, n_fft, phase_lock, tr)
    y = istft(out, n_fft, hop, length=target)
    return y.astype(np.float32)


def _stereo(x, steps_fn, target, hop, n_fft, phase_lock, preserve_transients):
    mid = x.mean(axis=1)
    spec_m = stft(mid, n_fft, hop)
    ts = steps_fn(spec_m.shape[1])
    tr = _transient_frames(np.abs(spec_m)) if preserve_transients else None
    _, adv = stretch_frames(spec_m, ts, hop, n_fft, phase_lock, tr)
    chans = []
    for c in range(x.shape[1]):
        spec = stft(x[:, c], n_fft, hop)
        # reuse the mid channel's phase trajectory, offset by this channel's own
        # phase at the analysis frame -> inter-channel phase (and width) survives
        n_bins, n_frames = spec.shape
        mag_all, ph_all = np.abs(spec), np.angle(spec)
        lo = np.clip(np.floor(ts).astype(int), 0, n_frames - 1)
        hi = np.clip(lo + 1, 0, n_frames - 1)
        frac = (ts - np.floor(ts))[None, :]
        mag = (1.0 - frac) * mag_all[:, lo] + frac * mag_all[:, hi]
        offset = ph_all[:, lo] - np.angle(spec_m[:, lo])
        out = mag * np.exp(1j * (adv[:, :len(ts)] + offset))
        chans.append(istft(out, n_fft, hop, length=target))
    return np.stack(chans, axis=1).astype(np.float32)
