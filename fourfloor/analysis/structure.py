"""Structural segmentation: boundaries, section labels, per-section loudness.

Boundaries come from Foote's checkerboard novelty on a self-similarity matrix
built from stacked chroma + MFCC features (J. Foote, "Automatic audio
segmentation using a measure of audio novelty", ICME 2000). Labels come from
agglomerative grouping of the resulting segments by feature distance, then a
naming pass: the most-repeated high-energy group becomes the hook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from . import features as F


@dataclass
class Section:
    """One structural span of the source track."""

    start: float
    end: float
    label: str
    group: int
    energy: float          # 0-1, normalised RMS
    rms_db: float
    repeats: int = 1

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "label": self.label,
            "group": self.group,
            "energy": round(self.energy, 3),
            "rms_db": round(self.rms_db, 2),
            "repeats": self.repeats,
        }


def _checkerboard(size: int) -> np.ndarray:
    """Gaussian-tapered checkerboard kernel of half-width ``size`` (Foote 2000)."""
    n = 2 * size
    g = np.outer(*(2 * [np.exp(-0.5 * (np.linspace(-2, 2, n)) ** 2)]))
    sign = np.ones((n, n))
    sign[:size, size:] = -1.0
    sign[size:, :size] = -1.0
    return g * sign


def _stack(feat: np.ndarray, lag: int = 4, step: int = 2) -> np.ndarray:
    """Time-delay embedding: stack ``lag`` frames spaced ``step`` apart."""
    cols = [np.roll(feat, -i * step, axis=1) for i in range(lag)]
    return np.vstack(cols)[:, : feat.shape[1] - (lag - 1) * step or None]


def novelty_curve(chroma: np.ndarray, timbre: np.ndarray, kernel: int = 32) -> np.ndarray:
    """Foote novelty from a cosine self-similarity matrix of stacked features."""
    def norm_rows(a: np.ndarray) -> np.ndarray:
        a = a - a.mean(axis=1, keepdims=True)
        s = a.std(axis=1, keepdims=True)
        return a / np.maximum(s, 1e-9)

    n = min(chroma.shape[1], timbre.shape[1])
    feat = np.vstack([norm_rows(chroma[:, :n]), 0.5 * norm_rows(timbre[:, :n])])
    feat = _stack(feat)
    norms = np.linalg.norm(feat, axis=0, keepdims=True)
    unit = feat / np.maximum(norms, 1e-9)
    ssm = unit.T @ unit
    m = ssm.shape[0]
    k = min(kernel, max(4, m // 8))
    kern = _checkerboard(k)
    nov = np.zeros(m)
    for i in range(k, m - k):
        nov[i] = float(np.sum(ssm[i - k:i + k, i - k:i + k] * kern))
    nov = np.maximum(nov, 0.0)
    if nov.max() > 0:
        nov /= nov.max()
    return nov


def _snap(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Snap each time to the nearest grid point (downbeats), keeping order."""
    if not len(grid):
        return times
    snapped = np.array([grid[int(np.argmin(np.abs(grid - t)))] for t in times])
    return np.unique(snapped)


def segment(x: np.ndarray, sr: int, downbeats: np.ndarray, bar_dur: float,
            chroma: np.ndarray | None = None,
            min_bars: int = 4) -> list[Section]:
    """Segment a track into labelled sections aligned to downbeats.

    ``min_bars`` is the shortest section we allow; house arrangements work in
    4- and 8-bar phrases so anything shorter is merged into its neighbour. It
    scales with the length of the track: a full song is described better by ten
    8-bar-minimum sections than by twenty 4-bar ones, while a short loop would
    collapse to a single block under the same rule.
    """
    fps = F.frame_rate(sr)
    total_bars = max(1, int((len(x) / sr) / max(bar_dur, 1e-6)))
    min_bars = min(min_bars * 2, max(min_bars, total_bars // 8))
    chroma = F.chromagram(x, sr) if chroma is None else chroma
    timbre = F.mfcc(x, sr)
    nov = novelty_curve(chroma, timbre)

    min_dist = max(2, int(round(min_bars * bar_dur * fps)))
    height = float(np.percentile(nov[nov > 0], 55)) if np.any(nov > 0) else 0.0
    peaks, _ = sps.find_peaks(nov, distance=min_dist, height=height)
    bounds = np.concatenate([[0.0], peaks / fps, [len(x) / sr]])
    if len(downbeats):
        inner = _snap(bounds[1:-1], downbeats) if len(bounds) > 2 else np.array([])
        bounds = np.unique(np.concatenate([[0.0], inner, [len(x) / sr]]))

    # merge anything shorter than min_bars
    merged = [float(bounds[0])]
    for b in bounds[1:]:
        if b - merged[-1] >= min_bars * bar_dur * 0.9 or b == bounds[-1]:
            merged.append(float(b))
    if len(merged) > 2 and merged[-1] - merged[-2] < min_bars * bar_dur * 0.9:
        merged.pop(-2)
    bounds = np.asarray(merged)
    if len(bounds) < 2:
        bounds = np.array([0.0, len(x) / sr])

    # per-segment descriptors
    descs, energies, rmsdbs = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        fa, fb = int(a * fps), max(int(b * fps), int(a * fps) + 1)
        ca = chroma[:, fa:min(fb, chroma.shape[1])]
        ta = timbre[:, fa:min(fb, timbre.shape[1])]
        cv = ca.mean(axis=1) if ca.size else np.zeros(12)
        tv = ta.mean(axis=1) if ta.size else np.zeros(timbre.shape[0])
        descs.append(np.concatenate([cv / max(cv.sum(), 1e-9), tv / max(np.abs(tv).max(), 1e-9)]))
        seg = x[int(a * sr):int(b * sr)]
        r = float(np.sqrt(np.mean(np.square(seg)))) if len(seg) else 1e-6
        energies.append(r)
        rmsdbs.append(20.0 * np.log10(max(r, 1e-6)))
    D = np.asarray(descs)
    emax = max(max(energies), 1e-9)

    # group similar segments (single-link at a distance threshold)
    n = len(D)
    groups = list(range(n))
    if n > 1:
        dist = np.linalg.norm(D[:, None, :] - D[None, :, :], axis=2)
        iu = np.triu_indices(n, 1)
        thr = float(np.percentile(dist[iu], 30)) if len(iu[0]) else 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] <= thr:
                    tgt, src = groups[i], groups[j]
                    if tgt != src:
                        groups = [tgt if g == src else g for g in groups]
    counts = {g: groups.count(g) for g in set(groups)}

    sections: list[Section] = []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        sections.append(Section(
            start=float(a), end=float(b), label="", group=groups[i],
            energy=float(energies[i] / emax), rms_db=float(rmsdbs[i]),
            repeats=counts[groups[i]],
        ))
    _label(sections)
    return sections


def _label(sections: list[Section]) -> None:
    """Name sections: hook = most repeated high-energy group, then verse/etc."""
    if not sections:
        return
    by_group: dict[int, list[Section]] = {}
    for s in sections:
        by_group.setdefault(s.group, []).append(s)

    def group_score(items: list[Section]) -> float:
        return float(np.mean([s.energy for s in items])) * (1.0 + 0.35 * (len(items) - 1))

    ranked = sorted(by_group.items(), key=lambda kv: -group_score(kv[1]))
    hook_group = ranked[0][0]
    verse_group = ranked[1][0] if len(ranked) > 1 else hook_group

    for s in sections:
        if s.group == hook_group:
            s.label = "hook"
        elif s.group == verse_group:
            s.label = "verse"
        elif s.energy < 0.45:
            s.label = "breakdown"
        else:
            s.label = "section"
    # the first and last low-energy spans read as intro/outro
    if sections[0].energy < 0.7 and sections[0].label != "hook":
        sections[0].label = "intro"
    if len(sections) > 1 and sections[-1].energy < 0.7 and sections[-1].label != "hook":
        sections[-1].label = "outro"


def energy_per_bar(x: np.ndarray, sr: int, downbeats: np.ndarray, bar_dur: float) -> list[float]:
    """Normalised RMS energy for each bar, for the session file's energy curve."""
    vals = []
    for t in downbeats:
        a, b = int(t * sr), int((t + bar_dur) * sr)
        seg = x[max(0, a):min(len(x), b)]
        vals.append(float(np.sqrt(np.mean(np.square(seg)))) if len(seg) else 0.0)
    m = max(vals) if vals else 1.0
    return [round(v / max(m, 1e-9), 4) for v in vals]
