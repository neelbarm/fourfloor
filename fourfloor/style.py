"""Learn a house style profile from a folder of reference remixes.

Every reference is analysed for the things that actually distinguish one
producer's house edits from another's: working tempo, hi-hat swing, how many
bars of DJ intro run before the four-on-the-floor kick locks in, whether there
is a breakdown and how long it is, spectral tilt, loudness and kick density.
The aggregate becomes defaults for ``fourfloor remix --style``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal as sps

from .analysis import F, analyze
from .audio import decode, find_audio, rms_db
from .dsp import dynamics as DY
from .dsp.hpss import hpss_stereo
from .dsp.pitch import plan_tempo

SCHEMA_VERSION = 1


@dataclass
class Style:
    """Aggregate profile learned from a reference folder."""

    bpm: float = 124.0
    bpm_spread: float = 0.0
    swing: float = 0.08
    intro_bars: int = 16
    breakdown_ratio: float = 1.0
    breakdown_bars: int = 16
    spectral_tilt: float = 0.0
    brightness_hz: float = 2500.0
    rms_db: float = -9.0
    peak_db: float = -0.5
    kick_density: float = 1.0
    length: float = 270.0
    n_refs: int = 0
    per_track: list[dict] = field(default_factory=list)

    # --- learned from original/remix pairs, when the folder has any
    n_pairs: int = 0
    tempo_ratio: float = 0.0           # remix BPM / original BPM
    beat_multiple: float = 0.0         # 0.5, 1 or 2; 0 when nothing was learned
    semitone_shift: float = 0.0        # chroma rotation, original -> remix
    vocal_band_margin_db: float = 0.0  # remix: harmonic vs percussive, 300 Hz-4 kHz
    presence_margin_db: float = 0.0    # the same comparison at 2-5 kHz
    per_pair: list[dict] = field(default_factory=list)

    def to_dict(self, anonymous: bool = False) -> dict:
        d = asdict(self)
        d["schema"] = SCHEMA_VERSION
        if anonymous:
            d.pop("per_track", None)
            d.pop("per_pair", None)
            d["n_refs"] = self.n_refs
            d["n_pairs"] = self.n_pairs
        return d

    @classmethod
    def load(cls, path: str | Path) -> "Style":
        try:
            data = json.loads(Path(path).read_text(encoding="utf8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not a valid style profile: {exc}") from None
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a valid style profile: expected a JSON object")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path, anonymous: bool = False) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(anonymous), indent=2) + "\n",
                        encoding="utf8")
        return path


def measure_swing(x: np.ndarray, sr: int, bpm: float, beat_offset: float) -> float:
    """Estimate hi-hat swing as the mean lateness of the odd 16ths.

    High-band (>6 kHz) onsets are folded onto the 16th-note grid; the offbeat
    16ths of a swung pattern sit late, and the mean of that lateness expressed
    as a fraction of a 16th is the swing amount. 0 is straight, 0.66 is triplet.
    """
    fps = F.frame_rate(sr)
    hi = F.band_energy(x, sr, 6000.0, 16000.0)
    flux = np.maximum(0.0, np.diff(hi, prepend=hi[:1]))
    if flux.max() <= 0:
        return 0.0
    flux /= flux.max()
    peaks, _ = sps.find_peaks(flux, height=0.12, distance=max(1, int(0.03 * fps)))
    if len(peaks) < 8:
        return 0.0
    step = 60.0 / bpm / 4.0          # one 16th
    times = peaks / fps - beat_offset
    pos = np.mod(times / step, 2.0)  # position within a pair of 16ths
    odd = pos[(pos > 0.5) & (pos < 1.5)]
    if len(odd) < 4:
        return 0.0
    return float(np.clip(np.median(odd) - 1.0, -0.2, 0.66))


def find_kick_lock(x: np.ndarray, sr: int, bpm: float) -> float:
    """Seconds before a sustained four-on-the-floor kick pattern starts.

    Inside a sliding 4-bar window the low band is correlated with itself at one
    beat of lag; a four-on-the-floor kick scores near 1, a sparser or absent
    kick scores low. The intro ends at the first window that clears both a
    relative threshold (70% of the track's own best) and an absolute floor, and
    whose successor also clears them -- the two-window requirement is what stops
    a single loud intro hit from being read as the drop.

    Returns 0.0 for a track that starts on the kick, which is common in the
    "extended mix" edits DJs trade.
    """
    fps = F.frame_rate(sr)
    low = F.band_energy(x, sr, 30.0, 110.0)
    if low.max() <= 0:
        return 0.0
    low = low / low.max()
    beat_frames = 60.0 / bpm * fps
    win = int(round(beat_frames * 16))
    if win < 8 or len(low) < win * 2:
        return 0.0
    lag = int(round(beat_frames))
    step = max(1, win // 8)
    scores: list[tuple[int, float]] = []
    for s in range(0, len(low) - win, step):
        seg = low[s:s + win]
        seg = seg - seg.mean()
        denom = float(np.sum(seg * seg))
        if denom <= 1e-9:
            scores.append((s, 0.0))
            continue
        # also require the window to carry real low-end level, so a quiet
        # filtered intro that happens to be periodic does not count as the drop
        level = float(np.mean(low[s:s + win]))
        corr = float(np.sum(seg[:-lag] * seg[lag:])) / denom
        scores.append((s, corr * min(1.0, level / 0.25)))
    if len(scores) < 2:
        return 0.0
    best = max(sc for _, sc in scores)
    if best <= 0.05:
        return 0.0
    floor = max(0.7 * best, 0.2)
    for i in range(len(scores) - 1):
        if scores[i][1] >= floor and scores[i + 1][1] >= floor:
            return float(scores[i][0] / fps)
    return 0.0


def spectral_tilt(x: np.ndarray, sr: int) -> tuple[float, float]:
    """Return (tilt dB/decade, spectral centroid Hz).

    Tilt is the slope of a least-squares line through the log-power spectrum
    against log frequency: how bright the master is, independent of level.
    """
    mag = np.abs(F.stft(x, F.N_FFT, F.HOP))
    freqs = np.fft.rfftfreq(F.N_FFT, 1.0 / sr)
    power = np.mean(mag ** 2, axis=1)
    sel = (freqs > 60.0) & (freqs < 16000.0) & (power > 0)
    if sel.sum() < 8:
        return 0.0, 0.0
    lf = np.log10(freqs[sel])
    lp = 10.0 * np.log10(power[sel])
    slope = float(np.polyfit(lf, lp, 1)[0])
    centroid = float(np.sum(freqs[sel] * power[sel]) / max(np.sum(power[sel]), 1e-12))
    return slope, centroid


def profile_track(path: str | Path) -> dict:
    """Measure one reference remix."""
    clip = decode(path)
    a = analyze(path, clip=clip)
    mono = clip.mono
    bpm = a.grid.bpm
    bar = 4.0 * 60.0 / bpm
    swing = measure_swing(mono, clip.sr, bpm, a.grid.first_downbeat)
    intro_sec = find_kick_lock(mono, clip.sr, bpm)
    tilt, centroid = spectral_tilt(mono, clip.sr)

    low = F.band_energy(mono, clip.sr, 30.0, 110.0)
    fps = F.frame_rate(clip.sr)
    thr = float(np.percentile(low, 80))
    peaks, _ = sps.find_peaks(low, height=thr, distance=max(1, int(0.3 * fps)))
    kicks_per_beat = len(peaks) / max(clip.duration / (60.0 / bpm), 1e-6)

    quiet = [s for s in a.sections if s.energy < 0.6]
    breakdown_sec = max((s.duration for s in quiet), default=0.0)

    return {
        "bpm": round(bpm, 2),
        "key": a.key.name,
        "camelot": a.key.camelot,
        "swing": round(swing, 4),
        "intro_sec": round(intro_sec, 2),
        "intro_bars": int(round(intro_sec / bar)),
        "breakdown_sec": round(breakdown_sec, 2),
        "breakdown_bars": int(round(breakdown_sec / bar)),
        "has_breakdown": breakdown_sec > bar * 3,
        "spectral_tilt": round(tilt, 3),
        "centroid_hz": round(centroid, 1),
        "rms_db": round(rms_db(clip.samples), 2),
        "peak_db": round(float(20 * np.log10(max(float(np.max(np.abs(clip.samples))), 1e-6))), 2),
        "kick_density": round(kicks_per_beat, 3),
        "duration": round(clip.duration, 1),
    }


def learn(folder: str | Path, progress=None) -> Style:
    """Analyse every audio file in ``folder`` and aggregate a Style."""
    pair_rows, pair_stats = learn_pairs(folder, progress)
    paired = {p for _, o, r in find_pairs(folder) for p in (o, r)}
    files = [f for f in find_audio(folder) if f not in paired]
    remixes = [r for _, _, r in find_pairs(folder)]
    files = files + [r for r in remixes if r not in files]
    if not files:
        raise RuntimeError(f"no audio files in {folder}")
    step = progress or (lambda *_a, **_k: None)
    rows: list[dict] = []
    for i, f in enumerate(files, 1):
        step("learn", f"[{i}/{len(files)}] {f.name}")
        row = profile_track(f)
        row["file"] = f.name
        rows.append(row)

    def med(key: str) -> float:
        return float(np.median([r[key] for r in rows]))

    bpms = [r["bpm"] for r in rows]
    breakdowns = [r for r in rows if r["has_breakdown"]]
    return Style(
        bpm=round(med("bpm"), 2),
        bpm_spread=round(float(np.std(bpms)), 2),
        swing=round(med("swing"), 4),
        intro_bars=int(round(med("intro_bars") / 8.0) * 8) or 8,
        breakdown_ratio=round(len(breakdowns) / len(rows), 3),
        breakdown_bars=int(round(
            (np.median([r["breakdown_bars"] for r in breakdowns]) if breakdowns else 16)
            / 8.0) * 8) or 8,
        spectral_tilt=round(med("spectral_tilt"), 3),
        brightness_hz=round(med("centroid_hz"), 1),
        rms_db=round(med("rms_db"), 2),
        peak_db=round(med("peak_db"), 2),
        kick_density=round(med("kick_density"), 3),
        length=round(med("duration"), 1),
        n_refs=len(rows),
        per_track=rows,
        per_pair=pair_rows,
        **pair_stats,
    )


# ---------------------------------------------------------------------------
# original / remix pairs
# ---------------------------------------------------------------------------

#: How a pair is named on disk. ``fourfloor fetch --pair`` writes this layout
#: and ``fourfloor learn`` reads it: two files whose names differ only in the
#: marker, in the folder being learned from or in a ``pairs`` subfolder of it.
PAIR_MARKERS = ("original", "remix")

#: How much better than "no shift" a rotation has to score before it is
#: reported as a transposition.
SHIFT_MARGIN = 0.03


def find_pairs(folder: str | Path) -> list[tuple[str, Path, Path]]:
    """Every ``<name>.original.<ext>`` / ``<name>.remix.<ext>`` couple in a folder.

    A pair is worth far more than two unrelated references: it is the only way
    to see what a remixer *changed* -- the tempo they landed on, whether they
    moved the key, how far they pushed the vocal against the kit -- rather than
    what a finished house record happens to look like.

    Both the folder itself and a ``pairs`` subfolder are searched, because that
    is where ``fourfloor fetch --pair`` puts them.
    """
    root = Path(folder)
    found: dict[str, dict[str, Path]] = {}
    for base in (root, root / "pairs"):
        if not base.is_dir():
            continue
        for f in find_audio(base):
            stem = f.stem
            for marker in PAIR_MARKERS:
                if stem.lower().endswith(f".{marker}"):
                    name = stem[: -len(marker) - 1]
                    found.setdefault(name, {})[marker] = f
    return [(name, sides["original"], sides["remix"])
            for name, sides in sorted(found.items())
            if set(sides) == set(PAIR_MARKERS)]


def chroma_shift(original: np.ndarray, remix: np.ndarray) -> int:
    """Semitones from one track to another, by chroma-rotation correlation.

    Asking which of the twelve rotations correlates best answers "was this
    transposed" directly, without depending on both key detections
    independently landing on the right tonic.

    Two details decide whether it works. Each frame is normalised before the
    average, so loud bars do not outvote quiet ones -- without that, the A/B
    pair used to develop this reads as transposed a fourth when both tracks are
    in fact in C minor. And the winner has to beat "no shift" by a margin:
    claiming a transposition that did not happen is a worse answer than
    claiming none, because a remix that stays in key is the common case.
    """
    def profile(chroma: np.ndarray) -> np.ndarray:
        c = np.asarray(chroma, dtype=float)
        c = c / np.maximum(np.linalg.norm(c, axis=0, keepdims=True), 1e-9)
        v = c.mean(axis=1)
        v = v - v.mean()
        return v / max(float(np.linalg.norm(v)), 1e-9)

    a, b = profile(original), profile(remix)
    scores = [float(np.dot(np.roll(a, k), b)) for k in range(12)]
    best = int(np.argmax(scores))
    if scores[best] - scores[0] < SHIFT_MARGIN:
        return 0
    return best - 12 if best > 6 else best


def profile_pair(original: str | Path, remix: str | Path) -> dict:
    """What one remixer did to one song.

    The balance numbers are measured on the remix with fourfloor's own
    harmonic/percussive split rather than with a separation model, so that what
    is learned here is directly comparable to what the engine measures while it
    is mixing -- a target and a measurement that disagree about how the signal
    was divided are not a target.
    """
    a = analyze(original)
    b = analyze(remix)
    tempo = plan_tempo(a.grid.bpm, b.grid.bpm)
    harm, perc = hpss_stereo(b.clip.samples, b.sr)
    return {
        "name": Path(remix).stem,
        "original_bpm": round(a.grid.bpm, 2),
        "remix_bpm": round(b.grid.bpm, 2),
        "tempo_ratio": round(b.grid.bpm / max(a.grid.bpm, 1e-6), 4),
        "beat_multiple": tempo.beat_multiple,
        "stretch_ratio": round(tempo.ratio, 4),
        "semitone_shift": chroma_shift(a.chroma, b.chroma),
        "original_key": a.key.name,
        "remix_key": b.key.name,
        "vocal_band_margin_db": round(
            DY.band_rms_db(harm, b.sr, DY.VOCAL_BAND)
            - DY.band_rms_db(perc, b.sr, DY.VOCAL_BAND), 2),
        "presence_margin_db": round(
            DY.band_rms_db(harm, b.sr, DY.PRESENCE_BAND)
            - DY.band_rms_db(perc, b.sr, DY.PRESENCE_BAND), 2),
        "length_ratio": round(b.duration / max(a.duration, 1e-6), 3),
    }


def learn_pairs(folder: str | Path, progress=None) -> tuple[list[dict], dict]:
    """Profile every pair in a folder and return ``(rows, aggregate)``."""
    pairs = find_pairs(folder)
    step = progress or (lambda *_a, **_k: None)
    rows = []
    for i, (name, original, remix) in enumerate(pairs, 1):
        step("learn", f"[pair {i}/{len(pairs)}] {name}")
        rows.append(profile_pair(original, remix))
    if not rows:
        return [], {}

    def med(key: str) -> float:
        return float(np.median([r[key] for r in rows]))

    return rows, {
        "n_pairs": len(rows),
        "tempo_ratio": round(med("tempo_ratio"), 4),
        "beat_multiple": float(np.median([r["beat_multiple"] for r in rows])),
        "semitone_shift": round(med("semitone_shift"), 2),
        "vocal_band_margin_db": round(med("vocal_band_margin_db"), 2),
        "presence_margin_db": round(med("presence_margin_db"), 2),
    }
