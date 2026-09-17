"""Track analysis: tempo, beat grid, key, chords, structure, loudness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..audio import SR, Clip, decode
from . import features as F
from .key import KeyEstimate, chords_per_bar, detect_key
from .structure import Section, energy_per_bar, segment
from .tempo import BeatGrid, analyze_beats

__all__ = [
    "Analysis", "analyze", "BeatGrid", "KeyEstimate", "Section", "F",
    "suggest_house_tempo",
]


def suggest_house_tempo(bpm: float) -> float:
    """Nearest sensible house tempo for a source BPM.

    The band is 124-132, not the 120-128 this used to use. Six commercial house
    remixes measure a median 131 BPM [125.2 .. 132.9], and the one A/B pair
    available settles the tie-break: a 146 BPM source became a 130 BPM remix, an
    11% stretch, where the old band's best answer was 124 and a 15% one. The
    rule the remixer followed was minimum stretch, so that is the rule here --
    the pull toward the middle of the band is only strong enough to break a
    genuine tie.
    """
    band = np.arange(124.0, 132.5, 1.0)
    best, best_cost = 128.0, np.inf
    for t in band:
        cost = min(abs(np.log2(t / max(bpm, 1e-6))),
                   abs(np.log2(t / max(2 * bpm, 1e-6))),
                   abs(np.log2(2 * t / max(bpm, 1e-6))))
        cost += 0.002 * abs(t - 128.0)
        if cost < best_cost:
            best, best_cost = float(t), cost
    return best


@dataclass
class Analysis:
    """Everything fourfloor knows about a source track."""

    path: str
    duration: float
    sr: int
    grid: BeatGrid
    key: KeyEstimate
    sections: list[Section]
    chords: list[dict]
    rms_db: float
    peak_db: float
    clip: Clip | None = field(default=None, repr=False)
    chroma: np.ndarray | None = field(default=None, repr=False)
    onset_env: np.ndarray | None = field(default=None, repr=False)

    @property
    def bar_dur(self) -> float:
        """Duration of one 4/4 bar in seconds at the source tempo."""
        return 4.0 * 60.0 / self.grid.bpm

    def section_by_label(self, label: str) -> list[Section]:
        return [s for s in self.sections if s.label == label]

    def hook(self) -> Section:
        """The section a remix should build its drops from."""
        hooks = self.section_by_label("hook")
        if hooks:
            return max(hooks, key=lambda s: s.duration)
        return max(self.sections, key=lambda s: s.energy * s.duration)

    def to_dict(self) -> dict:
        return {
            "file": Path(self.path).name,
            "duration": round(self.duration, 3),
            "sample_rate": self.sr,
            "tempo": self.grid.to_dict(),
            "suggested_house_bpm": suggest_house_tempo(self.grid.bpm),
            "key": self.key.to_dict(),
            "loudness": {"rms_db": round(self.rms_db, 2), "peak_db": round(self.peak_db, 2)},
            "sections": [s.to_dict() for s in self.sections],
            "chords": self.chords,
            "energy_per_bar": energy_per_bar(
                self.clip.mono if self.clip is not None else np.zeros(1),
                self.sr, self.grid.downbeats, self.bar_dur,
            ) if self.clip is not None else [],
        }


def analyze(path: str | Path, clip: Clip | None = None, keep_audio: bool = True) -> Analysis:
    """Run the full analysis chain on a file (or a pre-decoded clip)."""
    clip = clip if clip is not None else decode(path)
    mono = clip.mono
    grid = analyze_beats(mono, clip.sr)
    tuning = F.estimate_tuning(mono, clip.sr)
    chroma = F.chromagram(mono, clip.sr, tuning=tuning)
    key = detect_key(chroma, tuning)
    bar_dur = 4.0 * 60.0 / grid.bpm
    sections = segment(mono, clip.sr, grid.downbeats, bar_dur, chroma=chroma)
    chords = chords_per_bar(chroma, F.frame_rate(clip.sr), grid.downbeats, bar_dur)
    rms = 20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(mono)))), 1e-6))
    peak = 20.0 * np.log10(max(float(np.max(np.abs(mono))), 1e-6))
    return Analysis(
        path=str(path), duration=clip.duration, sr=clip.sr, grid=grid, key=key,
        sections=sections, chords=chords, rms_db=rms, peak_db=peak,
        clip=clip if keep_audio else None, chroma=chroma,
        onset_env=F.onset_strength(mono, clip.sr),
    )
