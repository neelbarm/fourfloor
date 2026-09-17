"""The map between source time and remix time, and the one place it is built.

Warping is where a remix is won or lost. The source is laid on a perfectly
periodic target grid by pinning every detected source beat to a target beat, so
after the warp the song's own pulse *is* the grid. The arranger then addresses
that warped audio in seconds -- "start this drop 27.7 seconds into the warped
source" -- and the engine drops those seconds onto a bar line.

That only works if everyone agrees on what "27.7 seconds into the warped source"
means. Before this module existed they did not. The warp put the source's first
beat one target beat into the buffer and the arranger assumed source time zero
was warped time zero, so every slot read the source from a point that was, on
Don Toliver's *Body*, 0.88 of a beat away from the downbeat it was supposed to
be. The kick landed on the grid, the song landed nearly a beat off it, and the
remix sounded drunk. Nothing in the code was wrong on its own; the two halves
simply disagreed, silently, and no test could see it.

So there is exactly one map, built once, and both halves read it:

* ``WarpMap.from_grid`` decides where every source beat goes, and picks the
  lead-in so that the source's *first downbeat lands on a target bar line*. Bar
  one of the song is bar one of the remix, by construction rather than by luck.
* ``build`` hands that map to the phase vocoder and returns the warped audio
  next to it.
* ``WarpMap.snap`` is what the arranger calls: give it a moment in the source
  and it returns the warped second of the nearest source downbeat that sits on a
  bar line, so a slot can never start mid-bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: How close a warped downbeat has to be to a bar line to count as on it.
#: The construction below makes them exact; this is only a guard against
#: floating-point dust.
BAR_TOL = 1e-6


@dataclass(frozen=True)
class WarpMap:
    """A piecewise-linear map from source seconds to warped (remix) seconds.

    ``in_times`` and ``out_times`` are the knots: source beat *j* is pinned to
    ``out_times`` so the warped audio runs at exactly ``target_bpm``. Everything
    else is interpolation.
    """

    in_times: np.ndarray
    out_times: np.ndarray
    target_bpm: float
    beat_multiple: float
    source_downbeats: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lead_beats: int = 0

    # -- geometry --------------------------------------------------------
    @property
    def beat(self) -> float:
        """One target beat in seconds."""
        return 60.0 / self.target_bpm

    @property
    def bar_dur(self) -> float:
        """One target bar (4/4) in seconds."""
        return 4.0 * self.beat

    @property
    def duration(self) -> float:
        """Length of the warped timeline in seconds."""
        return float(self.out_times[-1]) if len(self.out_times) else 0.0

    # -- mapping ---------------------------------------------------------
    def to_warped(self, t):
        """Source seconds -> warped seconds."""
        return np.interp(t, self.in_times, self.out_times)

    def to_source(self, w):
        """Warped seconds -> source seconds (the inverse map)."""
        return np.interp(w, self.out_times, self.in_times)

    # -- downbeats -------------------------------------------------------
    @property
    def downbeats(self) -> np.ndarray:
        """Every source downbeat, in warped seconds."""
        if not len(self.source_downbeats):
            return np.zeros(0)
        return np.asarray(self.to_warped(self.source_downbeats), dtype=float)

    @property
    def bar_downbeats(self) -> np.ndarray:
        """The source downbeats that land on a target bar line, snapped exact.

        With ``beat_multiple`` of 1 or 2 that is all of them. At 0.5 -- a source
        read double-time, where one source bar is half a target bar -- it is
        every other one, which is the right answer: a house bar has to start
        where the song's bar starts, not halfway through it.
        """
        w = self.downbeats
        if not len(w):
            return np.zeros(0)
        bars = w / self.bar_dur
        keep = np.abs(bars - np.round(bars)) < max(BAR_TOL / self.bar_dur, 1e-6)
        return np.round(bars[keep]) * self.bar_dur

    def snap(self, source_time: float) -> float:
        """Warped second of the bar-aligned source downbeat nearest ``source_time``.

        This is the only way a slot should ever get a ``source_start``. Anything
        else puts the song's bar line somewhere inside the remix's bar.
        """
        cand = self.bar_downbeats
        w = float(self.to_warped(source_time))
        if not len(cand):
            return round(w / self.bar_dur) * self.bar_dur
        return float(cand[int(np.argmin(np.abs(cand - w)))])

    def bars_between(self, source_start: float, source_end: float) -> int:
        """How many whole target bars a span of source time becomes."""
        span = float(self.to_warped(source_end)) - float(self.to_warped(source_start))
        return max(1, int(round(span / self.bar_dur)))

    # -- construction ----------------------------------------------------
    @classmethod
    def from_grid(cls, beats: np.ndarray, downbeat_index: int, target_bpm: float,
                  beat_multiple: float, source_duration: float) -> "WarpMap":
        """Pin a detected beat grid onto a target grid, bar one on bar one.

        The lead-in is the whole point. Source beat ``j`` is placed at
        ``(j - downbeat_index + lead) * beat_multiple`` target beats, and
        ``lead`` is chosen as the smallest multiple of four (eight when a source
        bar is only half a target bar) that is big enough to hold whatever
        happens before the first detected beat. Because ``lead`` is a multiple
        of four, the source's downbeats land on the target's downbeats -- not
        approximately, exactly.

        The head and the tail are extended at the local tempo rather than being
        squeezed into one beat each, which is what the old construction did: a
        track whose beat tracking stopped ten seconds before the file ended had
        those ten seconds crammed into half a second of warped audio.
        """
        beats = np.asarray(beats, dtype=float)
        tb = 60.0 / max(target_bpm, 1e-6)
        step = tb * beat_multiple                   # one source beat, warped
        if len(beats) < 2:
            rate = 1.0
            return cls(in_times=np.array([0.0, max(source_duration, 1.0)]),
                       out_times=np.array([0.0, max(source_duration, 1.0) * rate]),
                       target_bpm=target_bpm, beat_multiple=beat_multiple)

        period = float(np.median(np.diff(beats)))
        rate = step / max(period, 1e-9)             # warped seconds per source second
        db = int(np.clip(downbeat_index, 0, 3))
        grain = 8 if beat_multiple < 1.0 else 4
        lead = grain * int(np.ceil((db + 1) / grain))
        while (lead - db) * step < beats[0] * rate:
            lead += grain

        out = (np.arange(len(beats)) - db + lead) * step
        head_in, head_out = 0.0, out[0] - beats[0] * rate
        tail_in = max(source_duration, float(beats[-1]) + period)
        tail_out = out[-1] + (tail_in - float(beats[-1])) * rate
        in_times = np.concatenate([[head_in], beats, [tail_in]])
        out_times = np.concatenate([[max(head_out, 0.0)], out, [tail_out]])
        # np.interp needs strictly increasing knots
        out_times = np.maximum.accumulate(out_times)
        return cls(in_times=in_times, out_times=out_times, target_bpm=target_bpm,
                   beat_multiple=beat_multiple,
                   source_downbeats=beats[db::4], lead_beats=lead)

    @classmethod
    def from_analysis(cls, analysis, target_bpm: float, beat_multiple: float) -> "WarpMap":
        """The map an :class:`~fourfloor.analysis.Analysis` implies."""
        g = analysis.grid
        return cls.from_grid(g.beats, g.downbeat_index, target_bpm, beat_multiple,
                             analysis.duration)


def build(analysis, target_bpm: float, beat_multiple: float, semitones: int = 0,
          ) -> tuple[np.ndarray, WarpMap]:
    """Warp a source onto the target grid and return it with its map.

    The beat map goes straight to the phase vocoder as a piecewise time-warp, so
    tempo drift inside the source is absorbed beat by beat instead of by one
    global rate. When a pitch shift is asked for, the warp targets a grid
    ``2**(n/12)`` times longer and the resample afterwards brings it back to
    length while moving every partial by the interval -- so the map returned
    here is always in final, post-resample seconds.
    """
    from .dsp import phasevocoder as PV
    from .dsp.pitch import resample_ratio

    x = analysis.clip.samples
    sr = analysis.sr
    wm = WarpMap.from_analysis(analysis, target_bpm, beat_multiple)
    ratio_pitch = 2.0 ** (semitones / 12.0)

    if len(analysis.grid.beats) < 2:
        # No usable grid: one global stretch is all that is left.
        ratio = max(analysis.grid.bpm, 1e-6) / max(target_bpm * beat_multiple, 1e-6)
        y = PV.time_stretch(x, ratio * ratio_pitch)
        if semitones:
            y = resample_ratio(y, 1.0 / ratio_pitch)
        return y, wm

    stretched = PV.warp(x, wm.in_times, wm.out_times * ratio_pitch, sr,
                        int(round(wm.duration * ratio_pitch * sr)))
    if semitones:
        stretched = resample_ratio(stretched, 1.0 / ratio_pitch)
    return stretched, wm
