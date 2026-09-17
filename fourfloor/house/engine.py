"""Render an arrangement Plan into finished audio.

The engine owns the whole signal flow: it lays the warped source onto the bar
grid slot by slot, synthesises drums and bass over it, applies per-slot filter
sweeps and sidechain ducking, adds risers, impacts, chops and reverb throws at
phrase boundaries, and runs the master chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..arrange import Plan, Slot
from ..audio import add_at, fit, to_stereo, xfade
from ..dsp import dynamics as DY
from ..dsp import filters as FL
from ..dsp import reverb as RV
from . import bass as BA
from . import drums as DR


@dataclass
class Stems:
    """The source, pre-warped to the target grid and split into parts."""

    harmonic: np.ndarray      # vocals + chords (or demucs vocals + other)
    percussive: np.ndarray    # original drums, kept only as breakdown texture
    source_name: str = "hpss"

    @property
    def length(self) -> int:
        return len(self.harmonic)


def _loop_to(src: np.ndarray, start: int, want: int, period: int, sr: int) -> np.ndarray:
    """Take ``want`` samples from ``src`` at ``start``, looping a ``period``-long span.

    Every repeat begins at an exact multiple of ``period``, which is a whole
    number of bars, so the loop cannot drift away from the grid however long it
    runs. Seams are hidden by overlapping a short crossfade into the material
    that follows the loop point -- audio the source already has, read past the
    end of the span -- rather than by shortening the loop.

    That distinction is the whole bug this replaced. Joining each repeat with
    ``xfade`` returned ``len(a) + len(b) - fade`` samples, so every repetition
    was 24 ms short. Eight bars at 128 BPM is a 15-second period, and a drop
    three repeats long finished 72 ms -- most of a 16th note -- ahead of where
    the grid said it was, with the error growing all the way through.
    """
    if want <= 0:
        return np.zeros((0, src.shape[1]) if src.ndim == 2 else (0,), dtype=np.float32)
    period = max(period, int(0.25 * sr))
    fade = int(min(0.024 * sr, period // 4))
    shape = (want, src.shape[1]) if src.ndim == 2 else (want,)
    out = np.zeros(shape, dtype=np.float32)

    def take(a: int, n: int) -> np.ndarray:
        return fit(src[max(0, a): max(0, a) + n], n)

    t = np.linspace(0.0, 1.0, fade, dtype=np.float32) if fade > 1 else None
    rise = np.sin(t * np.pi / 2) if t is not None else None
    fall = np.cos(t * np.pi / 2) if t is not None else None
    if rise is not None and src.ndim == 2:
        rise, fall = rise[:, None], fall[:, None]

    pos = 0
    while pos < want:
        n = min(period + fade, want - pos)
        seg = np.array(take(start, n), dtype=np.float32, copy=True)
        if rise is not None and pos > 0:
            seg[:fade] *= rise
        if rise is not None and pos + period < want and n >= fade:
            seg[period:period + fade] *= fall[: max(0, n - period)]
        out[pos:pos + n] += seg
        pos += period
    return out


def _sweep_curve(n: int, spec: tuple[float, float] | None) -> np.ndarray | None:
    return None if spec is None else FL.exp_curve(n, spec[0], spec[1])


def _chop(src: np.ndarray, sr: int, beat: float, bars: int, bar_dur: float,
          seed: int = 31) -> np.ndarray:
    """Beat-aligned stutter: repeat 1/8 then 1/16 slices over the last bars.

    Slices are taken from the start of each sub-division so the chop stays
    rhythmically locked; the division halves in the final bar, which is the
    standard "gearing up" gesture before a drop.
    """
    n = len(src)
    out = np.zeros_like(src)
    rng = np.random.default_rng(seed)
    pos = 0
    bar = 0
    while pos < n:
        div = 8 if bar < bars - 1 else 16
        slice_len = max(64, int(round(bar_dur / div * sr)))
        src_at = int(round(bar * bar_dur * sr))
        grab = fit(src[src_at: src_at + slice_len], slice_len)
        reps = int(round(bar_dur * sr / slice_len))
        for r in range(reps):
            a = pos + r * slice_len
            if a >= n:
                break
            seg = grab.copy()
            env = np.ones(len(seg), dtype=np.float32)
            env[: min(48, len(seg))] *= np.linspace(0.0, 1.0, min(48, len(seg)))
            env[-min(48, len(seg)):] *= np.linspace(1.0, 0.0, min(48, len(seg)))
            gain = 0.75 + 0.25 * rng.random()
            add_at(out, seg * (env[:, None] if seg.ndim == 2 else env), a, gain)
        pos += int(round(bar_dur * sr))
        bar += 1
    return out


def _chord_at(chords: list[dict], src_time: float) -> tuple[int, bool]:
    """Nearest-preceding chord for a source timestamp."""
    if not chords:
        return 9, True
    best = chords[0]
    for c in chords:
        if c["time"] <= src_time + 1e-6:
            best = c
        else:
            break
    return int(best["root_pc"]), best["quality"] == "min"


class Engine:
    """Stateful renderer for one remix."""

    def __init__(self, sr: int, plan: Plan, stems: Stems, chords: list[dict],
                 semitones: int = 0, swing: float = 0.08, beat_multiple: float = 1.0,
                 src_bar_dur: float = 2.0, seed: int = 0, warp=None) -> None:
        self.sr = sr
        self.plan = plan
        self.stems = stems
        self.chords = chords
        self.semitones = semitones
        self.beat_multiple = beat_multiple
        self.src_bar_dur = src_bar_dur
        self.warp = warp
        self.kit = DR.Kit(sr=sr, swing=swing)
        self.rng = np.random.default_rng(seed)
        self.bar_dur = plan.bar_dur
        self.beat = plan.bar_dur / 4.0
        self.n = int(round(plan.total_bars * plan.bar_dur * sr))
        self.ir = RV.synth_ir(sr, seconds=1.6, decay=4.0)
        #: Per-slot sample spans of rendered source audio, filled by
        #: ``render_source``. Two slots live at once is a bug; the alignment
        #: gate reads this to prove it never happens.
        self.source_spans: list[tuple[int, int]] = []
        #: The individual buses of the last ``render``, kept so the alignment
        #: gate can measure the source on its own instead of through a mix the
        #: kit dominates.
        self.layers: dict[str, np.ndarray] = {}

    # -- helpers ---------------------------------------------------------
    def _bar_sample(self, bar: float) -> int:
        return int(round(bar * self.bar_dur * self.sr))

    def _slot_chord(self, slot: Slot, bar_in_slot: int) -> tuple[int, bool]:
        """Chord for a bar of a slot, read back through the warp to source time."""
        warped_t = slot.source_start + bar_in_slot * self.bar_dur
        if self.warp is not None:
            src_t = float(self.warp.to_source(warped_t))
        else:
            # a warped bar corresponds to 1/beat_multiple source bars
            src_t = warped_t / self.bar_dur / self.beat_multiple * self.src_bar_dur
        root, minor = _chord_at(self.chords, src_t)
        return (root + self.semitones) % 12, minor

    # -- layers ----------------------------------------------------------
    def render_source(self) -> tuple[np.ndarray, np.ndarray]:
        """Lay the warped source onto the grid, slot by slot, with per-slot FX.

        Returns ``(harmonic_bed, percussive_bed)``; the house kit replaces the
        original drums, so the percussive bed is only used as low-level texture
        where the plan asks for it.
        """
        harm = np.zeros((self.n, 2), dtype=np.float32)
        perc = np.zeros((self.n, 2), dtype=np.float32)
        # The same percussive bed at unity gain whatever the plan asks for. It
        # is never mixed in; it exists so the alignment gate can measure the
        # source's own drums -- the layer with the clearest onsets, and the one
        # a listener compares against the kick -- laid out exactly where the
        # arrangement puts the source.
        perc_ref = np.zeros((self.n, 2), dtype=np.float32)
        self.source_spans = []
        for slot in self.plan.slots:
            a = self._bar_sample(slot.start_bar)
            want = self._bar_sample(slot.end_bar) - a
            self.source_spans.append((a, a + want))
            period = int(round(max(slot.source_bars, 1) * self.bar_dur * self.sr))
            start = int(round(slot.source_start * self.sr))
            seg = _loop_to(self.stems.harmonic, start, want, period, self.sr)
            pseg = _loop_to(self.stems.percussive, start, want, period, self.sr)

            hp = _sweep_curve(want, slot.highpass)
            if hp is not None:
                seg = FL.sweep(seg, "highpass", self.sr, hp, q=0.72, order=2)
            lp = _sweep_curve(want, slot.lowpass)
            if lp is not None:
                seg = FL.sweep(seg, "lowpass", self.sr, lp, q=0.72, order=2)

            if slot.chops:
                nb = min(4, slot.bars)
                tail = want - self._bar_sample(nb) + a
                head = seg[: tail - a] if tail > a else seg
                chopped = _chop(seg[tail - a:], self.sr, self.beat, nb, self.bar_dur)
                seg = np.concatenate([head, chopped])[:want]
            if slot.reverb_throw:
                seg = self._throw(seg, slot)

            # fade slot edges so a filter-swept boundary never clicks
            edge = min(int(0.012 * self.sr), want // 8)
            if edge > 1:
                seg[:edge] *= np.linspace(0.0, 1.0, edge)[:, None]
                seg[-edge:] *= np.linspace(1.0, 0.0, edge)[:, None]

            add_at(harm, seg, a, slot.source_gain)
            add_at(perc_ref, pseg, a, 1.0)
            if slot.percussive_gain > 0:
                add_at(perc, pseg, a, slot.percussive_gain)
        self.layers["source_perc"] = perc_ref
        return harm, perc

    def _throw(self, seg: np.ndarray, slot: Slot) -> np.ndarray:
        """Reverb throw on the last bar of a section (a breakdown's exit gesture)."""
        n = len(seg)
        tail_start = max(0, n - int(round(2 * self.bar_dur * self.sr)))
        wet = RV.convolve(seg[tail_start:], self.ir)
        ramp = np.linspace(0.0, 1.0, len(wet), dtype=np.float32)[:, None] ** 1.5
        out = seg.copy()
        out[tail_start:] += wet * 0.55 * ramp
        return out

    def render_drums(self) -> tuple[np.ndarray, list[float]]:
        """Synthesise the kit across the whole arrangement.

        Returns the drum bus and the list of kick times, which drives sidechain.
        """
        out = np.zeros((self.n, 2), dtype=np.float32)
        kick_times: list[float] = []
        step_dur = self.bar_dur / 16.0

        for slot in self.plan.slots:
            pattern = DR.PATTERNS.get(slot.drum_pattern, DR.PATTERNS["drop"])
            for b in range(slot.bars):
                bar_index = slot.start_bar + b
                bar_t = bar_index * self.bar_dur
                last_of_phrase = slot.fill and (b == slot.bars - 1)
                for voice, steps in pattern.items():
                    if voice == "sub":
                        continue
                    sample = self.kit.samples.get(voice)
                    if sample is None:
                        continue
                    for step, vel in steps:
                        if last_of_phrase and voice in ("hat", "shaker") and step >= 12:
                            continue      # clear room for the fill
                        t = self.kit.step_time(bar_t, step, step_dur)
                        v = vel * (0.92 + 0.16 * self.rng.random())
                        add_at(out, to_stereo(sample), int(round(t * self.sr)), v)
                        if voice == "kick":
                            kick_times.append(t)
                if last_of_phrase:
                    self._fill(out, bar_t, step_dur)
            if slot.riser:
                self._riser(out, slot)
            if slot.impact:
                add_at(out, to_stereo(DR.impact(self.sr)),
                       self._bar_sample(slot.start_bar), 0.9)
        return out, sorted(kick_times)

    def _fill(self, out: np.ndarray, bar_t: float, step_dur: float) -> None:
        """Tom/perc fill over the last beat of a phrase."""
        freqs = (220.0, 180.0, 150.0, 120.0)
        for i, step in enumerate((12, 13, 14, 15)):
            t = bar_t + step * step_dur
            add_at(out, to_stereo(DR.tom(self.sr, freqs[i], seed=13 + i)),
                   int(round(t * self.sr)), 0.5 + 0.12 * i)

    def _riser(self, out: np.ndarray, slot: Slot) -> None:
        """Noise riser filling a build slot, ending exactly on the next downbeat."""
        a = self._bar_sample(slot.start_bar)
        seconds = slot.bars * self.bar_dur
        r = DR.riser_noise(self.sr, seconds)
        add_at(out, to_stereo(r), a, 0.62)

    def render_bass(self, kick_times: list[float]) -> np.ndarray:
        """Rolling offbeat bass following the per-bar chord roots."""
        out = np.zeros((self.n, 2), dtype=np.float32)
        for slot in self.plan.slots:
            if not slot.use_bass:
                continue
            for b in range(slot.bars):
                root, minor = self._slot_chord(slot, b)
                octave = BA.pick_bass_octave(root)
                bar_t = (slot.start_bar + b) * self.bar_dur
                # offbeat 8ths: the "rolling" house bass that dodges the kick
                for k in range(8):
                    if k % 2 == 0:
                        continue
                    t = bar_t + k * (self.bar_dur / 8.0)
                    oct_up = 1 if (k == 5) else 0     # a lift in the middle of the bar
                    f = BA.note_hz(root, octave + oct_up)
                    note = BA.bass_note(self.sr, f, self.bar_dur / 8.0 * 0.95)
                    add_at(out, to_stereo(note), int(round(t * self.sr)), 0.9)
            if slot.use_stabs:
                for b in range(slot.bars):
                    root, minor = self._slot_chord(slot, b)
                    bar_t = (slot.start_bar + b) * self.bar_dur
                    for k in (1, 3, 5, 7):
                        t = bar_t + k * (self.bar_dur / 8.0)
                        s = BA.stab(self.sr, root, minor, self.bar_dur / 8.0 * 0.8)
                        add_at(out, to_stereo(s), int(round(t * self.sr)), 0.55)
        return out

    def render(self) -> tuple[np.ndarray, dict]:
        """Full render. Returns ``(audio, metrics)``."""
        harm, perc = self.render_source()
        drums, kick_times = self.render_drums()
        bassline = self.render_bass(kick_times)

        # sidechain: one envelope per depth value used by the plan
        depths = sorted({s.sidechain for s in self.plan.slots if s.sidechain > 0})
        envs = {d: DY.sidechain_envelope(self.n, self.sr, np.asarray(kick_times), depth=d,
                                         release=0.20 * (60.0 / self.plan.target_bpm) / 0.4839)
                for d in depths}
        for slot in self.plan.slots:
            if slot.sidechain <= 0:
                continue
            a = self._bar_sample(slot.start_bar)
            b = self._bar_sample(slot.end_bar)
            env = envs[slot.sidechain][a:b]
            harm[a:b] = DY.apply_sidechain(harm[a:b], env)
            perc[a:b] = DY.apply_sidechain(perc[a:b], env)

        # the bass always ducks under the kick, hard
        bass_env = DY.sidechain_envelope(self.n, self.sr, np.asarray(kick_times),
                                         depth=0.85, release=0.16)
        bassline = DY.apply_sidechain(bassline, bass_env)
        # carve 40-90 Hz out of the source so the kick and bass own the sub
        harm = FL.apply(harm, "highpass", self.sr, 105.0, q=0.707, order=2)

        source_bus = harm * 1.35 + perc
        mix = source_bus + drums * 0.72 + bassline * 0.55
        self.layers.update({"source": source_bus, "kit": drums * 0.72,
                            "bass": bassline * 0.55})
        out = DY.master(mix, self.sr, peak_db=-1.0)

        metrics = {
            "kick_count": len(kick_times),
            "peak_db": float(20 * np.log10(max(float(np.max(np.abs(out))), 1e-6))),
            "rms_db": float(20 * np.log10(max(float(np.sqrt(np.mean(out ** 2))), 1e-6))),
        }
        return out, metrics
