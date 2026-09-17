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
from ..audio import add_at, fit, to_stereo
from ..dsp import dynamics as DY
from ..dsp import filters as FL
from ..dsp import reverb as RV
from . import bass as BA
from . import drums as DR


#: Bus gains into the master. They are named because the balance report and the
#: adaptive vocal-band solve both have to measure the buses as the master sees
#: them, not as they leave their renderers.
HARM_GAIN = 1.35
PERC_GAIN = 1.0
DRUM_GAIN = 0.72
BASS_GAIN = 0.55

#: Source-layer crossfade at a slot boundary, in beats. Milliseconds are the
#: wrong unit here: a cut between two sections is a musical event, and the ear
#: reads a fade that lasts a beat as part of the arrangement rather than as a
#: repair.
XFADE_BEATS = 1.0

#: How much of a loop's own tail is blended with the bar before its start, so a
#: repeat continues the line instead of splicing onto it.
LOOP_SEAM_BEATS = 0.5

#: Every synthesised one-shot is cut off while it is still ringing -- an open
#: hat at 13% of its peak, a clap at 11%. That truncation is a step, and a step
#: is a click. Six milliseconds of fade is far shorter than any decay we care
#: about and removes the whole family of them.
LAND_MS = 6.0

#: A drop is cut in, not faded in: the source starts at level on beat 1 behind
#: nothing more than a de-click ramp. Measuring twenty drops across six
#: commercial house remixes says the same thing -- a drop is a +4.9 dB step,
#: eighteen of the twenty have no drum gap in front of them at all, and the
#: monotonic filter ramps that do appear last a median 0.13 beats. So: a step,
#: no vacuum, no ride into it.
SLAM_MS = 4.0

#: Beats over which a section sweeps and thins on its way into a breakdown.
#: Leaving a drop is the one boundary the references do ride: the centroid
#: climbs about 412 Hz across it while the kit's low end is taken away. They
#: are gentle about the level, though -- a drop exits only 2.9 dB down, with
#: its last bar of kick played complete -- so the thinning here is spectral
#: first and a small trim second.
EXIT_SWEEP_BEATS = 3.0

#: Where the source's exit low-pass lands. A breakdown opens its own filter
#: from here, so the two meet at the same cutoff and the sweep is continuous
#: across the boundary instead of stepping at it.
EXIT_LOWPASS_HZ = 4200.0

#: Where the kit's exit high-pass climbs to, and how far the kit is pulled
#: back, over those same beats.
EXIT_KIT_HIGHPASS_HZ = 420.0
EXIT_KIT_DUCK_DB = -3.0

#: How long a filter or gain move is given to resolve onto a downbeat. Short
#: enough to read as "on the beat", long enough not to be a step.
RESOLVE_MS = 8.0

#: The riser covers the build's last two bars, not the whole build. The
#: references keep their risers short and strong -- 1.9 to 3.4 dB per beat of
#: 4-16 kHz over the last two bars -- rather than riding one for eight.
RISER_BARS = 2

#: The riser's decay past the drop downbeat. Ending a full-scale noise sweep on
#: a sample boundary is the loudest click in the whole render.
RISER_TAIL_BEATS = 0.5

#: Reverb thrown across a drop-to-breakdown boundary, so the kit stops but the
#: pad and vocal carry over rather than being guillotined.
THROW_SPILL_GAIN = 0.5
THROW_SPILL_SECONDS = 1.6


@dataclass
class Stems:
    """The source, pre-warped to the target grid and split into parts."""

    harmonic: np.ndarray      # vocals + chords (or demucs vocals + other)
    percussive: np.ndarray    # original drums, kept only as breakdown texture
    source_name: str = "hpss"

    @property
    def length(self) -> int:
        return len(self.harmonic)


def _gather(src: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``src[idx]`` with out-of-range positions reading silence, always a copy."""
    ok = (idx >= 0) & (idx < len(src))
    out = src[np.where(ok, idx, 0)]
    out[~ok] = 0.0
    return np.asarray(out, dtype=np.float32)


def _equal_power(t: np.ndarray, stereo: bool) -> tuple[np.ndarray, np.ndarray]:
    """Complementary cos/sin gains for a crossfade parameter ``t`` in [0, 1]."""
    out = np.cos(t * np.pi / 2).astype(np.float32)
    into = np.sin(t * np.pi / 2).astype(np.float32)
    return (out[:, None], into[:, None]) if stereo else (out, into)


def _loop_to(src: np.ndarray, start: int, want: int, period: int, sr: int,
             pre: int = 0, seam: int = 0) -> np.ndarray:
    """Take ``want`` samples from ``src`` at ``start``, looping a ``period``-long span.

    Positions are computed modulo ``period``, so every repeat sits on exactly the
    same phase of the bar grid. The previous implementation crossfaded each
    repeat onto the last with ``xfade``, which returns a buffer shorter than the
    two inputs by the fade length: the loop lost 24 ms per repeat and walked
    ahead of the grid by most of a 16th note over a long drop.

    ``pre`` asks for extra samples *before* ``start``, at the loop phase they
    would have had, so a caller can crossfade a slot boundary with material that
    belongs to the same loop instead of fading to silence and back.

    ``seam`` smooths the wrap the way a sampler does: the loop's own tail fades
    into whatever precedes ``start``, so the sample after the wrap continues the
    line exactly. When the span starts too close to the head of the source to
    have a "before", the blend mirrors -- the loop's head fades out of the
    material that follows the loop instead.
    """
    total = max(0, pre) + max(0, want)
    if total <= 0:
        return np.zeros((0, src.shape[1]) if src.ndim == 2 else (0,), dtype=np.float32)
    period = max(int(period), 64)
    pre = max(0, min(pre, period))

    phase = np.mod(np.arange(total, dtype=np.int64) - pre, period)
    out = _gather(src, start + phase)

    seam = int(min(seam, period // 4))
    if seam > 0:
        if start >= seam:
            mask = phase >= period - seam
            t = (phase[mask] - (period - seam)).astype(np.float32) / seam
            alt = _gather(src, start + phase[mask] - period)
        else:
            mask = phase < seam
            t = 1.0 - phase[mask].astype(np.float32) / seam
            alt = _gather(src, start + phase[mask] + period)
        if mask.any():
            keep, blend = _equal_power(t, out.ndim == 2)
            out[mask] = out[mask] * keep + alt * blend
    return out


def _width(x: np.ndarray) -> float:
    """Side-over-mid RMS: 0 is mono, the references sit at 0.16 in a drop."""
    if x.ndim != 2 or x.shape[1] < 2:
        return 0.0
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5
    return float(DY.rms(side) / max(DY.rms(mid), 1e-9))


def _land(x: np.ndarray, sr: int, ms: float = LAND_MS) -> np.ndarray:
    """Fade the last few milliseconds of a one-shot to zero."""
    n = min(len(x), max(2, int(ms * 0.001 * sr)))
    if n < 2:
        return x
    out = np.array(x, dtype=np.float32, copy=True)
    ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)
    out[-n:] *= ramp[:, None] if out.ndim == 2 else ramp
    return out


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
                 src_bar_dur: float = 2.0, seed: int = 0) -> None:
        self.sr = sr
        self.plan = plan
        self.stems = stems
        self.chords = chords
        self.semitones = semitones
        self.beat_multiple = beat_multiple
        self.src_bar_dur = src_bar_dur
        self.kit = DR.Kit(sr=sr, swing=swing)
        self.rng = np.random.default_rng(seed)
        self.bar_dur = plan.bar_dur
        self.beat = plan.bar_dur / 4.0
        self.n = int(round(plan.total_bars * plan.bar_dur * sr))
        self.ir = RV.synth_ir(sr, seconds=1.6, decay=4.0)
        self.buses: dict[str, np.ndarray] = {}
        # Land every voice: the kit renders its one-shots to a fixed length and
        # lets them stop wherever the envelope has got to, which puts a step at
        # the end of every open hat, clap and tom.
        self.voices = {k: _land(v, sr) for k, v in self.kit.samples.items()}
        self.impact = _land(DR.impact(sr), sr)

    # -- helpers ---------------------------------------------------------
    def _bar_sample(self, bar: float) -> int:
        return int(round(bar * self.bar_dur * self.sr))

    def _slot_chord(self, slot: Slot, bar_in_slot: int) -> tuple[int, bool]:
        """Chord for a bar of a slot, read back through the warp to source time."""
        warped_t = slot.source_start + bar_in_slot * self.bar_dur
        # a warped bar corresponds to 1/beat_multiple source bars
        src_t = warped_t / self.bar_dur / self.beat_multiple * self.src_bar_dur
        root, minor = _chord_at(self.chords, src_t)
        return (root + self.semitones) % 12, minor

    # -- layers ----------------------------------------------------------
    def _slot(self, index: int) -> Slot | None:
        slots = self.plan.slots
        return slots[index] if 0 <= index < len(slots) else None

    def _kind(self, index: int) -> str | None:
        slot = self._slot(index)
        return slot.kind if slot is not None else None

    def _has_kick(self, slot: Slot | None) -> bool:
        """Whether a slot's pattern plays a kick at all.

        Read from the pattern rather than from the slot kind, so a change to the
        kit's patterns changes what the transitions do with them.
        """
        if slot is None:
            return False
        pattern = DR.PATTERNS.get(slot.drum_pattern, {})
        return bool(pattern.get("kick"))

    def _tail_sweep(self, buf: np.ndarray, start: int, end: int, kind: str,
                    f0: float, f1: float, resolve: bool = False) -> None:
        """Sweep a filter from ``f0`` to ``f1`` over ``[start, end)``, in place.

        The biquad is given a bar of settling time before ``start`` with the
        cutoff held at ``f0`` -- at the open value that is very nearly a no-op,
        but it means the filter arrives at the sweep with the state the material
        actually implies rather than the zeros a cold ``lfilter`` starts from.
        With ``resolve`` the sweep continues a few milliseconds past ``end`` and
        runs back to ``f0``, so the cutoff is open again on the downbeat without
        the filter state jumping there.
        """
        settle = min(self._bar_sample(1), start)
        back = int(RESOLVE_MS * 0.001 * self.sr) if resolve else 0
        a, b = start - settle, min(len(buf), end + back)
        if b - a < 128 or end <= start:
            return
        seg = buf[a:b]
        curve = [np.full(settle, f0), FL.exp_curve(end - start, f0, f1)]
        if b > end:
            curve.append(FL.exp_curve(b - end, f1, f0))
        wet = FL.sweep(seg, kind, self.sr, np.concatenate(curve), q=0.72, order=2)
        blend = min(int(0.02 * self.sr), max(settle // 2, 0))
        if blend > 1:
            keep, come = _equal_power(np.linspace(0.0, 1.0, blend, dtype=np.float32),
                                      wet.ndim == 2)
            wet[:blend] = seg[:blend] * keep + wet[:blend] * come
        buf[a:b] = wet

    def _duck_into(self, buf: np.ndarray, at: int, beats: float, depth_db: float,
                   recover: bool = True) -> None:
        """Ramp a bus down over the beats before ``at``, in place.

        With ``recover`` the last few milliseconds run back to unity, so the
        section that starts on the downbeat starts at full level and the ringing
        tails of the bar before do not step.
        """
        k = int(round(beats * self.beat * self.sr))
        a = max(0, at - k)
        if at - a < 64:
            return
        depth = 10.0 ** (depth_db / 20.0)
        n = at - a
        curve = 1.0 + (depth - 1.0) * (np.linspace(0.0, 1.0, n, dtype=np.float32) ** 1.4)
        back = min(int(RESOLVE_MS * 0.001 * self.sr), n // 2)
        if recover and back > 1:
            curve[-back:] = np.linspace(float(curve[-back]), 1.0, back, dtype=np.float32)
        buf[a:at] *= curve[:, None] if buf.ndim == 2 else curve

    def _beat_fade(self, slot: Slot) -> int:
        """A beat, clamped to a quarter of the slot it has to fit inside."""
        want = int(round(XFADE_BEATS * self.beat * self.sr))
        return max(0, min(want, int(slot.bars * self.bar_dur * self.sr / 4.0)))

    def _xfade_len(self, index: int) -> int:
        """Fade-in length, in samples, at the boundary *entering* slot ``index``.

        One beat for an ordinary boundary: the incoming slot is rendered that
        much early, at the loop phase it will have, and arrives at full level
        exactly on beat 1.

        A drop is the exception. There the incoming material is cut in behind
        nothing but a de-click ramp, because a drop that fades in over a beat is
        not a drop. The outgoing build still gets its full beat of fade, so the
        two do not sum to unity across the boundary -- that shortfall is the
        vacuum in front of the drop, and it is deliberate.
        """
        slots = self.plan.slots
        if index <= 0 or index >= len(slots):
            return 0
        if slots[index].kind == "drop":
            return int(SLAM_MS * 0.001 * self.sr)
        return min(self._beat_fade(slots[index]), self._beat_fade(slots[index - 1]))

    def _fadeout_len(self, index: int) -> int:
        """Fade-out length at the end of slot ``index``.

        Always a beat: it is the outgoing side of the boundary and it only ever
        eats into its own slot, so it does not care what comes next.
        """
        slot = self._slot(index)
        return 0 if slot is None else self._beat_fade(slot)

    def _sweep_over(self, spec: tuple[float, float] | None, want: int,
                    pre: int) -> np.ndarray | None:
        """A slot's cutoff curve, held flat across the crossfade lead-in.

        Holding the start value through the lead-in means the biquad has a whole
        beat of the right material to settle on before anything is audible, so
        the slot opens with a warmed-up filter rather than with the state a
        cold ``lfilter`` invents.
        """
        if spec is None:
            return None
        curve = FL.exp_curve(want, spec[0], spec[1])
        return curve if pre <= 0 else np.concatenate([np.full(pre, spec[0]), curve])

    def render_source(self) -> tuple[np.ndarray, np.ndarray]:
        """Lay the warped source onto the grid, slot by slot, with per-slot FX.

        Returns ``(harmonic_bed, percussive_bed)``; the house kit replaces the
        original drums, so the percussive bed is only used as low-level texture
        where the plan asks for it.

        Every slot is rendered a beat early and handed to ``add_at`` a beat
        early: the lead-in carries the same loop phase the slot will have, and
        the two sides of a boundary use complementary equal-power gains, so the
        source layer crossfades across a beat instead of ducking to silence and
        back through a pair of 12 ms edge fades.
        """
        harm = np.zeros((self.n, 2), dtype=np.float32)
        perc = np.zeros((self.n, 2), dtype=np.float32)
        seam = int(round(LOOP_SEAM_BEATS * self.beat * self.sr))
        last = len(self.plan.slots) - 1
        for i, slot in enumerate(self.plan.slots):
            a = self._bar_sample(slot.start_bar)
            want = self._bar_sample(slot.end_bar) - a
            pre = self._xfade_len(i)
            post = self._fadeout_len(i)
            head = 0 if i > 0 else self._beat_fade(slot)

            period = int(round(max(slot.source_bars, 1) * self.beat_multiple
                               * self.bar_dur * self.sr))
            start = int(round(slot.source_start * self.sr))
            seg = _loop_to(self.stems.harmonic, start, want, period, self.sr,
                           pre=pre, seam=seam)
            pseg = _loop_to(self.stems.percussive, start, want, period, self.sr,
                            pre=pre, seam=seam)

            hp = self._sweep_over(slot.highpass, want, pre)
            if hp is not None:
                seg = FL.sweep(seg, "highpass", self.sr, hp, q=0.72, order=2)
            lp = self._sweep_over(slot.lowpass, want, pre)
            if lp is not None:
                seg = FL.sweep(seg, "lowpass", self.sr, lp, q=0.72, order=2)

            if slot.chops:
                nb = min(4, slot.bars)
                cut = want - self._bar_sample(nb)
                if cut > 0:
                    chopped = _chop(seg[pre + cut:], self.sr, self.beat, nb, self.bar_dur)
                    seg = np.concatenate([seg[: pre + cut], chopped])[: pre + want]
            if slot.reverb_throw:
                seg = self._throw(seg, slot)
            if self._kind(i + 1) == "breakdown":
                self._exit_to_breakdown(seg, self.plan.slots[i + 1], want)
                self._spill_throw(harm, seg, a + want, slot.source_gain)

            for n_fade, at_head in ((pre or head, True), (post, False)):
                if n_fade <= 1:
                    continue
                t = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
                out_g, in_g = _equal_power(t, True)
                if at_head:
                    seg[:n_fade] *= in_g
                else:
                    seg[len(seg) - n_fade:] *= out_g

            add_at(harm, seg, a - pre, slot.source_gain)
            if slot.percussive_gain > 0:
                add_at(perc, pseg, a - pre, slot.percussive_gain)
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

    def _exit_to_breakdown(self, seg: np.ndarray, nxt: Slot, want: int) -> None:
        """Sweep the source down into the cutoff the breakdown opens from.

        A breakdown used to arrive as a step: the outgoing section ran to the
        boundary wide open and the breakdown's own filter started at 4.2 kHz on
        the next sample. Sweeping the last three beats down to exactly where the
        breakdown starts makes the two halves one continuous move.
        """
        if nxt.lowpass is None and nxt.highpass is not None:
            return
        land = nxt.lowpass[0] if nxt.lowpass else EXIT_LOWPASS_HZ
        k = min(int(round(EXIT_SWEEP_BEATS * self.beat * self.sr)), want // 2)
        if k < 128:
            return
        self._tail_sweep(seg, len(seg) - k, len(seg), "lowpass", self.sr * 0.49, land)

    def _spill_throw(self, harm: np.ndarray, seg: np.ndarray, boundary: int,
                     gain: float) -> None:
        """Throw the last bar into reverb that carries past the boundary.

        The wet tail is added to the bed directly rather than to the slot, so it
        is not caught by the slot's own fade-out: the kit stops on the downbeat
        and the pad and vocal ring on into the breakdown.
        """
        bar = self._bar_sample(1)
        tail = np.asarray(seg[max(0, len(seg) - bar):], dtype=np.float32)
        if len(tail) < 256:
            return
        pad = np.concatenate([tail, np.zeros((int(THROW_SPILL_SECONDS * self.sr), 2),
                                             dtype=np.float32)])
        wet = RV.convolve(pad, self.ir)
        ramp = np.linspace(0.0, 1.0, len(tail), dtype=np.float32) ** 1.5
        wet[:len(tail)] *= ramp[:, None]
        add_at(harm, wet, boundary - len(tail), gain * THROW_SPILL_GAIN)

    def shape_kit_transitions(self, drums: np.ndarray) -> None:
        """Thin and sweep the kit across section boundaries, in place.

        Two gestures, both measured off a commercial house remix of one of the
        test sources rather than chosen by ear:

        - into a breakdown the kit does not stop dead. Over three beats it loses
          about 7 dB and a rising high-pass takes its low end away, which is why
          the reference's spectral centroid climbs into the boundary while its
          drum level falls. The filter resolves back open on the downbeat.
        Nothing is done in front of a drop on purpose: eighteen of twenty
        reference drops have no drum gap at all, so the kit plays its last bar
        complete and the drop arrives as a step on top of it.
        """
        for i, slot in enumerate(self.plan.slots[:-1]):
            nxt = self.plan.slots[i + 1]
            b = self._bar_sample(nxt.start_bar)
            if nxt.kind == "breakdown" and self._has_kick(slot):
                k = int(round(EXIT_SWEEP_BEATS * self.beat * self.sr))
                self._tail_sweep(drums, b - k, b, "highpass", 20.0,
                                 EXIT_KIT_HIGHPASS_HZ, resolve=True)
                self._duck_into(drums, b, EXIT_SWEEP_BEATS, EXIT_KIT_DUCK_DB)


    def render_drums(self) -> tuple[np.ndarray, list[float]]:
        """Synthesise the kit across the whole arrangement.

        Returns the drum bus and the list of kick times, which drives sidechain.
        """
        out = np.zeros((self.n, 2), dtype=np.float32)
        kick_times: list[float] = []
        step_dur = self.bar_dur / 16.0

        for i, slot in enumerate(self.plan.slots):
            pattern = DR.PATTERNS.get(slot.drum_pattern, DR.PATTERNS["drop"])
            entering = slot.kind == "build" and not self._has_kick(self._slot(i - 1))
            for b in range(slot.bars):
                bar_index = slot.start_bar + b
                bar_t = bar_index * self.bar_dur
                last_of_phrase = slot.fill and (b == slot.bars - 1)
                for voice, steps in pattern.items():
                    if voice == "sub":
                        continue
                    sample = self.voices.get(voice)
                    if sample is None:
                        continue
                    gate = self._entry_gain(voice, b, slot.bars) if entering else 1.0
                    if gate <= 0.0:
                        continue
                    for step, vel in steps:
                        if last_of_phrase and voice in ("hat", "shaker") and step >= 12:
                            continue      # clear room for the fill
                        t = self.kit.step_time(bar_t, step, step_dur)
                        v = vel * gate * (0.92 + 0.16 * self.rng.random())
                        add_at(out, to_stereo(sample), int(round(t * self.sr)), v)
                        if voice == "kick":
                            kick_times.append(t)
                if last_of_phrase and not (entering and b < slot.bars - 2):
                    self._fill(out, bar_t, step_dur)
            if slot.riser:
                self._riser(out, slot)
            if slot.impact:
                add_at(out, to_stereo(self.impact),
                       self._bar_sample(slot.start_bar), 0.9)
        return out, sorted(kick_times)

    def _entry_gain(self, voice: str, bar: int, bars: int) -> float:
        """How much of a voice plays, ``bar`` bars into a build that follows a
        section with no kick.

        Coming out of a breakdown the whole kit used to appear at once on the
        build's first downbeat, which is the loudest edit in the arrangement.
        The parts come back in the order a DJ brings them back: hats from the
        top, rising; the clap at the half way point; the kick for the last eight
        beats, so it arrives as the build's own last gesture rather than as a
        surprise.
        """
        if voice == "kick":
            return 1.0 if bar >= max(0, bars - 2) else 0.0
        if voice == "clap":
            return 0.0 if bar < 1 else float(0.7 + 0.3 * (bar / max(bars - 1, 1)))
        if voice in ("hat", "ohat", "shaker"):
            return float(0.8 + 0.2 * (bar / max(bars - 1, 1)))
        return 1.0

    def _fill(self, out: np.ndarray, bar_t: float, step_dur: float) -> None:
        """Tom/perc fill over the last beat of a phrase."""
        freqs = (220.0, 180.0, 150.0, 120.0)
        for i, step in enumerate((12, 13, 14, 15)):
            t = bar_t + step * step_dur
            add_at(out, to_stereo(_land(DR.tom(self.sr, freqs[i], seed=13 + i), self.sr)),
                   int(round(t * self.sr)), 0.5 + 0.12 * i)

    def _riser(self, out: np.ndarray, slot: Slot) -> None:
        """Noise riser over the build's last two bars, peaking on the next downbeat.

        Two changes from the eight-bar ride this used to be. It is short: the
        references run their risers over the last two bars and only on about one
        drop in three, and long monotonic ramps into a drop do not appear in the
        corpus at all. And it lands: stopping a full-scale noise sweep on a
        sample boundary is a full-amplitude step, and was the single loudest
        click in the render. It now starts from silence, peaks on the downbeat
        and washes out over half a beat past it.
        """
        bars = min(RISER_BARS, slot.bars)
        a = self._bar_sample(slot.end_bar - bars)
        tail = int(round(RISER_TAIL_BEATS * self.beat * self.sr))
        r = to_stereo(DR.riser_noise(self.sr, bars * self.bar_dur))
        lead = min(int(0.01 * self.sr), len(r) // 8)
        if lead > 1:
            r = np.array(r, dtype=np.float32, copy=True)
            r[:lead] *= np.linspace(0.0, 1.0, lead, dtype=np.float32)[:, None]
        if tail > 8:
            wash = r[len(r) - tail:] * np.linspace(1.0, 0.0, tail,
                                                   dtype=np.float32)[:, None] ** 2.0
            r = np.concatenate([r, wash])
        add_at(out, r, a, 0.74)

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
        self.shape_kit_transitions(drums)
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

        self.buses = {"harmonic": harm * HARM_GAIN, "percussive": perc * PERC_GAIN,
                      "drums": drums * DRUM_GAIN, "bass": bassline * BASS_GAIN}
        mix = sum(self.buses.values())
        out = DY.master(mix, self.sr, peak_db=-1.0)

        metrics = {
            "kick_count": len(kick_times),
            "peak_db": float(20 * np.log10(max(float(np.max(np.abs(out))), 1e-6))),
            "rms_db": float(20 * np.log10(max(float(np.sqrt(np.mean(out ** 2))), 1e-6))),
            "balance": self.balance_report(),
        }
        return out, metrics

    # -- measurement -----------------------------------------------------
    def balance_report(self) -> list[dict]:
        """Per-slot source-versus-kit levels, as they arrive at the master bus.

        ``vocal_band_db`` is the gap between the source bed and the house kit
        inside 300 Hz - 4 kHz: positive means the source is on top, which is
        where a vocal has to sit to survive a drop. The full-band column is
        there to show what the kick is doing to the same comparison.
        """
        if not self.buses:
            return []
        harm = self.buses["harmonic"]
        kit = self.buses["drums"]
        bed = kit + self.buses["bass"] + self.buses["percussive"]
        mix = harm + bed
        full = (0.0, self.sr * 0.5)
        report: list[dict] = []
        for slot in self.plan.slots:
            a, b = self._bar_sample(slot.start_bar), self._bar_sample(slot.end_bar)
            row = {"slot": slot.index, "kind": slot.kind,
                   "start": round(slot.start_bar * self.bar_dur, 3)}
            for name, buf in (("source", harm), ("kit", kit), ("bed", bed)):
                row[f"{name}_vocal_band_db"] = round(
                    DY.band_rms_db(buf[a:b], self.sr, DY.VOCAL_BAND), 2)
                row[f"{name}_presence_db"] = round(
                    DY.band_rms_db(buf[a:b], self.sr, DY.PRESENCE_BAND), 2)
                row[f"{name}_full_band_db"] = round(
                    DY.band_rms_db(buf[a:b], self.sr, full), 2)
            row["vocal_band_margin_db"] = round(
                row["source_vocal_band_db"] - row["kit_vocal_band_db"], 2)
            row["presence_margin_db"] = round(
                row["source_presence_db"] - row["kit_presence_db"], 2)
            row["full_band_margin_db"] = round(
                row["source_full_band_db"] - row["kit_full_band_db"], 2)
            # The reference spec's axis: the source against everything else in
            # the mix, not against the kit alone.
            row["source_minus_bed_vocal_db"] = round(
                row["source_vocal_band_db"] - row["bed_vocal_band_db"], 2)
            row["source_minus_bed_full_db"] = round(
                row["source_full_band_db"] - row["bed_full_band_db"], 2)
            row["rms_db"] = round(DY.rms_db(mix[a:b]), 2)
            row["width"] = round(_width(mix[a:b]), 3)
            report.append(row)
        return report
