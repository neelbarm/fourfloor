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

#: How long a filter or gain move is given to resolve onto a downbeat. A 32nd
#: note at 124 BPM is 60 ms, so 25 still reads as "on the one" -- and it is the
#: window the swept buffer is blended back into the dry one over, which is what
#: stops the filter's state being dropped on the downbeat.
RESOLVE_MS = 25.0

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

#: How deep the kick ducks the source's low band, and how long the envelope
#: takes to come back to within 1 dB of unity. The reference corpus wants -12
#: to -18 dB recovering in 80 to 110 ms; below -20 dB it stops reading as
#: pumping and starts reading as gating. The plan's per-slot sidechain values
#: are relative, so they are rescaled to land the deepest one here rather than
#: being used as absolute depths.
SIDECHAIN_LOW_DIP_DB = -14.0
SIDECHAIN_RECOVERY = 0.095
SIDECHAIN_SHAPE = 2.0
BASS_DIP_DB = -17.0

#: Where the source should sit relative to the rest of the mix in a section
#: that has a kick. Both numbers come from the commercial house remix of one of
#: the test sources, split the way fourfloor splits: source = vocals + other,
#: bed = drums + bass. Its drop measures +1.7 dB in 300 Hz - 4 kHz and -2.9 dB
#: in 2 - 5 kHz. fourfloor's fixture measured +6.9 and -13.4: the source was
#: simultaneously too loud in the low mids and absent where a voice is
#: understood.
VOCAL_BAND_TARGET_DB = 1.5
PRESENCE_TARGET_DB = -3.0

#: Bounds on the adaptive correction. The solve is a nudge toward a measured
#: target, not a normaliser: a source with nothing at 3 kHz should not have its
#: noise floor lifted looking for something that is not there, and a quiet
#: source should not be shouted at. Trimming down is allowed further than
#: lifting up, because pulling a pad out of the kick's way is safe and pushing
#: a thin source is not.
PRESENCE_LIFT_MAX_DB = 6.0
PRESENCE_CUT_MAX_DB = 6.0
#: Where the bed has to sit under the source across the whole spectrum in a
#: drop. The reference remix reads -12.0 dB: level with the source where a
#: voice is understood, and burying it everywhere else, which is what a kick
#: and a bass line are for. Unlike the band targets this one is met by staging
#: the bed up rather than by pulling the source down, and it is measured rather
#: than assumed -- the drums and the bass are rebuilt independently of this
#: file, so their absolute level is not something to hard-code against.
FULL_BAND_TARGET_DB = -12.0
DRUM_STAGE_RANGE_DB = (-6.0, 14.0)

#: The bass and the kick share 40-120 Hz as near-equals in the references: the
#: bass sits 0.8 dB under the drums there, not tucked beneath them. Solved
#: rather than set, for the same reason -- the kit and the bass line are built
#: in their own modules, and a change to either one's weight would otherwise
#: silently move the whole low end.
BASS_VS_KICK_DB = BA.SUB_LEVEL_VS_KICK_DB
BASS_BAND = (40.0, 120.0)
BASS_STAGE_RANGE_DB = (-8.0, 12.0)

BALANCE_TRIM_DOWN_DB = 6.0
BALANCE_TRIM_UP_DB = 3.0

#: A bell centred at 3 kHz lands inside 300 Hz - 4 kHz too, so lifting presence
#: puts some of the body back. Taking it off again with another broadband trim
#: just undoes the presence lift -- the two corrections chase each other around
#: the same band. It comes off in the low mids instead, where the source is
#: crowding the kick and the bass and where a 3 kHz bell has nothing to lose.
BODY_BAND = (300.0, 1600.0)
BODY_Q = 0.7
BODY_CUT_MAX_DB = 5.0

#: A bell at 693 Hz does not cover the whole of 300 Hz - 4 kHz, so removing one
#: dB from the band's RMS costs rather more than one dB at the bell's peak.
BODY_CUT_LEVERAGE = 1.6

#: The presence bell's Q. Wide: it has a two-and-a-half-octave band to move,
#: not a resonance to fix.
PRESENCE_Q = 0.6

#: With demucs stems the vocal and everything else tonal arrive separately, and
#: the reference remix treats them very differently: in its drop the "other"
#: stem sits 11 dB under the vocal, against 5 dB in its builds and 1 to 9 dB in
#: the original song. Pushing the pads and guitars down in the drop is how a
#: real remix makes room for a voice without turning the voice up.
OTHER_DUCK_DB = {"drop": -6.0, "build": -2.0}


@dataclass
class Stems:
    """The source, pre-warped to the target grid and split into parts."""

    harmonic: np.ndarray      # vocals + chords (or demucs vocals + other)
    percussive: np.ndarray    # original drums, kept only as breakdown texture
    source_name: str = "hpss"
    vocal: np.ndarray | None = None   # demucs only: the vocal alone
    other: np.ndarray | None = None   # demucs only: everything tonal that is not the vocal

    @property
    def length(self) -> int:
        return len(self.harmonic)

    @property
    def split(self) -> bool:
        """True when the vocal can be treated separately from the rest."""
        return self.vocal is not None and self.other is not None


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
                 semitones: int = 0, swing: float = 0.0, beat_multiple: float = 1.0,
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
        self.balance_moves: list[dict] = []
        self.vocal_bed: np.ndarray | None = None
        self.drum_stage_db = 0.0
        self.bass_stage_db = 0.0
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

    @staticmethod
    def _gestures(slot: Slot | None, where: str, kind: str) -> list[dict]:
        """The slot's transition descriptors of one kind, in time order.

        The planner writes what each boundary is supposed to *do*; the engine
        decides how to make that sound. Kinds the engine has no gesture for are
        simply not matched here, which is how an unknown kind becomes a no-op.
        """
        if slot is None:
            return []
        descs = getattr(slot, f"transition_{where}", None) or []
        return [d for d in descs if d.get("kind") == kind]

    def _beat_sample(self, beat: float) -> int:
        return int(round(beat * self.beat * self.sr))

    def _has_kick(self, slot: Slot | None) -> bool:
        """Whether a slot's pattern plays a kick at all.

        Read from the pattern rather than from the slot kind, so a change to the
        kit's patterns changes what the transitions do with them.
        """
        if slot is None:
            return False
        return bool(self._pattern(slot).get("kick"))

    @staticmethod
    def _pattern(slot: Slot) -> dict:
        """The pattern a slot will actually be played with.

        The same fallback ``render_drums`` uses, so anything that asks whether a
        slot has a kick gets the answer the render will give it rather than an
        answer about a pattern name that may not exist.
        """
        return DR.PATTERNS.get(slot.drum_pattern, DR.PATTERNS["drop"])

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
        if back > 1:
            # The swept buffer stops here and the dry signal carries on, which
            # throws away the filter's memory: even with the cutoff back at the
            # open value, a high-pass is still holding back everything it has
            # integrated, and dropping that in one sample is a step. It was the
            # worst discontinuity in the render -- 8 ms past a drop's last
            # downbeat. Crossfading wet into dry over the resolve window means
            # the two are already equal by the time the swap happens.
            keep, come = _equal_power(np.linspace(0.0, 1.0, back, dtype=np.float32),
                                      wet.ndim == 2)
            wet[len(wet) - back:] = wet[len(wet) - back:] * keep + seg[len(seg) - back:] * come
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
        if self._gestures(slots[index], "in", "drop_in") or slots[index].kind == "drop":
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
        self.vocal_bed = np.zeros((self.n, 2), dtype=np.float32) \
            if self.stems.split else None
        seam = int(round(LOOP_SEAM_BEATS * self.beat * self.sr))
        last = len(self.plan.slots) - 1
        for i, slot in enumerate(self.plan.slots):
            a = self._bar_sample(slot.start_bar)
            want = self._bar_sample(slot.end_bar) - a
            pre = self._xfade_len(i)
            post = self._fadeout_len(i)
            head = 0 if i > 0 else self._beat_fade(slot)

            # `source_end` is the planner's exact warped exit point, chosen on a
            # downbeat of the source's own phrasing; falling back to the bar
            # arithmetic only matters for a plan written before it existed.
            span = getattr(slot, "source_end", 0.0) - slot.source_start
            period = int(round(span * self.sr)) if span > 1e-6 else int(round(
                max(slot.source_bars, 1) * self.beat_multiple * self.bar_dur * self.sr))
            start = int(round(slot.source_start * self.sr))
            layers = [(self.stems.vocal, 0.0), (self.stems.other,
                                                OTHER_DUCK_DB.get(slot.kind, 0.0))] \
                if self.stems.split else [(self.stems.harmonic, 0.0)]
            segs = [(_loop_to(buf, start, want, period, self.sr, pre=pre, seam=seam),
                     10.0 ** (duck / 20.0)) for buf, duck in layers]
            pseg = _loop_to(self.stems.percussive, start, want, period, self.sr,
                            pre=pre, seam=seam)

            hp = self._sweep_over(slot.highpass, want, pre)
            lp = self._sweep_over(slot.lowpass, want, pre)
            shaped = []
            for seg, duck in segs:
                if hp is not None:
                    seg = FL.sweep(seg, "highpass", self.sr, hp, q=0.72, order=2)
                if lp is not None:
                    seg = FL.sweep(seg, "lowpass", self.sr, lp, q=0.72, order=2)
                if slot.chops:
                    nb = min(4, slot.bars)
                    cut = want - self._bar_sample(nb)
                    if cut > 0:
                        chopped = _chop(seg[pre + cut:], self.sr, self.beat, nb,
                                        self.bar_dur)
                        seg = np.concatenate([seg[: pre + cut], chopped])[: pre + want]
                if slot.reverb_throw:
                    seg = self._throw(seg, slot)
                if self._kind(i + 1) == "breakdown":
                    self._exit_to_breakdown(seg, self.plan.slots[i + 1], want)
                    self._spill_throw(harm, seg, a + want, slot.source_gain * duck)

                for n_fade, at_head in ((pre or head, True), (post, False)):
                    if n_fade <= 1:
                        continue
                    t = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
                    out_g, in_g = _equal_power(t, True)
                    if at_head:
                        seg[:n_fade] *= in_g
                    else:
                        seg[len(seg) - n_fade:] *= out_g
                shaped.append((seg, duck))

            for k, (seg, duck) in enumerate(shaped):
                add_at(harm, seg, a - pre, slot.source_gain * duck)
                if k == 0 and self.stems.split:
                    add_at(self.vocal_bed, seg, a - pre, slot.source_gain * duck)
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

    def _exit_beats(self, nxt: Slot) -> float:
        """How many beats of sweep the plan wants over a ``drums_out`` boundary."""
        descs = self._gestures(nxt, "in", "drums_out")
        return float(descs[0].get("beats", EXIT_SWEEP_BEATS)) if descs \
            else EXIT_SWEEP_BEATS

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
        k = min(self._beat_sample(self._exit_beats(nxt)), want // 2)
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
            outs = self._gestures(nxt, "in", "drums_out")
            if outs and self._has_kick(slot):
                beats = self._exit_beats(nxt)
                strength = float(outs[0].get("strength", 1.0))
                k = self._beat_sample(beats)
                self._tail_sweep(drums, b - k, b, "highpass", 20.0,
                                 EXIT_KIT_HIGHPASS_HZ * strength + 20.0 * (1 - strength),
                                 resolve=True)
                self._duck_into(drums, b, beats, EXIT_KIT_DUCK_DB * strength)
            for d in self._gestures(slot, "out", "silence_beat"):
                a0 = self._beat_sample(d.get("beat", 0))
                self._duck_into(drums, a0 + self._beat_sample(d.get("beats", 1)),
                                float(d.get("beats", 1)), -60.0)


    def render_drums(self) -> tuple[np.ndarray, list[float]]:
        """Synthesise the kit across the whole arrangement.

        Returns the drum bus and the list of kick times, which drives sidechain.
        """
        out = np.zeros((self.n, 2), dtype=np.float32)
        kick_times: list[float] = []
        step_dur = self.bar_dur / 16.0

        for i, slot in enumerate(self.plan.slots):
            pattern = self._pattern(slot)
            entering = slot.kind == "build" and not self._has_kick(self._slot(i - 1))
            # Once the planner writes transition descriptors it decides where a
            # fill goes -- only about one into-drop boundary in five has one --
            # and `slot.fill` is only a fallback for a plan written without them.
            planned = bool(slot.transition_in or slot.transition_out)
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
                if last_of_phrase and not planned and not (entering and b < slot.bars - 2):
                    self._fill(out, bar_t, step_dur)
            sweeps = self._gestures(slot, "out", "sweep_up")
            for d in sweeps:
                self._riser(out, slot, d)
            if not sweeps and slot.riser:
                self._riser(out, slot, None)
            for d in self._gestures(slot, "out", "reverse_cymbal"):
                self._reverse_cymbal(out, d)
            for d in self._gestures(slot, "out", "fill"):
                beat = int(d.get("beat", 0))
                self._fill(out, beat * self.beat, self.bar_dur / 16.0)
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

    def _reverse_cymbal(self, out: np.ndarray, desc: dict) -> None:
        """A reversed crash swelling into the downbeat the descriptor names.

        Noise through a rising band-pass, amplitude reversed so it grows into
        the boundary and stops there -- the standard way of covering a section
        change that a bare cut leaves exposed.
        """
        beats = max(1.0, float(desc.get("beats", 2)))
        n = self._beat_sample(beats)
        if n < 256:
            return
        rng = np.random.default_rng(101)
        noise = rng.standard_normal(n).astype(np.float32)
        swept = FL.sweep(noise, "bandpass", self.sr,
                         FL.exp_curve(n, 900.0, 9000.0), q=0.7, order=2)
        env = (np.linspace(0.0, 1.0, n, dtype=np.float32) ** 2.4)
        env[-min(n, int(0.008 * self.sr)):] *= np.linspace(
            1.0, 0.0, min(n, int(0.008 * self.sr)), dtype=np.float32)
        swell = swept * env
        peak = float(np.max(np.abs(swell)))
        if peak <= 0:
            return
        add_at(out, to_stereo(swell / peak),
               self._beat_sample(desc.get("beat", 0)),
               0.34 * float(desc.get("strength", 1.0)))

    def _riser(self, out: np.ndarray, slot: Slot, desc: dict | None) -> None:
        """Noise riser over the build's last two bars, peaking on the next downbeat.

        Two changes from the eight-bar ride this used to be. It is short: the
        references run their risers over the last two bars and only on about one
        drop in three, and long monotonic ramps into a drop do not appear in the
        corpus at all. And it lands: stopping a full-scale noise sweep on a
        sample boundary is a full-amplitude step, and was the single loudest
        click in the render. It now starts from silence, peaks on the downbeat
        and washes out over half a beat past it.
        """
        if desc is not None:
            a = self._beat_sample(desc.get("beat", 0))
            seconds = max(1, int(desc.get("beats", 8))) * self.beat
            gain = 0.74 * float(desc.get("strength", 1.0))
        else:
            bars = min(RISER_BARS, slot.bars)
            a = self._bar_sample(slot.end_bar - bars)
            seconds = bars * self.bar_dur
            gain = 0.74
        tail = int(round(RISER_TAIL_BEATS * self.beat * self.sr))
        r = to_stereo(DR.riser_noise(self.sr, seconds))
        lead = min(int(0.01 * self.sr), len(r) // 8)
        if lead > 1:
            r = np.array(r, dtype=np.float32, copy=True)
            r[:lead] *= np.linspace(0.0, 1.0, lead, dtype=np.float32)[:, None]
        if tail > 8:
            wash = r[len(r) - tail:] * np.linspace(1.0, 0.0, tail,
                                                   dtype=np.float32)[:, None] ** 2.0
            r = np.concatenate([r, wash])
        add_at(out, r, a, gain)

    def render_bass(self, kick_times: list[float]) -> np.ndarray:
        """The rolling house bass, slot by slot, on the plan's chord roots.

        The line itself comes from ``bass.render_bassline``: rolling sixteenths
        with slides and passing notes, kept inside a six-semitone span around
        50 Hz, which is what the references measure and what the octave-bouncing
        offbeat eighths this used to write were not. The engine's job is to hand
        it the chords for each slot and place the result on the grid.
        """
        out = np.zeros((self.n, 2), dtype=np.float32)
        for slot in self.plan.slots:
            if not slot.use_bass:
                continue
            chords = [self._slot_chord(slot, b) for b in range(slot.bars)]
            line = BA.render_bassline(self.sr, self.bar_dur, chords, bars=slot.bars,
                                      seed=slot.index)
            add_at(out, to_stereo(line), self._bar_sample(slot.start_bar))
            if slot.use_stabs:
                for b in range(slot.bars):
                    root, minor = chords[b]
                    bar_t = (slot.start_bar + b) * self.bar_dur
                    for k in (1, 3, 5, 7):
                        t = bar_t + k * (self.bar_dur / 8.0)
                        stab = BA.stab(self.sr, root, minor, self.bar_dur / 8.0 * 0.8)
                        add_at(out, to_stereo(_land(stab, self.sr)),
                               int(round(t * self.sr)), 0.55)
        return out

    def render(self) -> tuple[np.ndarray, dict]:
        """Full render. Returns ``(audio, metrics)``."""
        harm, perc = self.render_source()
        drums, kick_times = self.render_drums()
        self.shape_kit_transitions(drums)
        bassline = self.render_bass(kick_times)

        env = self._duck_curve(kick_times)
        harm = DY.split_sidechain(harm, self.sr, env)
        perc = DY.split_sidechain(perc, self.sr, env)

        # the bass always ducks under the kick, hard, and full-band: it has
        # nothing above the crossover worth protecting.
        bass_env = DY.sidechain_envelope(
            self.n, self.sr, np.asarray(kick_times),
            depth=1.0 - 10.0 ** (BASS_DIP_DB / 20.0),
            release=SIDECHAIN_RECOVERY / self._recovery_fraction(BASS_DIP_DB),
            shape=SIDECHAIN_SHAPE)
        bassline = DY.apply_sidechain(bassline, bass_env)
        # carve 40-90 Hz out of the source so the kick and bass own the sub
        harm = FL.apply(harm, "highpass", self.sr, 105.0, q=0.707, order=2)

        harm, drums, bassline = self.balance_bands(harm, perc, drums, bassline)

        self.buses = {"harmonic": harm * HARM_GAIN, "percussive": perc * PERC_GAIN,
                      "drums": drums * DRUM_GAIN, "bass": bassline * BASS_GAIN}
        mix = sum(self.buses.values())
        out = DY.master(mix, self.sr, peak_db=-1.0)

        metrics = {
            "kick_count": len(kick_times),
            "peak_db": float(20 * np.log10(max(float(np.max(np.abs(out))), 1e-6))),
            "rms_db": float(20 * np.log10(max(float(np.sqrt(np.mean(out ** 2))), 1e-6))),
            "balance": self.balance_report(),
            "balance_moves": self.balance_moves,
        }
        return out, metrics

    # -- balance ---------------------------------------------------------
    @staticmethod
    def _recovery_fraction(dip_db: float, shape: float = SIDECHAIN_SHAPE) -> float:
        """Fraction of the release a ``dip_db`` duck spends getting back to -1 dB.

        ``sidechain_envelope`` recovers along ``(1-d) + d*t**shape``. Solving it
        for the moment the envelope reaches -1 dB turns the corpus's "recovers
        in 80 to 110 ms" into a release time, whatever depth the plan asked for.
        """
        d = 1.0 - 10.0 ** (dip_db / 20.0)
        if d <= 1e-6:
            return 1.0
        t = (10.0 ** (-1.0 / 20.0) - 1.0 + d) / d
        return float(np.clip(t, 1e-3, 1.0) ** (1.0 / shape))

    def _duck_curve(self, kick_times: list[float]) -> np.ndarray:
        """One ducking envelope for the whole track, with a per-slot depth.

        The old code built a separate envelope per depth value and pasted them
        together at slot boundaries, so the amount of ducking stepped at every
        edit. Here the envelope is built once at full depth and scaled by a
        per-sample depth curve that ramps across boundaries, which is both
        continuous and cheaper.

        The plan's ``sidechain`` numbers are relative weights, not decibels, so
        they are rescaled to land the deepest one at the corpus's -14 dB.
        """
        deepest = max((s.sidechain for s in self.plan.slots), default=0.0)
        if deepest <= 0:
            return np.ones(self.n, dtype=np.float32)
        want = 1.0 - 10.0 ** (SIDECHAIN_LOW_DIP_DB / 20.0)
        scale = min(want / deepest, 0.95 / deepest)
        duck = DY.sidechain_envelope(
            self.n, self.sr, np.asarray(kick_times), depth=1.0, attack=0.004,
            release=SIDECHAIN_RECOVERY / self._recovery_fraction(SIDECHAIN_LOW_DIP_DB),
            shape=SIDECHAIN_SHAPE)
        depth = self._slot_curve([min(s.sidechain * scale, 0.95)
                                  for s in self.plan.slots])
        return (1.0 - (1.0 - duck) * depth).astype(np.float32)

    def _slot_curve(self, values: list[float], ramp_beats: float = 1.0) -> np.ndarray:
        """Per-slot numbers as a per-sample curve that ramps across boundaries.

        Anything decided per slot -- a ducking depth, a corrective gain -- has to
        arrive as a ramp rather than as a step, or the fix for one complaint
        becomes an instance of the other.
        """
        out = np.zeros(self.n, dtype=np.float32)
        for slot, v in zip(self.plan.slots, values):
            a, b = self._bar_sample(slot.start_bar), self._bar_sample(slot.end_bar)
            out[a:b] = float(v)
        if len(out):
            out[self._bar_sample(self.plan.total_bars):] = float(values[-1]) if values else 0.0
        r = int(round(ramp_beats * self.beat * self.sr))
        if r > 1:
            for i in range(1, len(self.plan.slots)):
                b = self._bar_sample(self.plan.slots[i].start_bar)
                lo, hi = max(0, b - r // 2), min(self.n, b + r // 2)
                if hi - lo > 1:
                    out[lo:hi] = np.linspace(float(out[lo]), float(out[hi - 1]),
                                             hi - lo, dtype=np.float32)
        return out

    def _slot_levels(self, buf: np.ndarray, band: tuple[float, float],
                     gain: float) -> list[float]:
        """Per-slot band level of a bus, in dB, at the gain the master will see."""
        out = []
        for slot in self.plan.slots:
            a, b = self._bar_sample(slot.start_bar), self._bar_sample(slot.end_bar)
            out.append(DY.band_rms_db(buf[a:b] * gain, self.sr, band))
        return out

    def balance_bands(self, harm: np.ndarray, perc: np.ndarray, drums: np.ndarray,
                      bassline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Put the source in front of the kit where it has to be, by measurement.

        Nothing here is a fixed gain. Every move is the distance between what
        this particular render measures and a target taken from commercial
        remixes, clamped, smoothed across slot boundaries and -- for the
        presence move -- gated by an envelope of the source's own vocal-band
        energy, so the lift happens while there is a voice to lift and stops
        when there is not.

        The corrections, in order:

        0. **Staging.** The drum bus against the full-band target and then the
           bass against the kick in 40 - 120 Hz, both measured rather than
           assumed, so a rebuilt kit or bass line does not silently move the
           whole balance.
        1. **Body.** The source's 300 Hz - 4 kHz level against the whole bed,
           trimmed toward the reference's +1.7 dB. On the fixture it was +6.9:
           the pad was sitting where the kick and bass belong, which is why the
           bed measured 6 dB weaker full-band than the reference's 12.
        2. **Presence.** Then, on the trimmed source, 2 - 5 kHz against the kit.
           Half the gap is closed by a bell on the source and half by the same
           bell inverted on the kit, so the section does not simply get brighter
           -- the hats and the clap move out of the way and the voice moves into
           the gap. Body runs first because trimming the source also trims its
           presence, and the two corrections used to undo each other.
        3. **Low mids**, by the amount the presence bell put back into the body.
           It comes out at 693 Hz rather than off the whole source, so it does
           not simply undo step 2 -- the net move is a tilt, not a fader.

        Only sections that actually play a kick are corrected; a breakdown has
        no kit to balance against.
        """
        centre = float(np.sqrt(DY.PRESENCE_BAND[0] * DY.PRESENCE_BAND[1]))
        full_band = (0.0, self.sr * 0.5)
        live = [self._has_kick(slot) for slot in self.plan.slots]
        n_slots = len(self.plan.slots)

        def bed_of(kit: np.ndarray, stage: float = 1.0) -> np.ndarray:
            return (kit * DRUM_GAIN + bassline * BASS_GAIN) * stage + perc * PERC_GAIN

        # 0. stage the bed. The kit and the bass line are built in their own
        #    modules and their absolute level is theirs to choose -- a kick
        #    given a shorter, truer sub tail drops the drum bus by the better
        #    part of ten dB without anything here being wrong -- so the mix
        #    measures what it was handed rather than assuming it. Two gains,
        #    solved in order against two targets, sized off the drops because
        #    that is where both targets are defined.
        def stage_of(want: float, got: list[float], lo_hi) -> float:
            live_drops = [want - got[i] for i in range(n_slots)
                          if live[i] and self.plan.slots[i].kind == "drop"]
            return float(np.clip(np.median(live_drops) if live_drops else 0.0, *lo_hi))

        src_f = self._slot_levels(harm, full_band, HARM_GAIN)
        bed_f = self._slot_levels(bed_of(drums), full_band, 1.0)
        self.drum_stage_db = stage_of(
            -FULL_BAND_TARGET_DB, [bed_f[i] - src_f[i] for i in range(n_slots)],
            DRUM_STAGE_RANGE_DB)
        drums = drums * 10.0 ** (self.drum_stage_db / 20.0)

        kick_low = self._slot_levels(drums, BASS_BAND, DRUM_GAIN)
        bass_low = self._slot_levels(bassline, BASS_BAND, BASS_GAIN)
        self.bass_stage_db = stage_of(
            BASS_VS_KICK_DB, [bass_low[i] - kick_low[i] for i in range(n_slots)],
            BASS_STAGE_RANGE_DB)
        bassline = bassline * 10.0 ** (self.bass_stage_db / 20.0)

        # 1. body, so the source stops crowding the bed's own band
        src_v = self._slot_levels(harm, DY.VOCAL_BAND, HARM_GAIN)
        bed_v = self._slot_levels(bed_of(drums), DY.VOCAL_BAND, 1.0)
        trim = [float(np.clip(VOCAL_BAND_TARGET_DB - (src_v[i] - bed_v[i]),
                              -BALANCE_TRIM_DOWN_DB, BALANCE_TRIM_UP_DB))
                if live[i] else 0.0 for i in range(n_slots)]
        if any(trim):
            gain = (10.0 ** (self._slot_curve(trim) / 20.0))[:, None]
            harm = harm * gain
            if self.vocal_bed is not None:
                self.vocal_bed = self.vocal_bed * gain

        # 2. presence, measured on the trimmed source so the two do not fight
        drive = self._vocal_drive(harm)
        src_p = self._slot_levels(harm, DY.PRESENCE_BAND, HARM_GAIN)
        kit_p = self._slot_levels(drums, DY.PRESENCE_BAND, DRUM_GAIN)
        head = float(np.mean(drive[drive > 0.05])) if np.any(drive > 0.05) else 1.0
        lift, cut = [], []
        for i in range(n_slots):
            gap = max(0.0, PRESENCE_TARGET_DB - (src_p[i] - kit_p[i])) if live[i] else 0.0
            # the bell only plays at `drive`, and a bell centred in a band lifts
            # the band's RMS by less than its own peak gain, so ask for the
            # correction the measurement wants rather than half of it
            gap = gap / max(head, 0.2)
            lift.append(min(gap / 2.0, PRESENCE_LIFT_MAX_DB))
            cut.append(-min(gap / 2.0, PRESENCE_CUT_MAX_DB))
        if any(lift):
            gains = DY.db_to_bell(self._slot_curve(lift) * drive)
            if self.vocal_bed is not None:
                # with demucs the lift belongs to the voice alone: brightening
                # the pads too would just move the mud up the spectrum
                lifted = DY.moving_bell(self.vocal_bed, self.sr, centre, gains,
                                        q=PRESENCE_Q)
                harm = harm + (lifted - self.vocal_bed)
                self.vocal_bed = lifted
            else:
                harm = DY.moving_bell(harm, self.sr, centre, gains, q=PRESENCE_Q)
            drums = DY.moving_bell(drums, self.sr, centre,
                                   DY.db_to_bell(self._slot_curve(cut) * drive),
                                   q=PRESENCE_Q)

        # 3. what the presence bell put back into the body, taken out of the low
        #    mids rather than off the whole source
        src_v = self._slot_levels(harm, DY.VOCAL_BAND, HARM_GAIN)
        bed_v = self._slot_levels(bed_of(drums), DY.VOCAL_BAND, 1.0)
        body = [float(np.clip((VOCAL_BAND_TARGET_DB - (src_v[i] - bed_v[i]))
                              * BODY_CUT_LEVERAGE, -BODY_CUT_MAX_DB, 0.0))
                if live[i] else 0.0 for i in range(n_slots)]
        if any(body):
            harm = DY.moving_bell(harm, self.sr,
                                  float(np.sqrt(BODY_BAND[0] * BODY_BAND[1])),
                                  DY.db_to_bell(self._slot_curve(body)), q=BODY_Q)

        self.balance_moves = [
            {"slot": i, "kind": self.plan.slots[i].kind,
             "source_trim_db": round(trim[i], 2),
             "presence_lift_db": round(lift[i], 2),
             "kit_presence_db": round(cut[i], 2),
             "source_low_mid_db": round(body[i], 2),
             "drum_stage_db": round(self.drum_stage_db, 2),
             "bass_stage_db": round(self.bass_stage_db, 2)}
            for i in range(n_slots)]
        return harm.astype(np.float32), drums.astype(np.float32), bassline.astype(np.float32)

    def _vocal_drive(self, harm: np.ndarray) -> np.ndarray:
        """0 to 1 with the source's own vocal-band energy, per sample.

        Normalised against the loud end of its own distribution rather than
        against an absolute level, so a quiet source gets the same treatment as
        a loud one and the correction tracks phrases instead of sections.
        """
        env = DY.band_follower(harm, self.sr, DY.VOCAL_BAND, attack=0.02, release=0.18)
        ref = float(np.percentile(env, 75.0))
        if ref <= 1e-9:
            return np.zeros(len(env), dtype=np.float32)
        return (np.clip(env / ref, 0.0, 1.0) ** 0.6).astype(np.float32)

    def entry_cutoff(self, index: int) -> tuple[float, float]:
        """The (high-pass, low-pass) cutoffs in force at slot ``index``'s first sample.

        ``(0, sr/2)`` means open. This is the invariant a drop has to satisfy:
        whatever the build swept up to must be gone by the first kick, not a
        filter block later, and nothing the previous slot was doing on its way
        out may still be in force. The previous slot's exit sweep is taken into
        account, because a sweep that has not resolved by the downbeat is
        exactly the kind of "rough transition" this is here to catch.
        """
        open_hp, open_lp = 0.0, self.sr * 0.5
        slot = self._slot(index)
        if slot is None:
            return open_hp, open_lp
        hp = float(slot.highpass[0]) if slot.highpass else open_hp
        lp = float(slot.lowpass[0]) if slot.lowpass else open_lp
        prev = self._slot(index - 1)
        if prev is not None and slot.kind == "breakdown" and prev.lowpass is None:
            # the outgoing section hands the breakdown its own opening cutoff
            lp = min(lp, float(slot.lowpass[0]) if slot.lowpass else EXIT_LOWPASS_HZ)
        return hp, lp

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
