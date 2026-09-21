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
from ..dsp import pitch as PI
from ..dsp import reverb as RV
from . import bass as BA
from . import drums as DR


@dataclass
class Stems:
    """The source, pre-warped to the target grid and split into parts."""

    harmonic: np.ndarray      # vocals + chords (or demucs vocals + other)
    percussive: np.ndarray    # original drums, kept only as breakdown texture
    source_name: str = "hpss"
    bass: np.ndarray | None = None
    """The song's own low end, warped with everything else. When this is here
    the engine plays it instead of inventing a bassline from a chord estimate,
    which is the difference between a remix of a song and a remix of a guess."""
    bass_name: str = "synth"
    vocals: np.ndarray | None = None
    """The voice on its own, when the separation can give one. Kept apart from
    the rest of the harmonic bed so a drop can push everything else down and
    out of the vocal's way instead of turning the whole bed down with it."""
    other: np.ndarray | None = None

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

    # Equal-gain, not equal-power. A loop seam joins the end of a bar to the
    # start of the same bar, and in sustained material -- a pad, a held chord,
    # a sub -- those are nearly the same signal. Crossfading correlated audio
    # with sine/cosine gains sums to 1.41, a 3 dB bump at every seam, which on a
    # looped pad is a pulse you can hear. Linear gains sum to one whatever the
    # correlation, at the cost of a small dip where the two sides are unrelated.
    rise = np.linspace(0.0, 1.0, fade, dtype=np.float32) if fade > 1 else None
    fall = (1.0 - rise) if rise is not None else None
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


#: What each kind of slot does to a sampled drum loop: (level, high-pass Hz).
#: A record's drums arrive as one finished stereo bus, so the only honest way to
#: arrange them is the way a DJ does it on a mixer -- with the level and the
#: filter. The intro is thinned so it does not fight whatever is still playing;
#: the breakdown keeps only the top of the loop, which is the hats, so the
#: section breathes without going silent; the drop is the record.
KIT_SECTION: dict[str, tuple[float, float | None]] = {
    "intro": (0.55, 220.0),
    "intro_full": (0.78, 150.0),
    "build": (0.88, 120.0),
    "drop": (1.0, None),
    # The second drop has to be the bigger one, or the track sags where it
    # should peak: a decibel and a half of level, an extra offbeat hat layer
    # and a lift at the top of the loop.
    "drop_var": (1.18, None),
    "breakdown": (0.34, 3800.0),
    "outro": (0.72, 150.0),
}

#: What a drop does to everything that is not the voice: ``(level, high-pass)``
#: for the separated ``other`` stem. A rap record's instrumental is a wall of
#: mid-range, and under a house kit it is the thing the vocal has to fight --
#: "the vocal drowns under mid-range clutter" was the listening note that put
#: this here. In the drops it comes down seven decibels and loses everything
#: below 250 Hz, which is the kick's and the bass's anyway. In the breakdown it
#: comes all the way back, because there the instrumental *is* the section.
OTHER_SECTION: dict[str, tuple[float, float | None]] = {
    "intro": (0.70, 180.0),
    "intro_full": (0.60, 200.0),
    "build": (0.58, 220.0),
    "drop": (0.42, 250.0),
    "drop_var": (0.40, 260.0),
    "breakdown": (1.0, None),
    "outro": (0.78, 180.0),
}

#: How much comes off a sampled loop's own body, in decibels. The listening
#: note on two renders Neel sat through was the same: "drums a little quieter".
#: It comes off the loop and the layers on top of it, never off the sub kick
#: underneath -- what was too loud was the record's mids and highs against the
#: vocal, not the weight below them, and taking the whole bus down would have
#: removed exactly the part that was right.
LOOP_TRIM_DB = -1.75

#: How loud the synthesised kick sits under the loop's own kicks, per section.
#: A sampled loop from a 2015 record often has less sub than a 2024 system
#: expects; this puts it back without replacing the loop's character.
KIT_REINFORCE: dict[str, float] = {
    "intro": 0.34, "intro_full": 0.46, "build": 0.54,
    "drop": 0.68, "drop_var": 0.72, "breakdown": 0.0, "outro": 0.42,
}


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
                 src_bar_dur: float = 2.0, seed: int = 0, warp=None,
                 drum_kit=None, kick_reinforce: bool = True,
                 bass_mode: str = "source", drums_db: float = 0.0,
                 vocal_mode: str = "flow") -> None:
        self.sr = sr
        self.plan = plan
        self.stems = stems
        self.chords = chords
        self.semitones = semitones
        self.beat_multiple = beat_multiple
        self.src_bar_dur = src_bar_dur
        self.warp = warp
        self.kit = DR.Kit(sr=sr, swing=swing)
        # Not the kit's own kick. This one goes underneath a sampled loop that
        # already has a beater click and a body of its own; what it is there to
        # add is the sub a record cut in 2015 does not have, so it is tuned an
        # octave down from the synthesised kit's, with almost no click and a
        # short tail that clears before the offbeat.
        self.sub_kick = DR.kick(sr, length=0.42, f_start=130.0, f_end=47.0,
                                pitch_decay=0.036, amp_decay=0.115, click=0.10,
                                drive=2.2)
        self.drum_kit = drum_kit
        self.kick_reinforce = kick_reinforce
        self.bass_mode = bass_mode
        self.drums_gain = 10.0 ** (float(drums_db) / 20.0)
        self.vocal_mode = vocal_mode
        self.source_bass_bed: np.ndarray | None = None
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
        # The song's own low end, laid out through the identical spans so it
        # cannot drift away from the part of the song it belongs to.
        bass = (np.zeros((self.n, 2), dtype=np.float32)
                if self.stems.bass is not None else None)
        # The voice on its own, laid out exactly as it is mixed, so the gate can
        # ask the question a listener asks: is the singing on the beat?
        vocal_bed = np.zeros((self.n, 2), dtype=np.float32)
        self.source_spans = []
        for slot in self.plan.slots:
            a = self._bar_sample(slot.start_bar)
            want = self._bar_sample(slot.end_bar) - a
            self.source_spans.append((a, a + want))
            period = int(round(max(slot.source_bars, 1) * self.bar_dur * self.sr))
            start = int(round(slot.source_start * self.sr))
            seg, voc_only = self._harmonic_span(slot, start, want, period)
            add_at(vocal_bed, voc_only, a, slot.source_gain)
            pseg = _loop_to(self.stems.percussive, start, want, period, self.sr)
            if bass is not None:
                add_at(bass, _loop_to(self.stems.bass, start, want, period, self.sr),
                       a, 1.0)

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
        self.layers["vocals"] = vocal_bed
        self.layers["source_perc"] = perc_ref
        # The percussive bed as it is actually mixed -- which with a real kit in
        # play should be silence, and the gate checks that rather than trusting
        # a reading of the arrangement code.
        self.layers["source_drums"] = perc
        self.source_bass_bed = bass
        return harm, perc

    #: One four-bar unit of chopped vocal, as ``(what, beats)``: fresh phrases,
    #: the first one again, a two-beat stutter of it, and gaps for the drums to
    #: answer into. It is the oldest arrangement in dance music and it is what a
    #: listener recognises as "the vocal splits and repeats".
    #:
    #: The pieces are one and two beats, not four. A slice only re-synchronises
    #: at its own start, so with a triplet flow inside it the longer the slice
    #: the further the voice gets from the beat before the next one pulls it
    #: back. Two beats is about as long as a triplet can run before it is
    #: audibly arguing with the kick.
    CHOP_PATTERN = (("take", 2), ("take", 2), ("again", 2), ("rest", 2),
                    ("take", 2), ("stut", 1), ("stut", 1), ("again", 2),
                    ("rest", 2))

    def _chop_vocal(self, voc: np.ndarray, want: int) -> np.ndarray:
        """Rebuild a vocal out of slices that start on syllables and land on beats.

        Each slice begins at a vocal onset -- a syllable, not an arbitrary
        sample -- and is placed at a grid position, so its first and loudest
        transient is exactly on the beat by construction. Nothing is stretched.
        Inside a slice the voice keeps the timing it was sung with, which is the
        point: a triplet triplet-feels for four beats and then the next slice
        re-synchronises, instead of a whole verse walking away from the kick.
        """
        from ..analysis.alignment import onset_times

        beat_n = max(64, int(round(self.beat * self.sr)))
        out = np.zeros((want, voc.shape[1]) if voc.ndim == 2 else (want,),
                       dtype=np.float32)
        onsets, strength = onset_times(voc, self.sr)
        if len(onsets) < 4:
            return np.array(fit(voc, want), dtype=np.float32, copy=True)
        # Only syllables worth starting a phrase on.
        keep = np.asarray(strength) >= np.percentile(strength, 35)
        starts = (np.asarray(onsets)[keep] * self.sr).astype(int)
        starts = starts[starts < len(voc) - beat_n]
        if len(starts) < 4:
            return np.array(fit(voc, want), dtype=np.float32, copy=True)

        fade_in = max(8, int(0.006 * self.sr))
        fade_out = max(16, int(0.014 * self.sr))

        def slice_at(i: int, beats: int) -> np.ndarray:
            a = int(starts[i % len(starts)])
            seg = np.array(fit(voc[a: a + beats * beat_n], beats * beat_n),
                           dtype=np.float32, copy=True)
            env = np.ones(len(seg), dtype=np.float32)
            env[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
            env[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
            return seg * (env[:, None] if seg.ndim == 2 else env)

        pos, cursor, first, last = 0, 0, None, None
        while pos < want:
            for what, beats in self.CHOP_PATTERN:
                if pos >= want:
                    break
                n = beats * beat_n
                if what == "take":
                    seg = slice_at(cursor, beats)
                    cursor += 1
                    if first is None:
                        first = seg
                    last = seg
                elif what == "again":
                    seg = first if first is not None else last
                elif what == "stut":
                    seg = (last[:n] if last is not None and len(last) >= n
                           else slice_at(cursor, beats))
                else:
                    pos += n
                    continue
                add_at(out, seg[:min(n, want - pos)], pos, 1.0)
                pos += n
            first = None                 # a new four bars, a new phrase to keep
        return out

    def _harmonic_span(self, slot: Slot, start: int, want: int,
                       period: int) -> np.ndarray:
        """The source bed for one slot, with the instrumental put in its place.

        When the separation gave a vocal and an "everything else" apart, a drop
        takes seven decibels off the everything-else and high-passes it, so the
        voice and the kit own the middle of the mix; a breakdown hands it all
        back, because there the instrumental is the section. Without separate
        stems this is the harmonic bed as it comes.
        """
        if self.stems.vocals is None or self.stems.other is None:
            bed = _loop_to(self.stems.harmonic, start, want, period, self.sr)
            return bed, np.zeros_like(bed)
        voc = _loop_to(self.stems.vocals, start, want, period, self.sr)
        oth = _loop_to(self.stems.other, start, want, period, self.sr)
        # The breakdown keeps its vocal whole. There are no drums there for it
        # to fight, and one long clean phrase is what a breakdown is for.
        if self.vocal_mode == "chop" and slot.drum_pattern != "breakdown":
            voc = self._chop_vocal(voc, want)
        gain, hp = OTHER_SECTION.get(slot.drum_pattern, (1.0, None))
        if hp is not None:
            oth = FL.apply(oth, "highpass", self.sr, hp, q=0.707, order=2)
        return (voc + oth * gain).astype(np.float32), voc

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
        """Build the drum bus across the whole arrangement.

        Returns the bus and the list of kick times, which drives sidechain. A
        sampled kit replaces the synthesised pattern entirely when one is
        available; the risers, impacts and fills sit on top either way.
        """
        if self.drum_kit is not None:
            return self._render_sampled_drums()
        return self._render_synth_drums()

    def _render_sampled_drums(self) -> tuple[np.ndarray, list[float]]:
        """Lay a real record's eight bars across the arrangement.

        The loop is stretched to the target tempo once and tiled from bar zero,
        so its bar one is the remix's bar one and stays there: the tiling is by
        position, not by appending, which is the only way a loop repeated forty
        times finishes where the grid says it should.
        """
        loop, kick_offsets = self.drum_kit.at_bpm(self.plan.target_bpm)
        loop = to_stereo(np.asarray(loop, dtype=np.float32))
        period = len(loop)
        if period <= 0:
            return self._render_synth_drums()
        # `_loop_to` hides each seam in the loop's own head, which is the right
        # material: the loop was cut at a bar line out of continuous audio.
        bed = _loop_to(np.concatenate([loop, loop]), 0, self.n, period, self.sr)

        out = np.zeros((self.n, 2), dtype=np.float32)
        subs = np.zeros((self.n, 2), dtype=np.float32)      # trimmed separately
        kick_times: list[float] = []
        for slot in self.plan.slots:
            a = self._bar_sample(slot.start_bar)
            b = self._bar_sample(slot.end_bar)
            gain, hp = KIT_SECTION.get(slot.drum_pattern, KIT_SECTION["drop"])
            seg = np.array(bed[a:b], dtype=np.float32, copy=True)
            if hp is not None:
                seg = FL.apply(seg, "highpass", self.sr, hp, q=0.707, order=2)
            if slot.drum_pattern == "drop_var":
                seg = FL.apply(seg, "highshelf", self.sr, 7000.0, gain_db=2.5)
            edge = min(int(0.008 * self.sr), len(seg) // 8)
            if edge > 1:
                seg[:edge] *= np.linspace(0.0, 1.0, edge)[:, None]
                seg[-edge:] *= np.linspace(1.0, 0.0, edge)[:, None]
            add_at(out, seg, a, gain)

            reinforce = KIT_REINFORCE.get(slot.drum_pattern, 0.0) if self.kick_reinforce else 0.0
            for t in self._kick_times_in(kick_offsets, period, a, b):
                if reinforce > 0:
                    add_at(subs, to_stereo(self.sub_kick),
                           int(round(t * self.sr)), reinforce)
                # A muted breakdown has no kick, so nothing should duck to one.
                if gain > 0.4:
                    kick_times.append(t)

            if slot.drum_pattern == "drop_var":
                self._lift(out, slot)
            if slot.riser:
                self._riser(out, slot)
            if slot.impact:
                add_at(out, to_stereo(DR.impact(self.sr)),
                       self._bar_sample(slot.start_bar), 0.9)
            if slot.fill:
                last_bar = (slot.start_bar + slot.bars - 1) * self.bar_dur
                self._fill(out, last_bar, self.bar_dur / 16.0)
        trim = 10.0 ** (LOOP_TRIM_DB / 20.0)
        return (out * trim + subs).astype(np.float32), sorted(kick_times)

    def _lift(self, out: np.ndarray, slot: Slot) -> None:
        """An extra offbeat open hat and a shaker over a sampled loop.

        A single eight-bar loop played twice is the same eight bars twice. The
        cheapest honest way to make the second drop bigger is the way a DJ does
        it on the fly: put another layer on top of it.
        """
        step = self.bar_dur / 16.0
        for b in range(slot.bars):
            bar_t = (slot.start_bar + b) * self.bar_dur
            for k in (2, 6, 10, 14):
                add_at(out, to_stereo(self.kit.samples["ohat"]),
                       int(round((bar_t + k * step) * self.sr)), 0.16)
            for k in range(1, 16, 4):
                add_at(out, to_stereo(self.kit.samples["shaker"]),
                       int(round((bar_t + k * step) * self.sr)), 0.09)

    def _kick_times_in(self, offsets: list[float], period: int,
                       a: int, b: int) -> list[float]:
        """Absolute times of the loop's own kicks inside a sample span."""
        if not offsets:
            return []
        loop_dur = period / float(self.sr)
        first = int(np.floor(a / float(period)))
        last = int(np.ceil(b / float(period)))
        out = []
        for k in range(first, last + 1):
            base = k * loop_dur
            for o in offsets:
                t = base + o
                if a <= t * self.sr < b:
                    out.append(float(t))
        return out

    def _render_synth_drums(self) -> tuple[np.ndarray, list[float]]:
        """Synthesise the kit across the whole arrangement.

        The fallback when no kit has been built. Returns the drum bus and the
        list of kick times, which drives sidechain.
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
        """The low end: the song's own, wherever the song has one.

        A synthesised bassline has to be told what note to play, and the only
        thing available to tell it is a chord estimate off the full mix. On a
        dense trap record that estimate is wrong often enough to matter, and a
        wrong bass note under a vocal singing the right one is not a stylistic
        choice, it is out of tune. So the default is the separated bass stem,
        warped and arranged exactly like the rest of the song.

        The one thing that does get synthesised is a sub, and only when the
        song's own bass has little below 80 Hz -- plenty of records do not, and
        a club system will find nothing there. Even then it follows the pitch
        the bass stem is actually playing, beat by beat, measured by
        autocorrelation. It never plays a note the record is not playing.
        """
        if self.bass_mode == "none":
            return np.zeros((self.n, 2), dtype=np.float32)
        if self.source_bass_bed is None or self.bass_mode == "synth":
            return self._render_synth_bass()
        if self.bass_mode == "sub":
            return self._render_house_sub()
        return self._render_source_bass()

    def _render_house_sub(self) -> np.ndarray:
        """A sub playing the record's notes on the house grid.

        What this is for: an 808 is a kick with a pitch, and under
        four-on-the-floor it reads as a second kick drum rather than as a bass
        line -- which is exactly what "two rhythms at once" means. Throwing it
        away and synthesising a bass line loses the song's harmony. Throwing
        away its *rhythm* and keeping its *pitch* loses nothing that matters:
        the notes are the record's, read off the separated stem one beat at a
        time by autocorrelation, and the rhythm is the one the kit is playing.

        One note per beat, held almost the whole beat, so the hard sidechain
        every bass bus gets turns it into the pump a house record has. Where the
        stem goes quiet or unpitched the sub goes with it, so it can never
        invent a bass line the record does not have.
        """
        bed = self.source_bass_bed
        out = np.zeros(self.n, dtype=np.float32)
        beat_n = max(64, int(round(self.beat * self.sr)))
        for slot in self.plan.slots:
            if not slot.use_bass:
                continue
            a, b = self._bar_sample(slot.start_bar), self._bar_sample(slot.end_bar)
            ref = float(np.sqrt(np.mean(np.square(bed[a:b].mean(axis=1)))))
            if ref <= 1e-6:
                continue
            phase, last_f0, carried = 0.0, 0.0, 0
            for i in range(a, b - beat_n + 1, beat_n):
                seg = bed[i:i + beat_n].mean(axis=1)
                level = float(np.sqrt(np.mean(np.square(seg))))
                f0 = PI.f0_autocorr(seg, self.sr, fmin=35.0, fmax=200.0) \
                    if level > 0.2 * ref else 0.0
                if f0 <= 0.0:
                    # hold the last note for one beat, then stop: the record
                    # has gone quiet and so should we
                    if last_f0 > 0.0 and carried < 1 and level > 0.1 * ref:
                        f0, carried = last_f0, carried + 1
                    else:
                        last_f0, phase = 0.0, 0.0
                        continue
                else:
                    carried = 0
                while f0 > 80.0:
                    f0 *= 0.5                # where a system can move air
                while f0 < 38.0:
                    f0 *= 2.0
                if last_f0 <= 0.0 or abs(f0 - last_f0) > 0.5:
                    phase = 0.0              # a new note starts from zero
                out[i:i + beat_n] += self._sub_note(f0, beat_n, phase) * level
                phase = float((phase + 2.0 * np.pi * f0 * beat_n / self.sr)
                              % (2.0 * np.pi))
                last_f0 = f0
        peak = float(np.max(np.abs(out)))
        if peak > 0:
            out = out / peak * 0.85
        return to_stereo(out)

    def _sub_note(self, f0: float, n: int, phase: float = 0.0) -> np.ndarray:
        """One sub note: a sine with just enough harmonic to survive a laptop."""
        t = np.arange(n) / self.sr
        # Almost pure. The listening note on the first attempt was that the low
        # end was "hollow and mid-heavy": harmonics of a 50 Hz note land in the
        # mid-range the vocal needs, and they are what makes a sub sound thin
        # rather than deep. One octave up at a tenth of the level is enough to
        # say where the note is on a laptop.
        sig = (np.sin(2.0 * np.pi * f0 * t + phase)
               + 0.11 * np.sin(2.0 * np.pi * 2 * f0 * t + 2 * phase))
        env = np.ones(n, dtype=np.float32)
        attack = max(8, int(0.006 * self.sr))
        release = max(16, int(0.030 * self.sr))
        env[:attack] *= np.linspace(0.0, 1.0, attack)
        env[-release:] *= np.linspace(1.0, 0.0, release)
        return (sig.astype(np.float32) * env * 1.6)

    def _render_source_bass(self) -> np.ndarray:
        bed = self.source_bass_bed
        out = np.zeros((self.n, 2), dtype=np.float32)
        for slot in self.plan.slots:
            if not slot.use_bass:
                continue
            a, b = self._bar_sample(slot.start_bar), self._bar_sample(slot.end_bar)
            seg = np.array(bed[a:b], dtype=np.float32, copy=True)
            edge = min(int(0.012 * self.sr), len(seg) // 8)
            if edge > 1:
                seg[:edge] *= np.linspace(0.0, 1.0, edge)[:, None]
                seg[-edge:] *= np.linspace(1.0, 0.0, edge)[:, None]
            add_at(out, seg, a, 1.0)
        out = FL.apply(out, "highpass", self.sr, 28.0, q=0.707, order=2)
        sub = self._sub_under(out)
        if sub is not None:
            out = out + sub
        return out

    def _sub_under(self, bass: np.ndarray) -> np.ndarray | None:
        """A sine following the bass stem's own pitch, when the sub is missing.

        Records mixed for phones, and plenty of older ones, have almost nothing
        below 80 Hz. On a club system that reads as no bass at all. This fills
        it in at the pitch the record is playing -- one reading per beat, from
        the bass stem itself -- and does nothing at all where the bass stem is
        silent or unpitched, which is what stops it inventing a bassline.
        """
        mono = bass.mean(axis=1)
        deep = FL.apply(mono, "lowpass", self.sr, 80.0, q=0.707, order=2)
        mid = FL.apply(mono, "bandpass", self.sr, 150.0, q=0.5, order=2)
        deep_rms = float(np.sqrt(np.mean(deep ** 2)))
        mid_rms = float(np.sqrt(np.mean(mid ** 2)))
        if mid_rms <= 1e-5 or deep_rms > 0.5 * mid_rms:
            return None                     # the record has its own sub

        beat_n = int(round(self.beat * self.sr))
        out = np.zeros(self.n, dtype=np.float32)
        phase = 0.0
        for i in range(0, self.n - beat_n, beat_n):
            seg = mono[i:i + beat_n]
            if float(np.sqrt(np.mean(seg ** 2))) < 0.2 * mid_rms:
                phase = 0.0
                continue
            f0 = PI.f0_autocorr(seg, self.sr, fmin=35.0, fmax=180.0)
            if f0 <= 0.0:
                phase = 0.0
                continue
            while f0 > 90.0:
                f0 *= 0.5                   # put it where a system can move air
            t = np.arange(beat_n) / self.sr
            env = np.ones(beat_n, dtype=np.float32)
            edge = max(8, beat_n // 24)
            env[:edge] *= np.linspace(0.0, 1.0, edge)
            env[-edge:] *= np.linspace(1.0, 0.0, edge)
            tone = np.sin(2.0 * np.pi * f0 * t + phase).astype(np.float32)
            out[i:i + beat_n] += tone * env * 0.5
            phase = float((phase + 2.0 * np.pi * f0 * beat_n / self.sr) % (2.0 * np.pi))
        return to_stereo(out * float(mid_rms) * 2.2)

    def _render_synth_bass(self) -> np.ndarray:
        """Rolling offbeat bass following the per-bar chord roots.

        The fallback for ``--bass synth``, and for a source nothing could be
        separated out of.
        """
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

        # Sidechain, split at 220 Hz. One envelope across the whole spectrum
        # has to be deep enough to clear the kick's sub, and then the vocal and
        # the hats breathe in and out with it -- the pumping wash that reads as
        # amateur. The bottom takes the deep, quick duck the kick needs; the top
        # moves just enough to read as groove.
        beat_scale = (60.0 / self.plan.target_bpm) / 0.4839
        kicks = np.asarray(kick_times)
        low_env = DY.sidechain_envelope(self.n, self.sr, kicks, depth=0.82,
                                        release=0.095 * beat_scale)
        depths = sorted({s.sidechain for s in self.plan.slots if s.sidechain > 0})
        envs = {d: DY.sidechain_envelope(self.n, self.sr, kicks, depth=0.55 * d,
                                         release=0.20 * beat_scale)
                for d in depths}
        for slot in self.plan.slots:
            if slot.sidechain <= 0:
                continue
            a = self._bar_sample(slot.start_bar)
            b = self._bar_sample(slot.end_bar)
            harm[a:b] = DY.split_sidechain(harm[a:b], self.sr, low_env[a:b],
                                           envs[slot.sidechain][a:b])
            perc[a:b] = DY.split_sidechain(perc[a:b], self.sr, low_env[a:b],
                                           envs[slot.sidechain][a:b])

        # the bass always ducks under the kick, hard
        bass_env = DY.sidechain_envelope(self.n, self.sr, np.asarray(kick_times),
                                         depth=0.85, release=0.10 * beat_scale)
        bassline = DY.apply_sidechain(bassline, bass_env)
        # carve 40-90 Hz out of the source so the kick and bass own the sub
        harm = FL.apply(harm, "highpass", self.sr, 105.0, q=0.707, order=2)

        # A separated bass stem arrives at the record's own level, which is
        # already right against the record's own vocal; a synthesised one is
        # built at full scale and has to be turned down to sit under anything.
        # HPSS's "bass" is the whole bottom of the harmonic half rather than an
        # isolated instrument, so it carries mud demucs would have given to
        # another stem and comes in lower.
        if self.bass_mode == "sub":
            bass_gain = 0.92
        elif self.source_bass_bed is not None and self.bass_mode == "source":
            bass_gain = {"demucs bass": 0.95, "hpss low band": 0.55}.get(
                self.stems.bass_name, 0.55)
        else:
            bass_gain = 0.55
        drum_bus = drums * 0.72 * self.drums_gain
        source_bus = harm * 1.35 + perc
        mix = source_bus + drum_bus + bassline * bass_gain
        self.layers.update({"source": source_bus, "harmonic": harm * 1.35,
                            "kit": drum_bus, "bass": bassline * bass_gain})
        out = DY.master(mix, self.sr, peak_db=-1.0)

        metrics = {
            "kick_count": len(kick_times),
            "peak_db": float(20 * np.log10(max(float(np.max(np.abs(out))), 1e-6))),
            "rms_db": float(20 * np.log10(max(float(np.sqrt(np.mean(out ** 2))), 1e-6))),
        }
        return out, metrics
