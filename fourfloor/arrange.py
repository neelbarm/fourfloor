"""Arrangement planning: map source sections onto a house form on a bar grid.

Everything is integer bars at the target tempo, in 8-bar phrases, so every cut
lands on a downbeat by construction. The planner is deterministic and its whole
output is serialised to ``*.plan.json`` -- if a remix sounds wrong you can read
exactly which decision caused it.

Three things beyond the bar grid decide whether a cut sounds like an edit or
like an arrangement, and all three are recorded in the plan:

**Phrase-aware cut points.** A downbeat is necessary but not sufficient: a
downbeat halfway through the word "again" is still a chop. Every slot's source
span is snapped, within a +/-2 bar search window, to a downbeat that is also
outside a sung phrase -- see :func:`fourfloor.analysis.structure.snap_cut`.
Each slot carries ``cut_in_reason`` / ``cut_out_reason`` saying which downbeat
was chosen and why, and ``cut_in_mid_phrase`` / ``cut_out_mid_phrase`` admitting
it when no clean option existed inside the window.

**Phrase-multiple spans.** ``source_bars`` -- which the engine uses as the loop
period when a slot is longer than its source -- is now always a whole number of
4-bar phrases, and 8-bar phrases for drops. It used to be whatever the segmenter
happened to return (9 bars, 13 bars), so a 32-bar drop restarted its source
three and a half times, each restart landing in a different place in the bar.

**Transition descriptors.** See :data:`TRANSITION_KINDS` below.

Transition schema
-----------------

Every slot carries two lists of transition descriptors, ``transition_in``
(gestures anchored to the slot's first downbeat) and ``transition_out``
(gestures that land in its final beats). Both are ordered in time and either
may be empty -- only at the very start and very end of the record, though: the
planner guarantees that every *internal* boundary has at least one descriptor
on one side of it, and :func:`validate` checks that.

Each descriptor is a plain JSON object::

    {
      "kind":     one of TRANSITION_KINDS,
      "beats":    how many beats the gesture occupies (>= 1),
      "beat":     absolute beat from the start of the arrangement where it
                  begins, so the engine never has to recompute bar maths,
      "strength": 0-1 intensity hint; 1.0 is "as written",
      "why":      human-readable reason, for the plan reader
    }

The six kinds, with the shapes measured across six commercial house remixes
(one A/B pair plus five club edits):

``drop_in``
    The kit lands on this downbeat. Measured ``kick_out_beats_before`` at
    ``build -> drop`` boundaries was 0.0 beats in all thirteen observed cases:
    the kick is exactly on the one, never early, never late. Median level jump
    +4.1 dB.
``drums_out``
    The kit stops at this downbeat for ``beats``. At ``drop -> breakdown`` the
    measured kick-out was 0.11 beats after the downbeat -- i.e. on it -- with a
    median drop of -3.2 dB. The two-beat window is where the engine should put
    the filter sweep that covers the hole.
``fill``
    A drum fill occupying the last ``beats`` of the outgoing slot. Reference
    fills ran 3-10 onsets in the final beat against a typical 0-12 per beat
    elsewhere, so this is a density instruction, not a level one. Only about one
    in five reference into-drop boundaries had a fill at all, so the planner
    emits one into the *final* drop and leaves the first drop a clean step.
``silence_beat``
    Everything mutes for ``beats``, immediately before a drop. **Off by
    default**: the reference corpus wants ``kick_in_beats_after = 0`` and no
    pre-drop gap on nine drops in ten, and a gap that is not earned reads as a
    dropout rather than as tension. Turn it on with
    ``SILENCE_BEAT_BEFORE_LAST_DROP`` or the ``silence_before_last_drop``
    argument to :func:`apply_transitions` when a track wants the effect.
``sweep_up``
    A filter/noise climb rising across ``beats`` into the next downbeat, sitting
    in the *outgoing* slot's last bars. Reference risers are short and strong --
    +1.9 to +3.4 dB per beat of 4-16 kHz over the last two bars -- not a climb
    across the whole build, and monotonic centroid ramps immediately before a
    drop ran a median of 0.13 beats. So this is eight beats, not thirty-two.
``reverse_cymbal``
    A reversed crash resolving *on* the following downbeat, so it starts
    ``beats`` before the boundary. The standard pickup into a build or a
    breakdown.

The engine is free to ignore any descriptor it has no synth for; the plan is a
score, not a command.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .analysis import Analysis, Section
from .analysis.structure import (CutPoint, VocalMap, section_hook_score,
                                 snap_cut, vocal_map)

# Form presets: (slot kind, bars, relative weight used when scaling to --length)
FORMS: dict[str, list[tuple[str, int]]] = {
    "club":  [("intro", 16), ("build", 8), ("drop", 32), ("breakdown", 16),
              ("build", 8), ("drop", 32), ("outro", 16)],
    "radio": [("intro", 8), ("build", 8), ("drop", 24), ("breakdown", 8),
              ("build", 8), ("drop", 24), ("outro", 8)],
    "tool":  [("intro", 32), ("build", 8), ("drop", 32), ("breakdown", 16),
              ("build", 8), ("drop", 32), ("outro", 32)],
}

SCALABLE = {"intro", "drop", "breakdown", "outro"}

#: The transition vocabulary, in the order a listener meets them. See the
#: module docstring for what each one asks the engine to do.
TRANSITION_KINDS = ("sweep_up", "drop_in", "drums_out", "fill",
                    "reverse_cymbal", "silence_beat")

BEATS_PER_BAR = 4

#: Emit a one-beat hole before the last drop? Off, because the reference corpus
#: puts the kick on the downbeat with no gap on nine drops in ten. Kept as a
#: switch rather than deleted: it is a real technique, just not the default one.
SILENCE_BEAT_BEFORE_LAST_DROP = False

#: How many beats a pre-drop riser occupies -- two bars, per the reference.
RISER_BEATS = 8


@dataclass
class Slot:
    """One arrangement slot: a span of target bars fed by a span of source audio."""

    kind: str                  # intro | build | drop | breakdown | outro
    index: int
    start_bar: int
    bars: int
    source_start: float        # seconds into the *warped* source
    source_bars: int           # how many source bars the span covers
    source_label: str
    drum_pattern: str
    source_gain: float
    highpass: tuple[float, float] | None = None   # (start Hz, end Hz) sweep
    lowpass: tuple[float, float] | None = None
    sidechain: float = 0.0
    use_bass: bool = False
    use_stabs: bool = False
    riser: bool = False
    impact: bool = False
    chops: bool = False
    reverb_throw: bool = False
    fill: bool = False
    percussive_gain: float = 0.0
    note: str = ""

    # --- phrase-aware editing (added; every field has a default so an older
    # --- reader of the plan JSON keeps working)
    source_end: float = 0.0            # warped seconds; end of the source span
    cut_in_reason: str = ""            # why the entry downbeat was chosen
    cut_out_reason: str = ""           # why the exit downbeat was chosen
    cut_in_mid_phrase: bool = False    # True when no clean entry was available
    cut_out_mid_phrase: bool = False
    cut_moved_bars: float = 0.0        # how far the entry moved off the boundary
    transition_in: list[dict] = field(default_factory=list)
    transition_out: list[dict] = field(default_factory=list)

    @property
    def end_bar(self) -> int:
        return self.start_bar + self.bars

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_start"] = round(self.source_start, 4)
        d["source_end"] = round(self.source_end, 4)
        d["cut_moved_bars"] = round(self.cut_moved_bars, 3)
        return d


@dataclass
class Plan:
    """A complete arrangement."""

    target_bpm: float
    bar_dur: float
    total_bars: int
    form: str
    slots: list[Slot] = field(default_factory=list)
    note: str = ""
    source: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.total_bars * self.bar_dur

    def transitions(self) -> list[dict]:
        """Every boundary in the arrangement, in time order.

        A flat view of the per-slot descriptors for anything that would rather
        walk boundaries than slots -- a DJ tool drawing markers, or the engine's
        transition pass. ``bar`` is the downbeat the boundary sits on.
        """
        out: list[dict] = []
        for i in range(1, len(self.slots)):
            prev, nxt = self.slots[i - 1], self.slots[i]
            out.append({
                "bar": nxt.start_bar,
                "beat": nxt.start_bar * BEATS_PER_BAR,
                "time": round(nxt.start_bar * self.bar_dur, 4),
                "from": prev.kind,
                "to": nxt.kind,
                "out": prev.transition_out,
                "in": nxt.transition_in,
            })
        return out

    def to_dict(self) -> dict:
        return {
            "target_bpm": round(self.target_bpm, 2),
            "bar_duration": round(self.bar_dur, 6),
            "total_bars": self.total_bars,
            "duration": round(self.duration, 3),
            "form": self.form,
            "creative_note": self.note,
            "source": self.source,
            "slots": [s.to_dict() for s in self.slots],
            "transitions": self.transitions(),
        }


def form_min_bars(form: list[tuple[str, int]]) -> int:
    """Shortest arrangement a form can express.

    Builds are fixed and every scalable slot floors at one 8-bar phrase, so a
    form cannot be squeezed below this no matter what ``--length`` asks for.
    """
    fixed = sum(b for k, b in form if k not in SCALABLE)
    return fixed + 8 * sum(1 for k, _ in form if k in SCALABLE)


def scale_form(form: list[tuple[str, int]], target_bars: int) -> list[tuple[str, int]]:
    """Scale a form to ``target_bars``, keeping every slot a multiple of 8 bars.

    Builds are never scaled -- an 8-bar build is an 8-bar build at any length --
    so the slack is distributed over the intro, drops, breakdown and outro.
    """
    fixed = sum(b for k, b in form if k not in SCALABLE)
    flex = sum(b for k, b in form if k in SCALABLE)
    room = max(target_bars - fixed, 8 * sum(1 for k, _ in form if k in SCALABLE))
    factor = room / max(flex, 1)

    out: list[tuple[str, int]] = []
    for kind, bars in form:
        if kind not in SCALABLE:
            out.append((kind, bars))
            continue
        scaled = max(8, int(round(bars * factor / 8.0)) * 8)
        out.append((kind, scaled))

    # trim or pad the largest scalable slots until the total is exact
    def total() -> int:
        return sum(b for _, b in out)

    guard = 0
    while total() != target_bars and guard < 64:
        guard += 1
        diff = target_bars - total()
        step = 8 if diff > 0 else -8
        idxs = sorted((i for i, (k, _) in enumerate(out) if k in SCALABLE),
                      key=lambda i: -out[i][1])
        moved = False
        for i in idxs:
            if step < 0 and out[i][1] <= 8:
                continue
            out[i] = (out[i][0], out[i][1] + step)
            moved = True
            break
        if not moved:
            break
    return out


# --------------------------------------------------------------------------
# Choosing what to play
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpan:
    """A snapped, phrase-multiple span of the source, and why it was chosen."""

    entry: CutPoint
    exit: CutPoint
    bars: int              # length in *source* bars, always a phrase multiple
    label: str

    @property
    def mid_phrase_cuts(self) -> int:
        return int(self.entry.mid_phrase) + int(self.exit.mid_phrase)


def source_vocal_map(analysis: Analysis, vocals: np.ndarray | None = None) -> VocalMap:
    """The vocal-activity map for a source, computed once and memoised.

    ``vocals`` is an isolated vocal (or harmonic) stem when the caller has one;
    ``fourfloor.stems.separate`` produces one either way and a Demucs ``vocals``
    stem roughly doubles the number of phrase gaps found. Without it the map is
    measured from the stereo mix using centre-channel extraction, which is good
    enough to phrase against on anything with a foreground vocal and correctly
    reports itself unusable on an instrumental.

    The result is cached on the ``Analysis`` because the planner asks for it
    once per slot and the whole point is that it is the same map every time.
    """
    cached = getattr(analysis, "_ff_vocal_map", None)
    if cached is not None and vocals is None:
        return cached
    clip = analysis.clip
    if clip is None and vocals is None:
        empty = vocal_map(np.zeros(2, dtype=np.float32), analysis.sr)
        try:
            analysis._ff_vocal_map = empty
        except AttributeError:                     # pragma: no cover
            pass
        return empty
    buf = clip.samples if clip is not None else np.zeros(2, dtype=np.float32)
    vm = vocal_map(buf, analysis.sr, vocals=vocals)
    if vocals is None:
        try:
            analysis._ff_vocal_map = vm
        except AttributeError:                     # pragma: no cover
            pass
    return vm


def choose_span(sec: Section, vmap: VocalMap, downbeats: np.ndarray,
                bar_dur: float, duration: float, *, phrase: int = 4,
                min_bars: int = 4, max_bars: int = 32) -> SourceSpan:
    """Pick the source span for a slot: a phrase-multiple run from a clean entry.

    The entry is snapped first, because it is the cut the listener notices --
    it is where the slot *starts*, with nothing before it to mask the seam.
    Then the length is chosen from the multiples of ``phrase`` that fit inside
    the track, scoring each candidate by how clean its exit lands and how close
    it stays to the length of the section we were aiming at. Choosing the
    length, rather than snapping the exit independently, is what keeps the span
    an exact number of phrases: a drop whose source is 24 bars loops seamlessly,
    one whose source is 23 bars does not.
    """
    entry = snap_cut(sec.start, downbeats, vmap, bar_dur, role="entry")
    want = max(min_bars, int(round(sec.duration / max(bar_dur, 1e-6))))
    options = [n for n in range(phrase, max_bars + 1, phrase) if n >= min_bars]
    if not options:
        options = [min_bars]

    best: tuple[float, int, CutPoint] | None = None
    for n in options:
        t_exit = entry.time + n * bar_dur
        if t_exit > duration + 0.25 * bar_dur:
            continue
        # the exit is fixed by the length, so score it in place rather than
        # letting snap_cut move it and break the phrase multiple
        cut = snap_cut(t_exit, np.asarray([t_exit]), vmap, bar_dur, role="exit")
        cost = cut.cost + 0.20 * abs(n - want) / max(want, 1)
        if best is None or cost < best[0]:
            best = (cost, n, cut)

    if best is None:                       # the source is shorter than one phrase
        n = max(1, int(duration / max(bar_dur, 1e-6)))
        t_exit = min(entry.time + n * bar_dur, duration)
        cut = snap_cut(t_exit, np.asarray([t_exit]), vmap, bar_dur, role="exit")
        return SourceSpan(entry, cut, n, sec.label)
    return SourceSpan(entry, best[2], best[1], sec.label)


def continue_span(prev: SourceSpan, vmap: VocalMap, downbeats: np.ndarray,
                  bar_dur: float, duration: float, *, phrase: int = 8,
                  min_bars: int = 8, max_bars: int = 32) -> SourceSpan:
    """The next span *forward* from where ``prev`` ended, or ``prev`` again.

    The reference remix walks its source contiguously -- 14 s to 111 s of the
    original, in order, across its whole first half -- and only then rewinds to
    replay earlier material under a new arrangement. Reusing one span for every
    drop, which is what this used to do, throws away 78% of the vocal the
    original has to offer and makes the second drop a literal repeat of the
    first.

    So each drop continues from the previous one. When fewer than ``min_bars``
    of source remain we rewind to ``prev`` rather than run off the end, and the
    span says so in its entry reason.
    """
    if duration - prev.exit.time < min_bars * bar_dur:
        rewound = SourceSpan(
            CutPoint(prev.entry.time,
                     prev.entry.reason + "; rewound -- less than "
                     f"{min_bars} bars of source left after the previous span",
                     prev.entry.mid_phrase, prev.entry.moved_bars, prev.entry.cost),
            prev.exit, prev.bars, prev.label)
        return rewound
    nxt = Section(prev.exit.time,
                  min(duration, prev.exit.time + prev.bars * bar_dur),
                  prev.label, 0, 0.5, -12.0)
    span = choose_span(nxt, vmap, downbeats, bar_dur, duration,
                       phrase=phrase, min_bars=min_bars, max_bars=max_bars)
    return SourceSpan(
        CutPoint(span.entry.time,
                 span.entry.reason + "; walks the source forward from the "
                 "previous drop rather than repeating it",
                 span.entry.mid_phrase, span.entry.moved_bars, span.entry.cost),
        span.exit, span.bars, span.label)


def choose_hook(sections: list[Section], vmap: VocalMap, bar_dur: float) -> Section:
    """The section the drops should be built from.

    Loudness alone -- which is what this used to be -- picks whichever span of
    the master is densest, and on a modern release that is as likely to be a
    bridge or an ad-lib stack as the part anyone would sing back. The score in
    :func:`~fourfloor.analysis.structure.section_hook_score` weighs vocal
    presence and repetition alongside energy, and discounts sections too short
    to fill a drop without an audible loop.
    """
    if not sections:
        raise ValueError("no sections to choose a hook from")
    max_rep = max(s.repeats for s in sections)
    return max(sections, key=lambda s: (section_hook_score(s, vmap, max_rep, bar_dur),
                                        s.duration))


def choose_breakdown(sections: list[Section], hook: Section,
                     vmap: VocalMap) -> Section:
    """Material for the breakdown: a verse, or failing that the pre-hook section.

    A breakdown wants a voice with the drums taken out from under it, so a
    verse beats the quietest thing on the record -- the quietest thing is often
    an instrumental tail with nothing to say. When the segmenter found no verse
    the section immediately before the hook is the next best guess: whatever
    leads into a chorus is written to leave space.
    """
    verses = [s for s in sections if s.label == "verse"]
    if verses:
        # among verses, the one with the most singing and the least energy
        return min(verses, key=lambda s: s.energy
                   - 0.6 * vmap.mean_activity(s.start, s.end))
    before = [s for s in sections if s.end <= hook.start + 1e-6]
    if before:
        return before[-1]
    quiet = [s for s in sections if s.label in ("breakdown", "intro")]
    return min(quiet or sections, key=lambda s: s.energy)


def choose_intro(sections: list[Section], hook: Section) -> Section:
    """Material for the DJ intro.

    The reference remix opens on the hook rather than on the original's intro,
    and fourfloor's intro slot is a filtered bed -- high-passed to 700 Hz at 0.42
    gain -- so putting the hook there teases it under the filter instead of
    spending it. ``hook`` is returned directly; the argument list keeps
    ``sections`` so the choice can grow a fallback without moving callers.
    """
    return hook if hook in sections else (sections[0] if sections else hook)


# --------------------------------------------------------------------------
# Transitions
# --------------------------------------------------------------------------


def _descriptor(kind: str, beats: int, beat: int, why: str,
                strength: float = 1.0) -> dict:
    """One transition descriptor. See the module docstring for the schema."""
    if kind not in TRANSITION_KINDS:                       # pragma: no cover
        raise ValueError(f"{kind!r} is not one of {TRANSITION_KINDS}")
    return {"kind": kind, "beats": int(max(1, beats)), "beat": int(beat),
            "strength": round(float(strength), 3), "why": why}


def apply_transitions(slots: list[Slot],
                      silence_before_last_drop: bool = SILENCE_BEAT_BEFORE_LAST_DROP
                      ) -> None:
    """Write ``transition_in`` / ``transition_out`` onto every slot, in place.

    The shapes come from the reference measurements quoted in the module
    docstring, and two of them are deliberately *less* than a house producer's
    instinct would write:

    * the riser sits in the outgoing slot's last two bars rather than climbing
      across the whole build, because measured ramps into a drop are a median
      0.13 beats long and the corpus riser spec is "short and strong";
    * a drop is a step, not a gap. Nine reference drops in ten put the kick on
      the downbeat with nothing before it, so ``silence_beat`` is off unless
      asked for, and the exit from a drop leaves its last bar intact.

    Every internal boundary gets at least one descriptor, on one side or the
    other; the first slot has nothing to transition *from* and the last nothing
    to transition *to*, so those two outer edges stay empty.
    """
    last_drop = max((i for i, s in enumerate(slots) if s.kind == "drop"), default=-1)
    for i, s in enumerate(slots):
        s.transition_in, s.transition_out = [], []
        start_beat = s.start_bar * BEATS_PER_BAR
        first, last = i == 0, i == len(slots) - 1

        # --- landing on this slot's first downbeat
        if first:
            pass
        elif s.kind == "drop":
            s.transition_in.append(_descriptor(
                "drop_in", 1, start_beat,
                "kit lands on the one as a step: every measured build->drop had "
                "the kick 0.0 beats off the downbeat, a +4.9 dB jump, no gap"))
        elif s.kind == "breakdown":
            s.transition_in.append(_descriptor(
                "drums_out", 2, start_beat,
                "kick out on the downbeat of the breakdown, two beats of filter "
                "sweep over the hole (measured kick-out 0.11 beats, -3.2 dB)"))
        elif s.kind == "outro":
            s.transition_in.append(_descriptor(
                "drums_out", 2, start_beat,
                "the drop's last bar plays complete; thin the tops for two beats "
                "into the mix-out (reference drop exit is only -2.9 dB)",
                strength=0.3))
        elif s.kind != "build":
            s.transition_in.append(_descriptor(
                "drums_out", 2, start_beat,
                "reset the groove before the section changes", strength=0.5))
        # a build needs no gesture of its own: the pickup that gets us there is
        # on the outgoing slot, and the riser belongs at the build's *end*

        # --- leaving this slot, anchored to its final beats
        if last:
            continue
        nxt = slots[i + 1]
        end_beat = s.end_bar * BEATS_PER_BAR
        if nxt.kind == "drop":
            s.transition_out.append(_descriptor(
                "sweep_up", min(RISER_BEATS, s.bars * BEATS_PER_BAR),
                end_beat - min(RISER_BEATS, s.bars * BEATS_PER_BAR),
                "short, strong riser over the last two bars: +1.9 to +3.4 dB per "
                "beat of 4-16 kHz, not a climb across the whole build",
                strength=0.8))
            if i + 1 == last_drop:
                s.transition_out.append(_descriptor(
                    "fill", 4, end_beat - BEATS_PER_BAR,
                    "fill through the last bar of the build into the final drop; "
                    "only ~1 in 5 reference into-drop boundaries has one, so the "
                    "first drop stays a clean step"))
                if silence_before_last_drop:
                    s.transition_out.append(_descriptor(
                        "silence_beat", 1, end_beat - 1,
                        "one beat of silence before the last drop (off by "
                        "default: 9 reference drops in 10 have no pre-drop gap)",
                        strength=0.6))
        elif nxt.kind in ("breakdown", "build"):
            s.transition_out.append(_descriptor(
                "reverse_cymbal", 2, end_beat - 2,
                f"reversed crash resolving on the {nxt.kind} downbeat"))
        # drop -> outro gets nothing on the way out: the reference exits a drop
        # at only -2.9 dB with the kick playing its last bar complete


def _drop_index(slots: list[Slot], i: int) -> int:
    """1-based position of ``slots[i]`` among the drops."""
    return sum(1 for s in slots[:i + 1] if s.kind == "drop")


# --------------------------------------------------------------------------
# The planner
# --------------------------------------------------------------------------


def plan(analysis: Analysis, target_bpm: float, beat_multiple: float,
         form_name: str = "club", length: float | None = None,
         swing: float = 0.08, has_stems: bool = False,
         vocals: np.ndarray | None = None) -> Plan:
    """Build the arrangement.

    ``beat_multiple`` is how many target beats one source beat occupies after
    the tempo plan, so a source section's bar count in *target* bars is
    ``source_bars * beat_multiple``.

    ``vocals`` is an optional isolated vocal stem for the *unwarped* source. It
    is only ever used to find phrase boundaries, never rendered, and the
    planner works without it -- but a Demucs ``vocals`` stem roughly doubles the
    number of phrase gaps found and therefore how often a cut can be placed in
    one.
    """
    bar_dur = 4.0 * 60.0 / target_bpm
    base = FORMS.get(form_name, FORMS["club"])
    if length:
        # Floor at what the form can actually express. Asking for less used to
        # leave `scale_form` unable to reach its target, and it returned a
        # longer arrangement without saying so.
        target_bars = max(form_min_bars(base), int(round(length / bar_dur / 8.0)) * 8)
        shape = scale_form(base, target_bars)
    else:
        shape = list(base)

    sections = analysis.sections or []
    if not sections:
        sections = [Section(0.0, analysis.duration, "hook", 0, 1.0, analysis.rms_db)]

    vmap = source_vocal_map(analysis, vocals)
    src_bar = analysis.bar_dur
    downbeats = np.asarray(analysis.grid.downbeats, dtype=float)

    hook = choose_hook(sections, vmap, src_bar)
    quiet = choose_breakdown(sections, hook, vmap)
    intro_src = choose_intro(sections, hook)

    # A drop is built from whole 8-bar phrases so a 32-bar slot can loop it
    # without the restart drifting across the bar; the breakdown works in 4-bar
    # phrases, which is as long as one usually wants.
    hook_span = choose_span(hook, vmap, downbeats, src_bar, analysis.duration,
                            phrase=8, min_bars=8, max_bars=32)
    quiet_span = choose_span(quiet, vmap, downbeats, src_bar, analysis.duration,
                             phrase=4, min_bars=4, max_bars=16)

    # Each drop walks the source forward from the one before it. `drop_spans`
    # is indexed by drop number, and the intro and the builds borrow the span
    # of whichever drop they lead into, so a build previews what is coming.
    n_drops = sum(1 for k, _ in shape if k == "drop")
    drop_spans: list[SourceSpan] = [hook_span]
    for _ in range(max(0, n_drops - 1)):
        drop_spans.append(continue_span(drop_spans[-1], vmap, downbeats, src_bar,
                                        analysis.duration, phrase=8,
                                        min_bars=8, max_bars=32))
    # which drop each slot in the form leads into (or came from, after the last)
    lead: list[int] = []
    seen = 0
    for kind, _ in shape:
        lead.append(min(seen, max(0, n_drops - 1)))
        if kind == "drop":
            seen += 1

    def warped(t: float) -> float:
        """Source time in seconds mapped into the warped (target-tempo) timeline."""
        return t / src_bar * beat_multiple * bar_dur

    def fill_cuts(s: Slot, span: SourceSpan) -> Slot:
        s.source_start = warped(span.entry.time)
        s.source_end = warped(span.exit.time)
        s.source_bars = span.bars
        s.cut_in_reason = span.entry.reason
        s.cut_out_reason = span.exit.reason
        s.cut_in_mid_phrase = span.entry.mid_phrase
        s.cut_out_mid_phrase = span.exit.mid_phrase
        s.cut_moved_bars = span.entry.moved_bars
        return s

    slots: list[Slot] = []
    bar = 0
    drop_i = 0
    for slot_i, (kind, bars) in enumerate(shape):
        ahead = drop_spans[lead[slot_i]] if drop_spans else hook_span
        if kind == "drop":
            span = ahead
            pattern = "drop" if drop_i == 0 else "drop_var"
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=0.0, source_bars=span.bars,
                     source_label=hook.label, drum_pattern=pattern,
                     source_gain=0.82, sidechain=0.62, use_bass=True,
                     use_stabs=(drop_i == 1), impact=True, fill=True,
                     percussive_gain=0.0 if has_stems else 0.12,
                     note="hook section, full kit, bass on the detected chord roots"
                          + (", offbeat stabs" if drop_i == 1 else ""))
            drop_i += 1
        elif kind == "intro":
            span = ahead
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=0.0, source_bars=span.bars,
                     source_label=intro_src.label, drum_pattern="intro",
                     source_gain=0.42, highpass=(700.0, 180.0), sidechain=0.45,
                     percussive_gain=0.0,
                     note="DJ intro: kick and hats only, source high-passed so it "
                          "does not fight the outgoing track")
        elif kind == "breakdown":
            span = quiet_span
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=0.0, source_bars=span.bars,
                     source_label=quiet.label, drum_pattern="breakdown",
                     source_gain=0.9, lowpass=(4200.0, 14000.0), sidechain=0.0,
                     reverb_throw=True, percussive_gain=0.18 if not has_stems else 0.0,
                     note="no kick, harmonic part only, filter opening across the section")
        elif kind == "build":
            # a build runs into a drop, so preview the hook material -- and use
            # exactly the hook's span, so the chops line up with what lands
            span = ahead
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=0.0, source_bars=span.bars,
                     source_label=hook.label, drum_pattern="build",
                     source_gain=0.6, highpass=(200.0, 1400.0), sidechain=0.5,
                     riser=True, chops=True, fill=True, percussive_gain=0.0,
                     note="riser + beat-aligned vocal chops, high-pass climbing "
                          "into the drop")
        else:  # outro
            span = ahead
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=0.0, source_bars=span.bars,
                     source_label=hook.label, drum_pattern="outro",
                     source_gain=0.3, highpass=(150.0, 2200.0), sidechain=0.4,
                     percussive_gain=0.0,
                     note="DJ outro: source filtered away, drums thin out for the "
                          "next mix")
        slots.append(fill_cuts(s, span))
        bar += bars

    apply_transitions(slots)

    cuts = 2 * len(slots)
    mid = sum(s.cut_in_mid_phrase + s.cut_out_mid_phrase for s in slots)
    tempo_note = (f"{analysis.grid.bpm:.2f} BPM source laid on a {target_bpm:.2f} BPM grid")
    return Plan(target_bpm=target_bpm, bar_dur=bar_dur, total_bars=bar, form=form_name,
                slots=slots, note=tempo_note,
                source={
                    "file": analysis.path.split("/")[-1],
                    "bpm": round(analysis.grid.bpm, 2),
                    "key": analysis.key.name,
                    "camelot": analysis.key.camelot,
                    "hook": hook.to_dict(),
                    "breakdown_source": quiet.to_dict(),
                    "phrasing": vmap.to_dict(),
                    "cut_quality": {
                        "cuts": cuts,
                        "mid_phrase": int(mid),
                        "clean_fraction": round(1.0 - mid / max(cuts, 1), 3),
                    },
                })


def validate(p: Plan) -> list[str]:
    """Check the invariants the arrangement math is supposed to guarantee."""
    problems: list[str] = []
    bar = 0
    for s in p.slots:
        if s.start_bar != bar:
            problems.append(f"slot {s.index} ({s.kind}) starts at bar {s.start_bar}, "
                            f"expected {bar}")
        if s.bars <= 0:
            problems.append(f"slot {s.index} ({s.kind}) has {s.bars} bars")
        if s.source_bars <= 0:
            problems.append(f"slot {s.index} ({s.kind}) reads {s.source_bars} source bars")
        for where, descs in (("in", s.transition_in), ("out", s.transition_out)):
            for d in descs:
                if d.get("kind") not in TRANSITION_KINDS:
                    problems.append(f"slot {s.index} transition_{where} has unknown "
                                    f"kind {d.get('kind')!r}")
                if int(d.get("beats", 0)) < 1:
                    problems.append(f"slot {s.index} transition_{where} "
                                    f"{d.get('kind')} spans {d.get('beats')} beats")
        bar += s.bars
    # every internal boundary needs something to cover it
    for i in range(1, len(p.slots)):
        if not p.slots[i - 1].transition_out and not p.slots[i].transition_in:
            problems.append(f"boundary at bar {p.slots[i].start_bar} "
                            f"({p.slots[i - 1].kind} -> {p.slots[i].kind}) has no "
                            f"transition descriptor")
    if bar != p.total_bars:
        problems.append(f"slots sum to {bar} bars, plan says {p.total_bars}")
    if p.total_bars % 8 != 0:
        problems.append(f"total {p.total_bars} bars is not a whole number of 8-bar phrases")
    return problems


def parse_length(text: str) -> float:
    """Parse ``4:30``, ``270`` or ``4m30s`` into seconds."""
    t = text.strip().lower().replace("m", ":").replace("s", "")
    try:
        if ":" in t:
            parts = [p for p in t.split(":") if p != ""]
            mins = float(parts[0])
            secs = float(parts[1]) if len(parts) > 1 else 0.0
            seconds = mins * 60.0 + secs
        else:
            seconds = float(t)
    except (ValueError, IndexError):
        raise ValueError(
            f"could not read --length {text!r}; use 4:30, 270 or 4m30s"
        ) from None
    if seconds <= 0:
        raise ValueError(f"--length {text!r} must be greater than zero")
    return seconds


def fmt_time(seconds: float) -> str:
    """Seconds to ``M:SS``."""
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def bars_to_samples(bars: int, bar_dur: float, sr: int) -> int:
    """Exact sample count for a whole number of bars."""
    return int(round(bars * bar_dur * sr))


def phrase_count(bars: int) -> int:
    """How many 8-bar phrases a slot spans (rounded up)."""
    return int(math.ceil(bars / 8.0))
