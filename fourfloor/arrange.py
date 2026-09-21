"""Arrangement planning: map source sections onto a house form on a bar grid.

Everything is integer bars at the target tempo, in 8-bar phrases, so every cut
lands on a downbeat by construction. The planner is deterministic and its whole
output is serialised to ``*.plan.json`` -- if a remix sounds wrong you can read
exactly which decision caused it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .analysis import Analysis, Section
from .warp import WarpMap

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


@dataclass
class Slot:
    """One arrangement slot: a span of target bars fed by a span of source audio."""

    kind: str                  # intro | build | drop | breakdown | outro
    index: int
    start_bar: int
    bars: int
    source_start: float        # seconds into the *warped* source, always a bar line
    source_bars: int           # target bars of source material behind this slot
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

    @property
    def end_bar(self) -> int:
        return self.start_bar + self.bars

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_start"] = round(self.source_start, 4)
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


def _pick(sections: list[Section], want: str, fallback: Section) -> Section:
    """Choose the best source section for a slot kind."""
    if want == "drop":
        hooks = [s for s in sections if s.label == "hook"]
        return max(hooks or sections, key=lambda s: s.energy * min(s.duration, 40.0))
    if want == "breakdown":
        quiet = [s for s in sections if s.label in ("verse", "breakdown", "intro")]
        pool = quiet or sections
        return min(pool, key=lambda s: s.energy)
    if want == "intro":
        return min(sections, key=lambda s: s.energy)
    return fallback


def _span_bars(wm: WarpMap, sec: Section) -> int:
    """How many target bars a source section becomes."""
    return wm.bars_between(sec.start, sec.end)


def _available_bars(wm: WarpMap, start_warped: float) -> int:
    """Whole target bars of source left from a warped position to the end."""
    return max(1, int((wm.duration - start_warped) // wm.bar_dur))


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def plan(analysis: Analysis, target_bpm: float, beat_multiple: float,
         form_name: str = "club", length: float | None = None,
         swing: float = 0.08, has_stems: bool = False,
         warp: WarpMap | None = None, has_kit: bool = False) -> Plan:
    """Build the arrangement.

    ``warp`` is the map the source was actually warped through. It is what makes
    a slot's ``source_start`` mean something: ``warp.snap`` turns a moment in the
    song into the warped second of the nearest source downbeat *that sits on a
    bar line*, so a slot can only ever begin where the song begins a bar. Passing
    nothing rebuilds the same map from the analysis, which is what the tests do;
    it is never a different map, because both come from ``WarpMap.from_grid``.

    Spans are contiguous and walk forward. A 32-bar drop is fed 32 bars of the
    song if the song has them, not four bars looped eight times, and the second
    drop carries on from where the first one stopped rather than replaying it.
    The breakdown is required to be somewhere else in the song entirely.

    ``has_kit`` silences the original-drum texture. That texture is there to
    give a synthesised kit something human underneath it; laid under a kit
    sampled off a record it is simply a second drummer, playing the source's
    rhythm against the record's.
    """
    keep_percussive = not (has_stems or has_kit)
    wm = warp if warp is not None else WarpMap.from_analysis(
        analysis, target_bpm, beat_multiple)
    bar_dur = wm.bar_dur
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
    hook = _pick(sections, "drop", sections[0])
    intro_src = _pick(sections, "intro", hook)

    hook_at = wm.snap(hook.start)
    hook_bars = _span_bars(wm, hook)

    # The breakdown has to be a different part of the song, and long enough to
    # be worth hearing: eight bars of real material, not a four-bar loop with a
    # filter on it. Prefer the quietest candidate that does not sit inside the
    # drop's span at all.
    def breakdown_pick() -> Section:
        pool = [s for s in sections
                if s.label in ("verse", "breakdown", "intro", "section")] or sections
        ok = [s for s in pool
              if _span_bars(wm, s) >= 8
              and not _overlaps(wm.snap(s.start), wm.snap(s.start) + _span_bars(wm, s) * bar_dur,
                                hook_at, hook_at + hook_bars * bar_dur)]
        if ok:
            return min(ok, key=lambda s: s.energy)
        other = [s for s in sections if s is not hook and _span_bars(wm, s) >= 8]
        if other:
            return min(other, key=lambda s: s.energy)
        return min(pool, key=lambda s: s.energy)

    quiet = breakdown_pick()
    quiet_at = wm.snap(quiet.start)

    def span(anchor: float, want_bars: int) -> tuple[float, int]:
        """Clamp a wanted span to what the source actually holds after ``anchor``."""
        anchor = max(0.0, min(anchor, max(0.0, wm.duration - bar_dur)))
        anchor = round(anchor / bar_dur) * bar_dur
        return anchor, max(1, min(want_bars, _available_bars(wm, anchor)))

    slots: list[Slot] = []
    bar = 0
    drop_i = 0
    drop_cursor = hook_at          # walks forward through the song across drops
    for kind, bars in shape:
        if kind == "drop":
            if drop_i > 0 and _available_bars(wm, drop_cursor) >= bars:
                src_at = drop_cursor
            else:
                src_at = hook_at
                drop_cursor = hook_at
            src_at, src_bars = span(src_at, bars)
            drop_cursor = src_at + src_bars * bar_dur
            pattern = "drop" if drop_i == 0 else "drop_var"
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=src_at, source_bars=src_bars,
                     source_label=hook.label, drum_pattern=pattern,
                     source_gain=0.82, sidechain=0.62, use_bass=True,
                     use_stabs=(drop_i == 1), impact=True, fill=True,
                     percussive_gain=0.12 if keep_percussive else 0.0,
                     note=f"hook from {fmt_time(wm.to_source(src_at))}, "
                          f"{src_bars} bars of source walked forward, full kit"
                          + (", offbeat stabs" if drop_i == 1 else ""))
            drop_i += 1
        elif kind == "intro":
            src_at, src_bars = span(wm.snap(intro_src.start), bars)
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=src_at, source_bars=src_bars,
                     source_label=intro_src.label, drum_pattern="intro",
                     source_gain=0.42, highpass=(700.0, 180.0), sidechain=0.45,
                     percussive_gain=0.0,
                     note="DJ intro: kick and hats only, source high-passed so it "
                          "does not fight the outgoing track")
        elif kind == "breakdown":
            src_at, src_bars = span(quiet_at, bars)
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=src_at, source_bars=src_bars,
                     source_label=quiet.label, drum_pattern="breakdown",
                     source_gain=0.9, lowpass=(4200.0, 14000.0), sidechain=0.0,
                     reverb_throw=True, percussive_gain=0.18 if keep_percussive else 0.0,
                     note=f"{quiet.label} from {fmt_time(quiet.start)}, a different "
                          "part of the song to the drops; no kick, filter opening")
        elif kind == "build":
            # Run into the drop out of the bars immediately before it, so the
            # build is the song's own approach rather than a preview of the
            # chorus played twice.
            want = drop_cursor if drop_i and _available_bars(wm, drop_cursor) >= bars \
                else max(0.0, hook_at - bars * bar_dur)
            src_at, src_bars = span(want, bars)
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=src_at, source_bars=src_bars,
                     source_label=hook.label, drum_pattern="build",
                     source_gain=0.6, highpass=(200.0, 1400.0), sidechain=0.5,
                     riser=True, chops=True, fill=True, percussive_gain=0.0,
                     note="riser + beat-aligned chops of the bars that lead into "
                          "the drop, high-pass climbing")
        else:  # outro
            src_at, src_bars = span(hook_at, bars)
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=src_at, source_bars=src_bars,
                     source_label=hook.label, drum_pattern="outro",
                     source_gain=0.3, highpass=(150.0, 2200.0), sidechain=0.4,
                     percussive_gain=0.0,
                     note="DJ outro: source filtered away, drums thin out for the "
                          "next mix")
        slots.append(s)
        bar += bars

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
        bar += s.bars
    if bar != p.total_bars:
        problems.append(f"slots sum to {bar} bars, plan says {p.total_bars}")
    if p.total_bars % 8 != 0:
        problems.append(f"total {p.total_bars} bars is not a whole number of 8-bar phrases")
    for s in p.slots:
        # A slot that starts mid-bar in the source puts the song's bar line
        # inside the remix's bar, which is exactly what "off beat" sounds like.
        bars = s.source_start / p.bar_dur
        if abs(bars - round(bars)) > 1e-4:
            problems.append(f"slot {s.index} ({s.kind}) starts {s.source_start:.4f}s "
                            f"into the source, which is not a bar line")
        if s.source_bars <= 0:
            problems.append(f"slot {s.index} ({s.kind}) has no source material")
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
