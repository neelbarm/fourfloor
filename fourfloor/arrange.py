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


def plan(analysis: Analysis, target_bpm: float, beat_multiple: float,
         form_name: str = "club", length: float | None = None,
         swing: float = 0.08, has_stems: bool = False) -> Plan:
    """Build the arrangement.

    ``beat_multiple`` is how many target beats one source beat occupies after
    the tempo plan, so a source section's bar count in *target* bars is
    ``source_bars * beat_multiple``.
    """
    bar_dur = 4.0 * 60.0 / target_bpm
    base = FORMS.get(form_name, FORMS["club"])
    if length:
        target_bars = max(32, int(round(length / bar_dur / 8.0)) * 8)
        shape = scale_form(base, target_bars)
    else:
        shape = list(base)

    sections = analysis.sections or []
    if not sections:
        sections = [Section(0.0, analysis.duration, "hook", 0, 1.0, analysis.rms_db)]
    hook = _pick(sections, "drop", sections[0])
    quiet = _pick(sections, "breakdown", hook)
    intro_src = _pick(sections, "intro", quiet)

    # source seconds -> warped seconds: one source bar becomes beat_multiple target bars
    src_bar = analysis.bar_dur

    def warped(t: float) -> float:
        """Source time in seconds mapped into the warped (target-tempo) timeline."""
        return t / src_bar * beat_multiple * bar_dur

    def src_bars_of(sec: Section) -> int:
        return max(1, int(round(sec.duration / src_bar)))

    slots: list[Slot] = []
    bar = 0
    drop_i = 0
    for kind, bars in shape:
        if kind == "drop":
            src, label = hook, hook.label
            pattern = "drop" if drop_i == 0 else "drop_var"
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=warped(src.start), source_bars=src_bars_of(src),
                     source_label=label, drum_pattern=pattern,
                     source_gain=0.82, sidechain=0.62, use_bass=True,
                     use_stabs=(drop_i == 1), impact=True, fill=True,
                     percussive_gain=0.0 if has_stems else 0.12,
                     note="hook section, full kit, bass on the detected chord roots"
                          + (", offbeat stabs" if drop_i == 1 else ""))
            drop_i += 1
        elif kind == "intro":
            src = intro_src
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=warped(src.start), source_bars=src_bars_of(src),
                     source_label=src.label, drum_pattern="intro",
                     source_gain=0.42, highpass=(700.0, 180.0), sidechain=0.45,
                     percussive_gain=0.0,
                     note="DJ intro: kick and hats only, source high-passed so it "
                          "does not fight the outgoing track")
        elif kind == "breakdown":
            src = quiet
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=warped(src.start), source_bars=src_bars_of(src),
                     source_label=src.label, drum_pattern="breakdown",
                     source_gain=0.9, lowpass=(4200.0, 14000.0), sidechain=0.0,
                     reverb_throw=True, percussive_gain=0.18 if not has_stems else 0.0,
                     note="no kick, harmonic part only, filter opening across the section")
        elif kind == "build":
            prev = slots[-1] if slots else None
            src = quiet if (prev is None or prev.kind != "breakdown") else quiet
            # a build runs into a drop, so preview the hook material
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=warped(hook.start), source_bars=src_bars_of(hook),
                     source_label=hook.label, drum_pattern="build",
                     source_gain=0.6, highpass=(200.0, 1400.0), sidechain=0.5,
                     riser=True, chops=True, fill=True, percussive_gain=0.0,
                     note="riser + beat-aligned vocal chops, high-pass climbing "
                          "into the drop")
        else:  # outro
            src = hook
            s = Slot(kind=kind, index=len(slots), start_bar=bar, bars=bars,
                     source_start=warped(src.start), source_bars=src_bars_of(src),
                     source_label=src.label, drum_pattern="outro",
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
    return problems


def parse_length(text: str) -> float:
    """Parse ``4:30``, ``270`` or ``4m30s`` into seconds."""
    t = text.strip().lower().replace("m", ":").replace("s", "")
    if ":" in t:
        parts = [p for p in t.split(":") if p != ""]
        mins = float(parts[0])
        secs = float(parts[1]) if len(parts) > 1 else 0.0
        return mins * 60.0 + secs
    return float(t)


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
