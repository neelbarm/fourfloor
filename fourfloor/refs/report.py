"""Terminal output for ``fourfloor refs``: the run, the table, the review.

Same rules as the rest of the CLI (:mod:`fourfloor.ui`): colour when stdout is
a TTY and ``NO_COLOR`` is unset, plain lines otherwise, nothing at all when
quiet. A long ingest prints as a list of links with their steps underneath,
because that is what it is -- when it stops, where it stopped is on the screen.
"""

from __future__ import annotations

from pathlib import Path

from .. import ui
from ..arrange import fmt_time

#: How each state reads, and in what colour.
STATUS = {
    "done": ("done", "green"),
    "needs_review": ("review", "yellow"),
    "standalone": ("standalone", "cyan"),
    "failed": ("failed", "red"),
    "pending": ("pending", "grey"),
    "dry-run": ("would fetch", "grey"),
}


def _paint(c: ui.C, colour: str, text: str) -> str:
    return getattr(c, colour, c.grey)(text)


class Run:
    """Prints one line per event while ``refs add`` works."""

    def __init__(self, c: ui.C, quiet: bool = False) -> None:
        self.c = c
        self.quiet = quiet
        self.links = 0

    def __call__(self, kind: str, text: str) -> None:
        if self.quiet:
            return
        c = self.c
        if kind == "link":
            self.links += 1
            print()
            print(ui.rule(c, f"{self.links}. {text[:70]}"))
        elif kind == "step":
            print(f"    {c.grey(text)}", flush=True)
        elif kind == "ok":
            print(f"  {c.green('✓')} {text}", flush=True)
        elif kind == "warn":
            print(ui.warn(c, text), flush=True)
        elif kind == "skip":
            print(f"  {c.dim('·')} {c.grey(text)}", flush=True)
        elif kind == "dry":
            print(f"  {c.cyan('?')} {text}", flush=True)
        else:
            print(ui.error(c, text), flush=True)


def _what(entry) -> str:
    """``Artist - Track (Remixer)``, from whatever the parser managed."""
    head = f"{entry.artist} - {entry.track}" if entry.artist else (entry.track or entry.title)
    return f"{head} ({entry.remixer})" if entry.remixer else head


def entry_lines(entry, c: ui.C) -> list[str]:
    """Two lines for one reference: what it is, then what is known about it."""
    label, colour = STATUS.get(entry.status, (entry.status, "grey"))
    out = [ui.kv(c, entry.slug[:26] or "?", c.bold(_what(entry)[:54]), pad=28)]
    bits: list[str] = [_paint(c, colour, label.ljust(10))]
    if entry.score:
        bits.append(f"{entry.score:.3f}")
    if entry.bpm_original and entry.bpm_remix:
        bits.append(c.grey(f"{entry.bpm_original:.0f} -> {entry.bpm_remix:.0f} BPM"))
    if entry.status == "done":
        bits.append(c.grey(f"{entry.semitones:+d} st"))
    if entry.lattice:
        bits.append(c.magenta(entry.lattice))
    if entry.treatment:
        bits.append(c.magenta(entry.treatment))
    if entry.kit:
        bits.append(c.grey(f"kit {entry.kit}"))
    if entry.error:
        bits.append(c.red(entry.error[:48]))
    out.append(ui.kv(c, "", "  ".join(bits), pad=28))
    return out


def list_report(entries: list, c: ui.C, home: Path | None = None) -> str:
    """The whole reference folder, a pair at a time."""
    lines = [ui.header(c, "references"), ""]
    if not entries:
        lines.append(ui.kv(c, "references", c.grey(
            "none yet — `fourfloor refs add <link> [<link> …]`")))
        lines.append("")
        return "\n".join(lines)
    for entry in entries:
        lines.extend(entry_lines(entry, c))
    lines.append("")
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    summary = "   ".join(f"{n} {STATUS.get(k, (k, ''))[0]}" for k, n in sorted(counts.items()))
    lines.append(ui.kv(c, "total", f"{len(entries)} links   {c.grey(summary)}"))
    if home is not None:
        lines.append(ui.kv(c, "folder", c.cyan(str(home))))
    lines.append("")
    return "\n".join(lines)


def review_report(entries: list, c: ui.C) -> str:
    """The undecided ones, with the candidates that nearly convinced it."""
    lines = [ui.header(c, "review"), ""]
    if not entries:
        lines.append(ui.kv(c, "review", c.grey("nothing is waiting on you")))
        lines.append("")
        return "\n".join(lines)
    for entry in entries:
        lines.append(ui.rule(c, entry.slug))
        lines.append("")
        lines.append(ui.kv(c, "remix", c.bold(entry.title[:60] or _what(entry))))
        lines.append(ui.kv(c, "parsed as", _what(entry)))
        lines.append(ui.kv(c, "best score", f"{c.bold(f'{entry.score:.3f}')}   "
                                            f"{c.grey(entry.method or 'vocal')}   "
                                            f"{c.grey(f'{entry.bpm_original:.0f} -> {entry.bpm_remix:.0f} BPM')}"))
        lines.append("")
        for i, cand in enumerate(entry.candidates[:4], start=1):
            lines.append(ui.kv(c, f"candidate {i}",
                               c.bold(str(cand.get("title", ""))[:56]), pad=14))
            secs = float(cand.get("duration") or 0.0)
            rank = float(cand.get("score") or 0.0)
            lines.append(ui.kv(c, "", f"{c.grey(str(cand.get('uploader', ''))[:28])}   "
                                      f"{c.cyan(fmt_time(secs))}   "
                                      f"{c.grey('rank %.2f' % rank)}",
                               pad=14))
            lines.append(ui.kv(c, "", c.cyan(str(cand.get("url", ""))), pad=14))
        lines.append("")
        lines.append(f"  {c.grey('keep it:')}  fourfloor refs accept {entry.slug}")
        lines.append(f"  {c.grey('drop it:')}  fourfloor refs reject {entry.slug}")
        lines.append("")
    return "\n".join(lines)


def table_report(result: dict, c: ui.C) -> str:
    """What the pairs taught: the four cells, and the thresholds they imply."""
    table = result.get("table") or {}
    style = result.get("style") or {}
    n_pairs = int(table.get("n_pairs") or 0)
    bpm = float(style.get("bpm") or 0.0)
    lines = ["", ui.rule(c, "what the pairs say"), ""]
    lines.append(ui.kv(c, "references", f"{c.bold(str(result.get('n_files', 0)))} files   "
                                        f"{c.grey('%d verified pairs' % n_pairs)}"))
    lines.append(ui.kv(c, "tempo", f"{c.bold('%.2f' % bpm)} BPM"))
    if style.get("tempo_ratio"):
        lines.append(ui.kv(c, "stretch", f"{float(style['tempo_ratio']):.3f}x   "
                                         f"{c.grey('original -> remix')}"))
    lines.append("")
    cells = table.get("cells") or {}
    if cells:
        lines.append(ui.kv(c, "original vocal", c.grey("what the remixer did"), pad=22))
        for key, cell in cells.items():
            lattice, treatment = (key.split("/") + ["?"])[:2]
            detail = [f"{c.bold(str(cell['n']))} pair{'s' if cell['n'] != 1 else ''}"]
            if cell.get("tempo_ratio"):
                detail.append(c.grey(f"stretch {cell['tempo_ratio']:.3f}x"))
            if cell.get("bpm"):
                detail.append(c.grey(f"{cell['bpm']:.0f} BPM"))
            if cell.get("vocal_kept") is not None:
                detail.append(c.grey(f"{cell['vocal_kept'] * 100:.0f}% of the voice kept"))
            lines.append(ui.kv(c, f"{lattice} -> {c.magenta(treatment)}",
                               "   ".join(detail), pad=22))
        lines.append("")
    if table.get("straight_lock") is not None:
        lines.append(ui.kv(c, "straight lock", f"{table['straight_lock']:.3f}   "
                                               f"{c.grey('--vocal auto plays a voice straight above this')}"))
    elif table.get("note"):
        lines.append(ui.warn(c, table["note"]))
    lines.append("")
    return "\n".join(lines)
