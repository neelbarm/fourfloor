"""Terminal rendering for a critique: the score, the bars, the verdict."""

from __future__ import annotations

from .. import ui

_BAR = "█"
_TRACK = "·"


def _tint(c: ui.C, score: float):
    if score >= 75:
        return c.green
    if score >= 50:
        return c.yellow
    return c.red


def bar(c: ui.C, score: float, width: int = 24) -> str:
    filled = int(round(max(0.0, min(100.0, score)) / 100.0 * width))
    tint = _tint(c, score)
    return tint(_BAR * filled) + c.dim(_TRACK * (width - filled))


def _wrap(text: str, width: int, indent: str) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(indent + cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(indent + cur)
    return lines


def render(crit, c: ui.C) -> str:
    """The full report for one critique."""
    w = ui.width()
    name = crit.path.rsplit("/", 1)[-1]
    tint = _tint(c, crit.score)
    lines = [
        ui.rule(c, name),
        "",
        f"  {tint(c.bold(f'{crit.score:5.1f}'))} {c.grey('/ 100')}   "
        f"{bar(c, crit.score, 34)}",
        "",
    ]
    lines += _wrap(crit.verdict, w - 4, "  ")
    lines.append("")
    pad = max(len(s.label) for s in crit.subs) + 1
    for s in sorted(crit.subs, key=lambda s: -s.weight):
        head = (f"  {c.grey(s.label.ljust(pad))} {bar(c, s.score)} "
                f"{_tint(c, s.score)(f'{s.score:5.1f}')} {c.dim(f'x{s.weight:.2f}')}")
        lines.append(head)
        lines += _wrap(s.detail, w - 6 - pad, " " * (pad + 4))
    lines.append("")
    m = crit.measured
    lines.append(ui.kv(c, "tempo", f"{m.bpm:.1f} BPM  ({m.duration / 60:.1f} min)"))
    lines.append(ui.kv(c, "on the grid", f"{m.on_grid * 100:.0f}% of onsets within 20 ms"))
    lines.append(ui.kv(c, "level", f"{m.rms_db:.1f} dBFS RMS, {m.crest_db:.1f} dB crest"))
    lines.append(ui.kv(c, "embedding", crit.backend))
    if crit.gated:
        lines.append("")
        lines.append(ui.warn(c, "score capped by the groove sub-score"))
    for note in crit.notes:
        lines.append(ui.warn(c, note))
    if crit.gemini:
        lines.append("")
        lines += _gemini_block(crit.gemini, c, w)
    lines.append("")
    lines.append(c.dim(f"  {crit.elapsed:.1f}s"))
    return "\n".join(lines)


_SEV = {1: "·", 2: "!", 3: "!!"}


def _gemini_block(g: dict, c: ui.C, w: int) -> list[str]:
    if g.get("error"):
        return [ui.warn(c, f"gemini: {g['error']}")]
    score = float(g.get("overall_score", 0))
    tint = _tint(c, score)
    out = [ui.rule(c, f"gemini ({g.get('model', '?')})"), ""]
    out.append(f"  {tint(c.bold(f'{score:5.1f}'))} {c.grey('/ 100')}   {bar(c, score, 34)}")
    out.append("")
    out += _wrap(g.get("verdict", ""), w - 4, "  ")
    issues = g.get("issues") or []
    if issues:
        out.append("")
        for i in issues[:14]:
            sev = _SEV.get(int(i.get("severity", 2)), "!")
            mark = c.red(sev) if i.get("severity", 2) >= 3 else (
                c.yellow(sev) if i.get("category") != "good" else c.green("+"))
            t = float(i.get("time_sec", 0))
            out.append(f"  {mark} {c.grey(f'{int(t) // 60}:{int(t) % 60:02d}')} "
                       f"{c.dim(i.get('category', ''))}  {i.get('note', '')}")
    sections = g.get("sections") or {}
    if sections:
        out.append("")
        for k, v in list(sections.items())[:10]:
            out += _wrap(f"{k}: {v}", w - 6, "    ")
    vs = g.get("vs_reference") or {}
    if vs:
        out.append("")
        out.append(c.grey("  the reference does better:"))
        for point in (vs.get("reference_does_better") or [])[:6]:
            out += _wrap(f"- {point}", w - 6, "    ")
        if vs.get("biggest_gap"):
            out.append("")
            out += _wrap(f"biggest gap: {vs['biggest_gap']}", w - 4, "  ")
    return out


def table(crits, c: ui.C) -> str:
    """A one-line-per-file comparison, used by the calibration report."""
    keys = ["similarity", "groove", "clarity", "clicks", "vocal", "loudness"]
    head = f"  {'file'.ljust(26)} {'total':>6}  " + "  ".join(k[:5].rjust(6) for k in keys)
    lines = [head, "  " + c.dim("-" * (len(head) - 2))]
    for crit in crits:
        name = crit.path.rsplit("/", 1)[-1][:26]
        cells = []
        for k in keys:
            s = crit.sub(k)
            cells.append(f"{s.score:6.1f}" if s else "     -")
        lines.append(f"  {name.ljust(26)} {crit.score:6.1f}  " + "  ".join(cells))
    return "\n".join(lines)
