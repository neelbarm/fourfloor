"""Terminal presentation: colour, the branded header, phase progress, reports.

Respects ``NO_COLOR`` and falls back to plain, spinner-free output whenever
stdout is not a TTY, so piping to a file or a CI log stays readable.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass

BRAND = "fourfloor"
TAGLINE = "any song, one house remix"

_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_BLOCKS = " ▁▂▃▄▅▆▇█"


def supports_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def is_tty() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


class C:
    """ANSI helpers that collapse to identity when colour is off."""

    def __init__(self, enabled: bool) -> None:
        self.on = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def dim(self, s: str) -> str:      return self._w("2", s)
    def bold(self, s: str) -> str:     return self._w("1", s)
    def cyan(self, s: str) -> str:     return self._w("38;5;44", s)
    def magenta(self, s: str) -> str:  return self._w("38;5;207", s)
    def green(self, s: str) -> str:    return self._w("38;5;84", s)
    def yellow(self, s: str) -> str:   return self._w("38;5;221", s)
    def red(self, s: str) -> str:      return self._w("38;5;203", s)
    def grey(self, s: str) -> str:     return self._w("38;5;245", s)


def width() -> int:
    return max(48, min(shutil.get_terminal_size((80, 24)).columns, 100))


@dataclass
class Phase:
    name: str
    detail: str = ""
    elapsed: float = 0.0
    done: bool = False


class Progress:
    """A phase list with a live spinner on a TTY, plain lines otherwise."""

    def __init__(self, c: C, quiet: bool = False) -> None:
        self.c = c
        self.quiet = quiet
        self.phases: list[Phase] = []
        self._start = time.time()
        self._phase_start = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._live = is_tty() and not quiet
        self._lines = 0

    def __call__(self, name: str, detail: str = "") -> None:
        self.finish_current()
        self.phases.append(Phase(name=name, detail=detail))
        self._phase_start = time.time()
        if self._live:
            self._ensure_thread()
        elif not self.quiet:
            print(f"  {name:<10} {detail}", flush=True)

    def finish_current(self) -> None:
        if self.phases and not self.phases[-1].done:
            self.phases[-1].done = True
            self.phases[-1].elapsed = time.time() - self._phase_start

    def _ensure_thread(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

    def _spin(self) -> None:
        i = 0
        while not self._stop.wait(0.09):
            self._draw(_SPIN[i % len(_SPIN)])
            i += 1

    def _draw(self, spinner: str) -> None:
        c = self.c
        out: list[str] = []
        for p in self.phases:
            if p.done:
                out.append(f"  {c.green('✓')} {c.bold(p.name):<20} {c.grey(p.detail)}"
                           f"  {c.dim(f'{p.elapsed:.1f}s')}")
            else:
                el = time.time() - self._phase_start
                out.append(f"  {c.cyan(spinner)} {c.bold(p.name):<20} {c.grey(p.detail)}"
                           f"  {c.dim(f'{el:.1f}s')}")
        text = "\n".join(out)
        sys.stdout.write(f"\033[{self._lines}A\033[J" if self._lines else "")
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        self._lines = len(out)

    def close(self) -> None:
        self.finish_current()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self._live:
            self._draw(" ")

    @property
    def total(self) -> float:
        return time.time() - self._start


def header(c: C, subtitle: str = TAGLINE) -> str:
    """The branded banner."""
    w = width()
    bar = c.magenta("█") + c.cyan("█") + c.magenta("█") + c.cyan("█")
    line = c.dim("─" * w)
    return f"\n{bar} {c.bold(BRAND)}  {c.grey(subtitle)}\n{line}"


def rule(c: C, label: str = "") -> str:
    w = width()
    if not label:
        return c.dim("─" * w)
    pad = max(0, w - len(label) - 3)
    return f"{c.dim('──')} {c.bold(label)} {c.dim('─' * pad)}"


def kv(c: C, key: str, value: str, pad: int = 18) -> str:
    return f"  {c.grey(key.ljust(pad))} {value}"


def shade(level: float) -> str:
    """One block character for an energy level in [0, 1]."""
    i = int(round(max(0.0, min(1.0, level)) * (len(_BLOCKS) - 1)))
    return _BLOCKS[i]


def timeline(c: C, spans: list[tuple[str, float, float, float]], total: float,
             cols: int | None = None) -> list[str]:
    """Render an ASCII section timeline with energy shading.

    ``spans`` is ``(label, start, end, energy)``. Returns a list of lines: the
    shaded bar, then the labels laid out under their spans.
    """
    cols = cols or (width() - 4)
    if total <= 0:
        return []
    bar_chars: list[str] = []
    label_row = [" "] * cols
    palette = {
        "intro": c.cyan, "build": c.yellow, "drop": c.magenta,
        "breakdown": c.cyan, "outro": c.grey, "hook": c.magenta,
        "verse": c.cyan, "section": c.grey,
    }
    for label, start, end, energy in spans:
        a = int(round(start / total * cols))
        b = max(a + 1, int(round(end / total * cols)))
        b = min(b, cols)
        colour = palette.get(label, c.grey)
        bar_chars.append(colour(shade(energy) * (b - a)))
        short = label[: max(0, b - a - 1)]
        for i, ch in enumerate(short):
            if a + i < cols:
                label_row[a + i] = ch
    return ["  " + "".join(bar_chars), "  " + c.grey("".join(label_row).rstrip())]


def warn(c: C, message: str) -> str:
    return f"  {c.yellow('!')} {c.yellow(message)}"


def error(c: C, message: str) -> str:
    return f"  {c.red('x')} {message}"
