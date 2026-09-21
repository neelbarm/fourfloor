"""The body of ``fourfloor critic``.

It lives here rather than in ``cli.py`` so the CLI's diff stays one
subparser and one dispatch line -- three agents are editing that file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .. import ui


def run(args, c: ui.C) -> int:
    from . import report as report_mod
    from .score import Critique, critique

    target = Path(args.input)
    quiet = args.json or args.quiet
    if not quiet:
        print(ui.header(c, f"critic {target.name}"))
    progress = ui.Progress(c, quiet=quiet) if not quiet else None

    def step(name: str) -> None:
        if progress:
            progress(name)

    crit: Critique = critique(target, refs=args.refs, embed=args.embed,
                              demucs=args.demucs, windows=args.windows,
                              on_step=step)
    if args.ear == "gemini":
        crit.gemini = _ask_gemini(target, args, step)
    if progress:
        progress.close()

    if args.json:
        print(json.dumps(crit.to_dict(), indent=2))
    else:
        print(report_mod.render(crit, c))
    return 0 if crit.score >= args.pass_mark else 1


def _ask_gemini(target: Path, args, step) -> dict:
    """Ask the ear, and turn any failure into a note instead of a crash."""
    from . import gemini as g
    from .score import session_for

    session = session_for(target)
    try:
        result = g.listen(target, ref=Path(args.ref) if args.ref else None,
                          source=Path(args.source) if args.source else None,
                          session=session, model=args.model, on_step=step)
    except g.GeminiError as exc:
        print(ui.warn(ui.C(False), f"gemini unavailable: {exc}"), file=sys.stderr)
        return {"error": str(exc)}
    if not args.no_feedback:
        written = g.append_markers(target, result, session)
        if written:
            result["feedback_file"] = str(written)
    return result


def add_parser(sub) -> None:
    """Register ``critic`` on an existing subparsers object."""
    p = sub.add_parser("critic", help="score a render: does it sound like house?")
    p.add_argument("input", help="a rendered remix (mp3, wav, ...)")
    p.add_argument("--refs", default=None, metavar="DIR",
                   help="folder of real house remixes to score against "
                        "(enables the learned-similarity sub-score)")
    p.add_argument("--embed", choices=("auto", "clap", "mfcc"), default="auto",
                   help="embedding backend (default: CLAP if installed, else mfcc)")
    p.add_argument("--windows", type=int, default=4, metavar="N",
                   help="how many 10 s drop windows to compare (default 4)")
    p.add_argument("--demucs", action="store_true",
                   help="separate a real vocal stem before judging the vocal "
                        "balance (minutes on CPU)")
    p.add_argument("--ear", choices=("gemini",), default=None,
                   help="also ask a model that can hear the audio")
    p.add_argument("--ref", default=None, metavar="TRACK",
                   help="with --ear: one real remix to compare against")
    p.add_argument("--source", default=None, metavar="TRACK",
                   help="with --ear: the original song the render came from")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="with --ear: force a model instead of picking the newest")
    p.add_argument("--no-feedback", action="store_true",
                   help="do not append markers to the remix folder's feedback.json")
    p.add_argument("--pass-mark", type=float, default=0.0, metavar="SCORE",
                   help="exit non-zero below this score (default 0: always succeed)")
    p.add_argument("--json", action="store_true", help="print the critique as JSON")
    p.add_argument("-q", "--quiet", action="store_true")
