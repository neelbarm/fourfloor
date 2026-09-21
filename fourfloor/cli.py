"""Command line interface: fetch, remix, inspect, learn, preview."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ui
from .arrange import FORMS, fmt_time
from .critic import command as critic_command
from .style import Style

VERSION = "0.1.0"


class CliError(ValueError):
    """A command line the parser refused.

    argparse's default is to print usage and call ``sys.exit``, which is right
    for a terminal and useless to the web app -- it needs the message so it can
    answer 400 with it. Raising instead means the option rules live in exactly
    one place: whatever ``fourfloor remix`` refuses, ``fourfloor serve``
    refuses too, with the same wording.
    """


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):        # noqa: D102 - argparse hook
        raise CliError(message)


def _parser() -> argparse.ArgumentParser:
    # subparsers inherit this class, so every subcommand raises too
    p = _Parser(
        prog="fourfloor",
        description="Turn any song into a house remix, with a session file a DJ app can sync to.",
    )
    p.add_argument("--version", action="version", version=f"fourfloor {VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("remix", help="build a house remix of a song")
    r.add_argument("input", nargs="?", help="source audio file (mp3, m4a, wav, flac…)")
    r.add_argument("--url", metavar="LINK", default=None,
                   help="fetch the source from a YouTube or SoundCloud link instead")
    r.add_argument("-o", "--output", help="output path (.mp3 or .wav)")
    r.add_argument("--bpm", type=float, help="target tempo (default: learned or suggested)")
    r.add_argument("--key", help="target key: 'Am', 'F#', '8A' or 'auto'")
    r.add_argument("--compatible-with", metavar="TRACK",
                   help="shift into a key that mixes with this track")
    r.add_argument("--style", metavar="STYLE.JSON", help="style profile from `fourfloor learn`")
    r.add_argument("--stems", choices=("hpss", "demucs"), default="hpss",
                   help="separation engine (demucs needs the [stems] extra)")
    r.add_argument("--length", default=None, help="target length, e.g. 4:30")
    r.add_argument("--form", choices=tuple(FORMS), default="club", help="arrangement preset")
    r.add_argument("--swing", type=float, default=None, help="hat swing, 0 to 0.66")
    r.add_argument("--kit", default=None, metavar="NAME",
                   help="drum kit built with `fourfloor kit build` "
                        "(default: the most recent one; 'none' for the synth kit)")
    r.add_argument("--bass", choices=("source", "synth"), default="source",
                   help="use the song's own bass stem (default) or synthesise one")
    r.add_argument("--no-kick-reinforce", action="store_true",
                   help="do not put a synth kick under a sampled loop's kicks")
    r.add_argument("--producer", action="store_true",
                   help="ask Claude to plan the arrangement (needs ANTHROPIC_API_KEY)")
    r.add_argument("--seed", type=int, default=0, help="randomisation seed")
    r.add_argument("--no-wav", action="store_true", help="write only the mp3")
    r.add_argument("--preview", action="store_true", help="also write preview.html")
    r.add_argument("--json", action="store_true", help="print the session JSON instead of a report")
    r.add_argument("-q", "--quiet", action="store_true")

    k = sub.add_parser("kit", help="build drum kits from real house records")
    ksub = k.add_subparsers(dest="kit_command", required=True)
    kb = ksub.add_parser("build", help="sample a kit from a house remix")
    kb.add_argument("input", help="a house record to take the drums from")
    kb.add_argument("--name", default=None, help="what to call the kit")
    kb.add_argument("--bars", type=int, default=8, help="loop length in bars (default 8)")
    kb.add_argument("--json", action="store_true")
    kl = ksub.add_parser("list", help="show the kits you have built")
    kl.add_argument("--json", action="store_true")

    i = sub.add_parser("inspect", help="analyse a track and print a report")
    i.add_argument("input", nargs="?")
    i.add_argument("--url", metavar="LINK", default=None,
                   help="fetch a YouTube or SoundCloud link into a temp folder first")
    i.add_argument("--json", action="store_true")

    f = sub.add_parser("fetch", help="download a track from a link (yt-dlp)")
    f.add_argument("url", nargs="?", help="a track, playlist or SoundCloud set")
    f.add_argument("--to", default=None, metavar="DEST",
                   help="'remixes', 'pairs' or a folder "
                        "(default: ~/Music/house-refs/remixes)")
    f.add_argument("--name", default=None, metavar="BODY",
                   help="name the file yourself; the body of a pair's name")
    f.add_argument("--as", dest="kind", choices=("original", "remix"), default=None,
                   help="save this link as one half of a learning pair")
    f.add_argument("--original", default=None, metavar="LINK",
                   help="with --remix and --name: fetch both halves of a pair")
    f.add_argument("--remix", dest="remix_url", default=None, metavar="LINK")
    f.add_argument("--json", action="store_true")
    f.add_argument("-q", "--quiet", action="store_true")

    l = sub.add_parser("learn", help="derive a style profile from a folder of remixes")
    l.add_argument("folder")
    l.add_argument("-o", "--output", default="style.json")
    l.add_argument("--anonymous", action="store_true",
                   help="omit per-file rows; save aggregate numbers only")
    l.add_argument("--json", action="store_true")

    v = sub.add_parser("preview", help="write a preview.html next to a remix")
    v.add_argument("input", help="a remix mp3/wav that has a .session.json beside it")
    v.add_argument("-o", "--output", default=None)

    e = sub.add_parser("export", help="write DJ files for a remix or a folder of them")
    e.add_argument("input", help="a remix mp3, or a folder of them, "
                                 "each with its .session.json beside it")
    e.add_argument("--set", dest="set_name", default=None, metavar="NAME",
                   help="what to call the Rekordbox playlist (default: the folder)")
    e.add_argument("--format", dest="formats", action="append", metavar="FMT",
                   help="rekordbox, serato, csv, tags or all; repeat or comma-separate "
                        "(default: rekordbox,csv)")
    e.add_argument("--out", default=None, metavar="DIR",
                   help="where rekordbox.xml and cues.csv go "
                        "(default: beside the remixes)")
    e.add_argument("--artist", default="fourfloor",
                   help="the Artist tag to write (default: fourfloor)")
    e.add_argument("--suffix", action="store_true",
                   help="append ' (fourfloor house remix)' to every title")
    e.add_argument("--no-verify", action="store_true",
                   help="skip the ffprobe check that tagging left the audio alone")
    e.add_argument("--json", action="store_true")

    b = sub.add_parser("batch", help="remix a folder into one set at one tempo")
    b.add_argument("folder", help="folder of original tracks")
    b.add_argument("-o", "--out", required=True, metavar="DIR",
                   help="where the remixes and the set files go")
    b.add_argument("--bpm", type=float, required=True,
                   help="one tempo for the whole set, e.g. 126")
    kg = b.add_mutually_exclusive_group()
    kg.add_argument("--key-lock", action="store_true", dest="key_lock",
                    help="keep every track in its own key (the default)")
    kg.add_argument("--key", default=None, metavar="STRATEGY",
                    help="'auto': shift each track up to ±2 semitones onto a key "
                         "that mixes with the one before it")
    b.add_argument("--set", dest="set_name", default=None, metavar="NAME",
                   help="what to call the set (default: the source folder's name)")
    b.add_argument("--stems", choices=("hpss", "demucs"), default="hpss")
    b.add_argument("--kit", default=None, metavar="NAME")
    b.add_argument("--form", choices=tuple(FORMS), default="club")
    b.add_argument("--length", default=None, help="target length per track, e.g. 4:30")
    b.add_argument("--swing", type=float, default=None)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--wav", action="store_true", help="also write a wav per track")
    b.add_argument("--artist", default="fourfloor")
    b.add_argument("--jobs", type=int, default=1,
                   help="render this many tracks at once (default 1)")
    b.add_argument("--resume", action="store_true",
                   help="skip tracks that already have an mp3 and a session file")
    b.add_argument("--json", action="store_true",
                   help="print set.json instead of the report")
    b.add_argument("-q", "--quiet", action="store_true")

    critic_command.add_parser(sub)

    s = sub.add_parser("serve", help="run the local web app: drop a song in a browser")
    s.add_argument("--port", type=int, default=4444, help="port to listen on (default 4444)")
    s.add_argument("--open", action="store_true", dest="open_browser",
                   help="open the app in your browser once it is up")
    s.add_argument("--home", default=None, metavar="DIR",
                   help="where uploads and remixes live (default ~/.fourfloor)")
    return p


def parse_remix_args(argv: list[str]):
    """Parse a ``remix`` command line, raising :class:`CliError` on a bad flag.

    This is the seam the web app validates through: it renders its form state
    as the flags a person would have typed, and gets argparse's own answer.
    """
    return _parser().parse_args(argv)


def remix_options(args):
    """Build :class:`~fourfloor.remix.RemixOptions` from parsed ``remix`` args."""
    from .remix import RemixOptions

    return RemixOptions(
        target_bpm=args.bpm, key=args.key, compatible_with=args.compatible_with,
        stems=args.stems, form=args.form, length=args.length, swing=args.swing,
        producer=args.producer, seed=args.seed, wav=not args.no_wav,
        kit=args.kit, bass=args.bass,
        kick_reinforce=not getattr(args, "no_kick_reinforce", False),
    )


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

def _inspect_report(a, c: ui.C) -> str:
    from .analysis import suggest_house_tempo
    from .dsp.pitch import plan_tempo

    lines = [ui.header(c, "inspect"), ""]
    lines.append(ui.kv(c, "file", c.bold(Path(a.path).name)))
    lines.append(ui.kv(c, "length", fmt_time(a.duration)))
    lines.append("")
    lines.append(ui.kv(c, "tempo", f"{c.bold(f'{a.grid.bpm:.2f}')} BPM"
                                   f"   {c.grey(f'{len(a.grid.beats)} beats tracked')}"))
    lines.append(ui.kv(c, "beat grid offset", f"{a.grid.beats[0]:.3f} s"
                       if len(a.grid.beats) else "n/a"))
    lines.append(ui.kv(c, "first downbeat", f"{a.grid.first_downbeat:.3f} s"
                       f"   {c.grey(f'confidence {a.grid.downbeat_confidence:.2f}')}"))
    lines.append(ui.kv(c, "key", f"{c.bold(a.key.name)}  {c.magenta(a.key.camelot)}"
                                 f"   {c.grey(f'confidence {a.key.confidence:.2f}')}"))
    lines.append(ui.kv(c, "tuning", f"{a.key.tuning * 100:+.0f} cents"))
    lines.append(ui.kv(c, "loudness", f"{a.rms_db:.1f} dB RMS   {a.peak_db:.1f} dB peak"))
    lines.append("")

    target = suggest_house_tempo(a.grid.bpm)
    tp = plan_tempo(a.grid.bpm, target)
    lines.append(ui.rule(c, "house target"))
    lines.append(ui.kv(c, "suggested tempo", f"{c.bold(f'{target:.2f}')} BPM"
                                             f"   {c.grey(tp.interpretation)}"
                                             f"   {c.grey(f'stretch x{tp.ratio:.3f}')}"))
    lines.append(ui.kv(c, "suggested key", f"{a.key.name} {a.key.camelot} "
                                           f"{c.grey('(no shift needed)')}"))
    if tp.warning:
        lines.append(ui.warn(c, tp.warning))
    lines.append("")

    lines.append(ui.rule(c, "structure"))
    spans = [(s.label, s.start, s.end, s.energy) for s in a.sections]
    lines.extend(ui.timeline(c, spans, a.duration))
    lines.append("")
    for s in a.sections:
        lines.append(ui.kv(
            c, f"{fmt_time(s.start)}-{fmt_time(s.end)}",
            f"{c.bold(s.label.ljust(10))} {ui.shade(s.energy) * 6}  "
            f"{c.grey(f'{s.rms_db:5.1f} dB   x{s.repeats}')}", pad=14))
    lines.append("")
    chords = [ch["name"] for ch in a.chords[:16]]
    if chords:
        lines.append(ui.kv(c, "chords (bars 1-16)", c.grey(" ".join(chords))))
    lines.append("")
    return "\n".join(lines)


def _remix_report(res, c: ui.C, elapsed: float) -> str:
    a = res.analysis
    p = res.plan
    lines = ["", ui.rule(c, "result"), ""]
    lines.append(ui.kv(c, "source", f"{c.bold(f'{a.grid.bpm:.2f}')} BPM   "
                                    f"{a.key.name} {a.key.camelot}   "
                                    f"{fmt_time(a.duration)}"))
    shift = f"{res.semitones:+d} st" if res.semitones else "unchanged"
    lines.append(ui.kv(c, "remix", f"{c.bold(f'{p.target_bpm:.2f}')} BPM   "
                                   f"{c.magenta(res.target_key.name)} "
                                   f"{c.magenta(res.target_key.camelot)}   "
                                   f"{fmt_time(p.duration)}   {c.grey(shift)}"))
    lines.append(ui.kv(c, "tempo mapping", f"{res.tempo_plan.interpretation}, "
                                           f"stretch x{res.tempo_plan.ratio:.3f}"))
    lines.append(ui.kv(c, "drums", (f"{c.bold(res.kit_name)} "
                                    f"{c.grey('(sampled from a real record)')}")
                       if res.kit_name else c.grey("synthesised kit "
                                                   "(build one: fourfloor kit build)")))
    lines.append(ui.kv(c, "bass", res.bass_source))
    lines.append(ui.kv(c, "levels", f"{res.metrics['peak_db']:.2f} dB peak   "
                                    f"{res.metrics['rms_db']:.2f} dB RMS   "
                                    f"{res.metrics['kick_count']} kicks"))
    lines.append("")
    lines.append(ui.rule(c, "arrangement"))
    spans = [(s.kind, s.start_bar * p.bar_dur, s.end_bar * p.bar_dur,
              {"intro": 0.45, "build": 0.7, "drop": 1.0,
               "breakdown": 0.35, "outro": 0.4}.get(s.kind, 0.5))
             for s in p.slots]
    lines.extend(ui.timeline(c, spans, p.duration))
    lines.append("")
    for s in p.slots:
        lines.append(ui.kv(
            c, f"{fmt_time(s.start_bar * p.bar_dur)}  bar {s.start_bar:>3}",
            f"{c.bold(s.kind.ljust(10))} {c.grey(f'{s.bars:>2} bars')}  "
            f"{c.grey('<- ' + s.source_label)}", pad=18))
    lines.append("")
    for w in res.warnings:
        lines.append(ui.warn(c, w))
    if res.warnings:
        lines.append("")
    lines.append(ui.rule(c, "files"))
    for label, path in res.paths.items():
        lines.append(ui.kv(c, label, c.cyan(str(path))))
    lines.append("")
    lines.append(f"  {c.green('done')} {c.grey(f'in {elapsed:.1f}s')}")
    lines.append("")
    return "\n".join(lines)


def _style_report(st: Style, c: ui.C) -> str:
    lines = [ui.header(c, "learn"), ""]
    lines.append(ui.kv(c, "references", f"{c.bold(str(st.n_refs))} tracks"))
    lines.append("")
    lines.append(ui.kv(c, "tempo", f"{c.bold(f'{st.bpm:.2f}')} BPM   "
                                   f"{c.grey(f'spread ±{st.bpm_spread:.2f}')}"))
    lines.append(ui.kv(c, "swing", f"{st.swing:+.3f}   "
                                   f"{c.grey('of a 16th on the offbeats')}"))
    lines.append(ui.kv(c, "dj intro", f"{st.intro_bars} bars before the kick locks"))
    lines.append(ui.kv(c, "breakdown", f"{st.breakdown_ratio * 100:.0f}% of refs, "
                                       f"~{st.breakdown_bars} bars"))
    lines.append(ui.kv(c, "kick density", f"{st.kick_density:.2f} per beat"))
    lines.append(ui.kv(c, "brightness", f"{st.brightness_hz:.0f} Hz centroid   "
                                        f"{c.grey(f'tilt {st.spectral_tilt:.2f} dB/decade')}"))
    lines.append(ui.kv(c, "loudness", f"{st.rms_db:.1f} dB RMS   {st.peak_db:.1f} dB peak"))
    lines.append(ui.kv(c, "typical length", fmt_time(st.length)))
    lines.append("")
    if st.per_track:
        lines.append(ui.rule(c, "per reference"))
        for r in st.per_track:
            lines.append(ui.kv(c, r.get("file", "?")[:28],
                               f"{r['bpm']:6.2f} BPM  {r['key']:>4} {r['camelot']:>3}  "
                               f"swing {r['swing']:+.3f}  intro {r['intro_bars']:>3} bars  "
                               f"{r['rms_db']:6.1f} dB", pad=30))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

class FetchLine:
    """One rewriting progress line: a bar, a percent and what is downloading.

    ``ui.Progress`` draws a list of named phases, which is right for a remix and
    wrong for a download -- there is one phase and it has a percentage. This
    keeps the same palette and the same rules: live redraw on a TTY, one plain
    line per quarter anywhere else, nothing at all when quiet.
    """

    WIDTH = 26

    def __init__(self, c: ui.C, quiet: bool = False) -> None:
        self.c = c
        self.quiet = quiet
        self.live = ui.is_tty() and not quiet
        self._step = -1
        self._last = 0.0
        self._drawn = False

    def __call__(self, pct: float, note: str = "") -> None:
        if self.quiet:
            return
        c = self.c
        pct = max(0.0, min(100.0, float(pct)))
        if self.live:
            filled = int(round(pct / 100 * self.WIDTH))
            bar = c.magenta("━" * filled) + c.dim("━" * (self.WIDTH - filled))
            label = note if len(note) <= 40 else note[:39] + "…"
            sys.stdout.write(f"\r  {bar} {c.bold(f'{pct:5.1f}%')}  {c.grey(label)}"
                             f"\033[K")
            sys.stdout.flush()
            self._drawn = True
            return
        if pct < self._last - 1:                  # the next track of a playlist
            self._step = -1
        self._last = pct
        step = int(pct // 25)
        if step > self._step:
            self._step = step
            print(f"  {pct:5.1f}%  {note}", flush=True)

    def close(self) -> None:
        if self._drawn:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            self._drawn = False


def _fetch_report(got: list, c: ui.C, dest: Path, elapsed: float) -> str:
    from . import fetch as fetch_mod
    from .arrange import fmt_time as _fmt

    lines = ["", ui.rule(c, "fetched"), ""]
    for f in got:
        lines.append(ui.kv(c, f.path.name[:28], c.bold((f.title or f.path.stem)[:52]),
                           pad=30))
        lines.append(ui.kv(c, "", f"{c.grey(f.site)}   "
                                  f"{c.grey(f.uploader or 'unknown artist')}   "
                                  f"{c.cyan(_fmt(f.duration))}   "
                                  f"{c.grey(fetch_mod.fmt_bytes(f.bytes))}", pad=30))
    lines.append("")
    lines.append(ui.kv(c, "folder", c.cyan(str(dest))))
    lines.append(ui.kv(c, "total", f"{len(got)} file{'s' if len(got) != 1 else ''}   "
                                   f"{c.grey(fetch_mod.fmt_bytes(fetch_mod.total_bytes(got)))}"))
    lines.append("")
    lines.append(f"  {c.green('done')} {c.grey(f'in {elapsed:.1f}s')}")
    lines.append("")
    return "\n".join(lines)


def cmd_fetch(args, c: ui.C) -> int:
    import time as _time

    from . import fetch as fetch_mod

    pair = bool(args.original or args.remix_url)
    if pair and not (args.original and args.remix_url and args.name):
        raise ValueError("a pair needs --original <url> --remix <url> --name <body>")
    if not pair and not args.url:
        raise ValueError("give me a link: `fourfloor fetch <url>`")
    if args.url and pair:
        raise ValueError("fetch either one link or a --original/--remix pair, not both")
    if args.kind and not args.name:
        raise ValueError("--as original/remix needs --name <body> to go with it")

    where = args.to if args.to is not None else ("pairs" if pair or args.kind
                                                 else "remixes")
    dest = fetch_mod.resolve_dest(where)
    for link in ([args.original, args.remix_url] if pair else [args.url]):
        fetch_mod.check_url(link)                 # refuse a bad link before the banner
    quiet = args.quiet or args.json
    if not quiet:
        print(ui.header(c, "fetch"))
        print()
        print(ui.kv(c, "into", c.cyan(str(dest))))
        print()

    line = FetchLine(c, quiet=quiet)
    t0 = _time.time()
    try:
        if pair:
            got = fetch_mod.fetch_pair(args.original, args.remix_url, args.name,
                                       dest_dir=dest, progress=line)
        else:
            got = fetch_mod.fetch_all(args.url, dest, name=args.name, kind=args.kind,
                                      progress=line)
    finally:
        line.close()

    if args.json:
        print(json.dumps({"dest": str(dest), "files": [f.to_dict() for f in got]},
                         indent=2))
    elif not quiet:
        print(_fetch_report(got, c, dest, _time.time() - t0))
    return 0


def _from_link(url: str, c: ui.C, quiet: bool = False):
    """Fetch ``url`` into a temp folder for a one-shot inspect or remix."""
    from . import fetch as fetch_mod

    line = FetchLine(c, quiet=quiet)
    if not quiet:
        print(ui.kv(c, "link", c.cyan(url)))
    try:
        got, cleanup = fetch_mod.fetch_to_temp(url, progress=line)
    finally:
        line.close()
    if not quiet:
        from .arrange import fmt_time as _fmt
        print(ui.kv(c, "fetched", f"{c.bold(got.title or got.path.stem)}   "
                                  f"{c.grey(got.uploader or got.site)}   "
                                  f"{c.cyan(_fmt(got.duration))}"))
        print()
    return got, cleanup


def cmd_inspect(args, c: ui.C) -> int:
    from .analysis import analyze

    if not args.input and not args.url:
        raise ValueError("give me a file, or `fourfloor inspect --url <link>`")
    cleanup = None
    src = args.input
    if args.url:
        got, cleanup = _from_link(args.url, c, quiet=args.json)
        src = got.path
    try:
        a = analyze(src)
    finally:
        if cleanup is not None:
            cleanup()
    if args.json:
        print(json.dumps(a.to_dict(), indent=2))
    else:
        print(_inspect_report(a, c))
    return 0


def cmd_remix(args, c: ui.C) -> int:
    from .remix import remix

    if not args.input and not args.url:
        raise ValueError("give me a file, or `fourfloor remix --url <link>`")
    quiet = args.quiet or args.json
    cleanup = None
    if args.url:
        if not quiet:
            print(ui.header(c, "fetch"))
            print()
        got, cleanup = _from_link(args.url, c, quiet=quiet)
        src = got.path
        # the temp folder goes away, so an unnamed output lands where you are
        out = Path(args.output) if args.output else Path.cwd() / f"{src.stem}.house.mp3"
    else:
        src = Path(args.input)
        out = Path(args.output) if args.output else src.with_suffix("").with_name(
            src.stem + ".house.mp3")
    style = Style.load(args.style) if args.style else None

    opts = remix_options(args)
    if not quiet:
        print(ui.header(c, f"remixing {src.name}"))
        print()
    progress = ui.Progress(c, quiet=quiet)
    try:
        res = remix(src, out, opts, style=style, progress=progress)
    finally:
        progress.close()
        if cleanup is not None:
            cleanup()

    if args.preview:
        from .preview import write_preview
        res.paths["preview"] = write_preview(res.paths.get("mp3", out), res.session)

    if args.json:
        print(json.dumps(res.session, indent=2))
    elif not quiet:
        print(_remix_report(res, c, progress.total))
    return 0


def cmd_kit(args, c: ui.C) -> int:
    from . import kit as kit_mod

    if args.kit_command == "list":
        rows = kit_mod.catalogue()
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        if not rows:
            print(ui.kv(c, "kits", c.grey("none yet — "
                                          "`fourfloor kit build <a house remix.mp3>`")))
            return 0
        print(ui.header(c, "kits"))
        print()
        for i, r in enumerate(rows):
            tag = c.grey("  (default)") if i == 0 else ""
            print(ui.kv(c, r["name"], f"{c.grey(r.get('source', '?')[:44])}   "
                                      f"{r.get('source_bpm', 0):.2f} BPM   "
                                      f"{r.get('bars', 8)} bars{tag}"))
        print()
        return 0

    if not args.json:
        print(ui.header(c, f"kit from {Path(args.input).name}"))
        print()
    progress = ui.Progress(c, quiet=args.json)
    try:
        k = kit_mod.build(args.input, name=args.name, bars=args.bars, progress=progress)
    finally:
        progress.close()
    if args.json:
        print(json.dumps(k.to_dict(), indent=2))
        return 0
    print()
    print(ui.rule(c, "kit"))
    print()
    print(ui.kv(c, "name", c.bold(k.name)))
    print(ui.kv(c, "from", f"{c.grey(k.source)}   {k.source_bpm:.2f} BPM"))
    print(ui.kv(c, "loop", f"{k.bars} bars at {kit_mod.CANONICAL_BPM:.0f} BPM   "
                           f"{c.grey(f'{len(k.loop) / k.sr:.2f}s')}"))
    print(ui.kv(c, "kicks", f"{len(k.kick_beats)} found"))
    print(ui.kv(c, "saved", c.cyan(str(k.path))))
    print()
    print(f"  {c.grey('use it with')} fourfloor remix song.mp3 --kit " + k.name)
    print()
    return 0


def cmd_learn(args, c: ui.C) -> int:
    from .style import learn

    if not args.json:
        print(ui.header(c, f"learning from {Path(args.folder).name}"))
        print()
    progress = ui.Progress(c, quiet=args.json)
    try:
        st = learn(args.folder, progress=progress)
    finally:
        progress.close()
    st.save(args.output, anonymous=args.anonymous)
    if args.json:
        print(json.dumps(st.to_dict(args.anonymous), indent=2))
    else:
        print(_style_report(st, c))
        print(ui.kv(c, "saved", c.cyan(str(args.output))))
        print()
    return 0


def cmd_preview(args, c: ui.C) -> int:
    from .preview import write_preview

    audio = Path(args.input)
    sess_path = audio.with_suffix("").with_suffix(".session.json")
    if not sess_path.is_file():
        sess_path = Path(str(audio.with_suffix("")) + ".session.json")
    if not sess_path.is_file():
        print(ui.error(c, f"no session file beside {audio.name} "
                          f"(expected {sess_path.name})"), file=sys.stderr)
        return 2
    sess = json.loads(sess_path.read_text(encoding="utf8"))
    path = write_preview(audio, sess, out=args.output)
    print(ui.kv(c, "preview", c.cyan(str(path))))
    return 0


def _export_report(res, c: ui.C) -> str:
    """What was written, and what a DJ does with it."""
    from .export import HOT_CUE_LETTERS

    lines = ["", ui.rule(c, "export"), ""]
    lines.append(ui.kv(c, "set", c.bold(res.set_name)))
    lines.append(ui.kv(c, "tracks", str(len(res.tracks))))
    lines.append("")
    for t in res.tracks:
        lines.append(ui.kv(c, t.path.name[:28],
                           f"{c.bold(f'{t.bpm:.2f}')} BPM   "
                           f"{c.magenta(t.key)} {c.magenta(t.camelot)}   "
                           f"{fmt_time(t.duration)}", pad=30))
        cues = "  ".join(f"{HOT_CUE_LETTERS[i]} {cue.name}"
                         for i, cue in enumerate(t.hot_cues))
        lines.append(ui.kv(c, "", c.grey(cues), pad=30))
    lines.append("")
    if res.tagged:
        lines.append(ui.rule(c, "tagged"))
        for row in res.tagged:
            probe = row.get("probe") or {}
            stream = "{}s {}".format(probe.get("duration", "?"),
                                     probe.get("codec_name", "?"))
            lines.append(ui.kv(c, Path(row["file"]).name[:28],
                               c.grey("{} bytes of ID3".format(row["bytes"]))
                               + "   " + c.grey("audio unchanged: " + stream),
                               pad=30))
        lines.append("")
    if res.files:
        lines.append(ui.rule(c, "files"))
        for label, path in res.files.items():
            lines.append(ui.kv(c, label, c.cyan(str(path))))
        lines.append("")
    for note in res.notes:
        lines.append(ui.warn(c, note))
    if res.notes:
        lines.append("")
    if "rekordbox" in res.files:
        lines.append(f"  {c.grey('import it:')} Rekordbox > Preferences > Advanced > "
                     f"rekordbox xml > Imported Library, pick this file, then find "
                     f"'{res.set_name}' under rekordbox xml in the tree")
        lines.append("")
    return "\n".join(lines)


def cmd_export(args, c: ui.C) -> int:
    from . import export as export_mod

    target = Path(args.input)
    set_name = args.set_name or (target.name if target.is_dir() else target.stem)
    if not args.json:
        print(ui.header(c, f"export {target.name}"))
    res = export_mod.export(
        target, set_name, formats=args.formats, out_dir=args.out,
        artist=args.artist, suffix=args.suffix, verify=not args.no_verify,
    )
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(_export_report(res, c))
    return 0


def cmd_batch(args, c: ui.C) -> int:
    from . import batch as batch_mod

    if args.key is not None and args.key.strip().lower() != "auto":
        raise ValueError("batch takes --key auto or --key-lock; per-track keys are "
                         "what `fourfloor remix --key` is for")
    strategy = "auto" if (args.key or "").strip().lower() == "auto" else "lock"
    quiet = args.quiet or args.json
    reporter = batch_mod.Reporter(c, quiet=quiet)
    manifest = batch_mod.run(
        args.folder, args.out, bpm=args.bpm, key_strategy=strategy,
        stems=args.stems, kit=args.kit, jobs=args.jobs, resume=args.resume,
        length=args.length, form=args.form, swing=args.swing, seed=args.seed,
        wav=args.wav, set_name=args.set_name, artist=args.artist,
        on_event=reporter,
    )
    if args.json:
        print(json.dumps(manifest, indent=2))
    elif not quiet:
        print(reporter.report(manifest))
    return 1 if manifest["summary"]["failed"] and not manifest["summary"]["ok"] else 0


def cmd_serve(args, c: ui.C) -> int:
    from .server import serve

    return serve(port=args.port, open_browser=args.open_browser, home=args.home, c=c)


def main(argv: list[str] | None = None) -> int:
    c = ui.C(ui.supports_color())
    try:
        args = _parser().parse_args(argv)
    except CliError as exc:
        print(ui.error(c, str(exc)), file=sys.stderr)
        print("  try `fourfloor --help`", file=sys.stderr)
        return 2
    handlers = {"remix": cmd_remix, "inspect": cmd_inspect, "learn": cmd_learn,
                "preview": cmd_preview, "serve": cmd_serve, "fetch": cmd_fetch,
                "kit": cmd_kit, "export": cmd_export, "batch": cmd_batch,
                "critic": critic_command.run}
    try:
        return handlers[args.command](args, c)
    except KeyboardInterrupt:
        print("\n" + ui.error(c, "interrupted"), file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        # str(FileNotFoundError(path)) is just the path, which reads as if the
        # tool printed a stray filename and gave up.
        print(ui.error(c, f"no such file: {exc}"), file=sys.stderr)
        return 1
    except NotADirectoryError as exc:
        print(ui.error(c, f"not a folder: {exc}"), file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(ui.error(c, str(exc)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
