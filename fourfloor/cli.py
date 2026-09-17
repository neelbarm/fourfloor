"""Command line interface: remix, inspect, learn, preview."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ui
from .arrange import FORMS, fmt_time
from .style import Style

VERSION = "0.1.0"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fourfloor",
        description="Turn any song into a house remix, with a session file a DJ app can sync to.",
    )
    p.add_argument("--version", action="version", version=f"fourfloor {VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("remix", help="build a house remix of a song")
    r.add_argument("input", help="source audio file (mp3, m4a, wav, flac…)")
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
    r.add_argument("--producer", action="store_true",
                   help="ask Claude to plan the arrangement (needs ANTHROPIC_API_KEY)")
    r.add_argument("--seed", type=int, default=0, help="randomisation seed")
    r.add_argument("--no-wav", action="store_true", help="write only the mp3")
    r.add_argument("--preview", action="store_true", help="also write preview.html")
    r.add_argument("--json", action="store_true", help="print the session JSON instead of a report")
    r.add_argument("-q", "--quiet", action="store_true")

    i = sub.add_parser("inspect", help="analyse a track and print a report")
    i.add_argument("input")
    i.add_argument("--json", action="store_true")

    l = sub.add_parser("learn", help="derive a style profile from a folder of remixes")
    l.add_argument("folder")
    l.add_argument("-o", "--output", default="style.json")
    l.add_argument("--anonymous", action="store_true",
                   help="omit per-file rows; save aggregate numbers only")
    l.add_argument("--json", action="store_true")

    v = sub.add_parser("preview", help="write a preview.html next to a remix")
    v.add_argument("input", help="a remix mp3/wav that has a .session.json beside it")
    v.add_argument("-o", "--output", default=None)
    return p


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

def cmd_inspect(args, c: ui.C) -> int:
    from .analysis import analyze

    a = analyze(args.input)
    if args.json:
        print(json.dumps(a.to_dict(), indent=2))
    else:
        print(_inspect_report(a, c))
    return 0


def cmd_remix(args, c: ui.C) -> int:
    from .remix import RemixOptions, remix

    src = Path(args.input)
    out = Path(args.output) if args.output else src.with_suffix("").with_name(
        src.stem + ".house.mp3")
    style = Style.load(args.style) if args.style else None

    opts = RemixOptions(
        target_bpm=args.bpm, key=args.key, compatible_with=args.compatible_with,
        stems=args.stems, form=args.form, length=args.length, swing=args.swing,
        producer=args.producer, seed=args.seed, wav=not args.no_wav,
    )
    quiet = args.quiet or args.json
    if not quiet:
        print(ui.header(c, f"remixing {src.name}"))
        print()
    progress = ui.Progress(c, quiet=quiet)
    try:
        res = remix(src, out, opts, style=style, progress=progress)
    finally:
        progress.close()

    if args.preview:
        from .preview import write_preview
        res.paths["preview"] = write_preview(res.paths.get("mp3", out), res.session)

    if args.json:
        print(json.dumps(res.session, indent=2))
    elif not quiet:
        print(_remix_report(res, c, progress.total))
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    c = ui.C(ui.supports_color())
    handlers = {"remix": cmd_remix, "inspect": cmd_inspect,
                "learn": cmd_learn, "preview": cmd_preview}
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
