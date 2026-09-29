"""The remix pipeline: analyse, warp onto the target grid, separate, arrange, render."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import arrange, session
from .analysis import Analysis, analyze, suggest_house_tempo
from .analysis.key import KeyEstimate, nearest_compatible, parse_key, semitone_shift
from .audio import SR, decode, write_mp3_under, write_wav
from .dsp.pitch import TempoPlan, plan_tempo
from .house.bass import choose_bass
from .house.vocal import choose_vocal
from .house.engine import Engine, Stems
from .stems import separate
from .style import Style
from . import kit as kits
from .warp import WarpMap, build as build_warp

#: The session schema promises a tempo a DJ tool can trust, and validates this
#: same window; reject an impossible target before any work happens.
MIN_TARGET_BPM = 60.0
MAX_TARGET_BPM = 200.0


@dataclass
class RemixOptions:
    """Everything the CLI can ask the pipeline to do."""

    target_bpm: float | None = None
    key: str | None = None
    compatible_with: str | None = None
    stems: str = "hpss"
    form: str = "club"
    length: str | None = None
    swing: float | None = None
    producer: bool = False
    seed: int = 0
    wav: bool = True
    kit: str | None = None
    """Name of a sampled drum kit, ``"none"`` for the synthesised one, or
    ``None`` for the most recently built kit if there is one."""
    bass: str = "auto"
    """What to do with the low end. ``auto`` measures the separated bass and
    keeps it unless it is an 808 doubling the kick; ``source`` always keeps it;
    ``sub`` always replaces it with a sub that follows its pitch on the house
    grid; ``none`` leaves the low end to the kick; ``synth`` is the old
    chord-guessing bass line."""
    kick_reinforce: bool = True
    drums_db: float = 0.0
    """Trim on the drum bus, in decibels, on top of the default level."""
    vocal: str = "auto"
    """``flow`` plays the voice as it was sung, ``chop`` cuts it into slices
    that start on syllables and land on beats, ``auto`` measures which the
    voice will stand."""
    keep_layers: bool = False
    """Hold on to the engine's individual buses so the alignment gate can
    measure the source layer without the kit shouting over it."""
    gate: bool = False
    """Run the layered alignment gate on the render (source layer, kit and
    arrangement spans, ~10-25 s more per track) and put its verdict in
    ``metrics`` under ``alignment_*`` keys, which is what ``fourfloor batch``
    records per track."""


@dataclass
class RemixResult:
    """Outputs and measurements of one remix."""

    audio: np.ndarray
    sr: int
    plan: arrange.Plan
    session: dict
    analysis: Analysis
    tempo_plan: TempoPlan
    warp: WarpMap | None
    kit_name: str | None
    bass_source: str
    vocal_mode: str
    semitones: int
    target_key: KeyEstimate
    paths: dict[str, Path]
    metrics: dict
    warnings: list[str]
    layers: dict[str, np.ndarray] = field(default_factory=dict)
    source_spans: list[tuple[int, int]] = field(default_factory=list)


def _resolve_key(a: Analysis, opts: RemixOptions) -> tuple[int, KeyEstimate, list[str]]:
    """Work out the semitone shift and the resulting key."""
    warnings: list[str] = []
    if opts.compatible_with:
        other = analyze(opts.compatible_with, keep_audio=False)
        shift, _ = nearest_compatible(a.key, other.key, max_shift=3)
        if shift == 0 and a.key.camelot not in _neighbours(other.key.camelot):
            warnings.append(
                f"no key within +/-3 semitones is Camelot-compatible with "
                f"{other.key.name} ({other.key.camelot}); keeping the original key"
            )
    elif opts.key and opts.key.lower() != "auto":
        pc, minor = parse_key(opts.key)
        shift = semitone_shift(a.key, pc, minor)
        if minor != a.key.is_minor:
            warnings.append(
                f"source is {a.key.name} and the target is "
                f"{'minor' if minor else 'major'}; pitch shifting moves the tonic "
                "but cannot change the mode, so only the tonic is matched"
            )
    else:
        shift = 0
    target = KeyEstimate((a.key.tonic + shift) % 12, a.key.is_minor, a.key.confidence)
    return int(shift), target, warnings


def _gate_metrics(audio, sr: int, bpm: float, engine, spans, sampled: bool) -> dict:
    """The layered alignment gate, flattened into ``alignment_*`` metrics.

    Judged on the isolated source layer, as the gate is calibrated for; the
    finished mix alone reads every render, good or bad, as failing its tails.
    """
    from .analysis.alignment import alignment_report

    rep = alignment_report(audio, sr, bpm, 0.0,
                           source_stem=engine.layers.get("source_perc"),
                           kit_layer=engine.layers.get("kit"), spans=spans,
                           kit_is_sampled=sampled)
    judged = rep.get("source", rep["mix"])
    out = {"alignment_ok": bool(rep["ok"]),
           "alignment_problems": list(rep["problems"]),
           "alignment_judged_on": rep["judged_on"],
           "alignment_median_ms": round(float(judged["median_ms"]), 2),
           "alignment_p90_ms": round(float(judged["p90_ms"]), 2),
           "alignment_within_20ms": round(float(judged["within_20ms"]), 4),
           "alignment_bar_phase": int(judged["bar_phase"])}
    if "kit" in rep:
        out["alignment_kit_median_ms"] = round(float(rep["kit"]["median_ms"]), 2)
    return out


def _bass_label(mode: str, stem_name: str) -> str:
    """What the report and the session file should say the low end is."""
    return {
        "source": stem_name,
        "sub": f"pitch-tracked sub from the {stem_name}",
        "none": "none (the kick carries the low end)",
        "synth": "synth",
    }.get(mode, mode)


def _neighbours(camelot: str) -> list[str]:
    from .analysis.key import camelot_neighbours
    return camelot_neighbours(camelot)


#: The phases ``remix`` reports through ``progress``, in order. The web app
#: draws this list before the job starts, so it lives next to the calls below.
PHASES = ("analyse", "warp", "separate", "arrange", "render", "write")


def output_paths(out: str | Path, wav: bool) -> list[Path]:
    """Every file :func:`remix` writes for ``out``: mp3, wav, session, plan."""
    out = Path(out)
    is_wav = out.suffix.lower() == ".wav"
    paths = [out.with_suffix(".mp3") if is_wav else out]
    if is_wav or wav:
        paths.append(out.with_suffix(".wav"))
    stem_base = out.with_suffix("")
    paths += [Path(f"{stem_base}.session.json"), Path(f"{stem_base}.plan.json")]
    return paths


def _same_file(a: Path, b: Path) -> bool:
    """True when ``a`` and ``b`` name one file (case-insensitive disks included)."""
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return a.resolve() == b.resolve()


def validate_options(opts: RemixOptions, out: str | Path,
                     source: str | Path | None = None) -> None:
    """Reject an impossible request before any work happens.

    Every check here is cheap and needs no audio, so both the CLI and the web
    app can run it up front: a bad flag used to surface either as a numpy error
    deep in the render or as a session-schema failure after a full minute of
    work, with a half-written mp3 left behind.

    With ``source``, also refuse an output that would write over it: an
    ``-o song.mp3`` beside ``song.wav`` writes ``song.wav`` too (the WAV copy),
    and the original was replaced by the remix with no warning.
    """
    out = Path(out)
    if opts.target_bpm is not None and not (MIN_TARGET_BPM <= opts.target_bpm <= MAX_TARGET_BPM):
        raise ValueError(
            f"--bpm {opts.target_bpm:g} is out of range; fourfloor targets "
            f"{MIN_TARGET_BPM:g}-{MAX_TARGET_BPM:g} BPM"
        )
    if opts.vocal not in ("auto", "flow", "chop"):
        raise ValueError(f"--vocal {opts.vocal!r} is not a mode; "
                         "use auto, flow or chop")
    if not (-24.0 <= opts.drums_db <= 12.0):
        raise ValueError(f"--drums-db {opts.drums_db:g} is out of range; use -24 to +12")
    if opts.swing is not None and not (0.0 <= opts.swing <= 0.66):
        raise ValueError(f"--swing {opts.swing:g} is out of range; use 0 to 0.66")
    if out.suffix.lower() not in (".mp3", ".wav"):
        raise ValueError(
            f"output must end in .mp3 or .wav, got {out.name!r}"
        )
    if opts.form not in arrange.FORMS:
        raise ValueError(
            f"--form {opts.form!r} is not a form; choose from "
            + ", ".join(arrange.FORMS)
        )
    if opts.stems not in ("hpss", "demucs"):
        raise ValueError(f"--stems {opts.stems!r} is not an engine; use hpss or demucs")
    if opts.bass not in ("auto", "source", "sub", "none", "synth"):
        raise ValueError(f"--bass {opts.bass!r} is not a bass; "
                         "use auto, source, sub, none or synth")
    if opts.length:
        arrange.parse_length(opts.length)          # raises with its own message
    if opts.key and opts.key.lower() != "auto":
        parse_key(opts.key)                        # ditto
    if opts.kit and opts.kit.strip().lower() != "none":
        # a misspelt kit used to be found only after the analysis, the warp
        # and (with demucs) minutes of separation
        name = kits.check_name(opts.kit)
        folder = kits.kits_home() / name
        if not ((folder / "meta.json").is_file() and (folder / "loop.wav").is_file()):
            raise ValueError(f"no kit called {name!r} in {kits.kits_home()}; "
                             "`fourfloor kit list` shows the ones you have")
    if source is not None:
        src = Path(source)
        for p in output_paths(out, opts.wav):
            if _same_file(p, src):
                raise ValueError(
                    f"writing {p.name} would replace the source {src.name}; "
                    "choose another -o")


def remix(path: str | Path, out: str | Path, opts: RemixOptions | None = None,
          style: Style | None = None, progress=None) -> RemixResult:
    """Turn a song into a house remix and write every output file."""
    opts = opts or RemixOptions()
    out = Path(out)
    step = progress or (lambda *_a, **_k: None)

    validate_options(opts, out, source=path)
    drum_kit = kits.resolve(opts.kit)

    step("analyse", "decoding and analysing the source")
    clip = decode(path)
    a = analyze(path, clip=clip)

    target_bpm = opts.target_bpm or (style.bpm if style else None) \
        or suggest_house_tempo(a.grid.bpm)
    swing = opts.swing if opts.swing is not None else (style.swing if style else 0.08)

    semitones, target_key, warnings = _resolve_key(a, opts)
    tempo = plan_tempo(a.grid.bpm, target_bpm)
    if tempo.warning:
        warnings.append(tempo.warning)

    step("warp", f"{a.grid.bpm:.2f} -> {target_bpm:.2f} BPM "
                 f"({tempo.interpretation}, x{tempo.ratio:.3f})"
                 + (f", {semitones:+d} semitones" if semitones else ""))
    warped, wmap = build_warp(a, target_bpm, tempo.beat_multiple, semitones)

    step("separate", f"{opts.stems} separation")
    stems = separate(warped, a.sr, opts.stems, want_bass=(opts.bass != "synth"))

    vocal_mode, vocal_why = opts.vocal, ""
    if opts.vocal == "auto":
        if stems.vocals is None:
            vocal_mode = "flow"
        else:
            vocal_mode, _vm, vocal_why = choose_vocal(stems.vocals, a.sr, target_bpm)
    if vocal_why:
        warnings.append(f"vocal: {vocal_why}")

    bass_mode, bass_why = opts.bass, ""
    if opts.bass == "auto":
        if stems.bass is None:
            bass_mode, bass_why = "synth", "nothing could be separated to play"
        else:
            bass_mode, _meas, bass_why = choose_bass(stems.bass, a.sr, target_bpm)
    if bass_why:
        warnings.append(f"bass: {bass_why}")

    step("arrange", f"{opts.form} form")
    length = arrange.parse_length(opts.length) if opts.length else (
        style.length if style and style.length else 270.0)
    p = arrange.plan(a, target_bpm, tempo.beat_multiple, form_name=opts.form,
                     length=length, swing=swing, has_stems=(opts.stems == "demucs"),
                     warp=wmap, has_kit=drum_kit is not None)
    if length and p.duration > length + 4.0 * p.bar_dur:
        min_bars = arrange.form_min_bars(arrange.FORMS.get(opts.form, arrange.FORMS["club"]))
        warnings.append(
            f"the {opts.form} form is at least {min_bars} bars "
            f"({arrange.fmt_time(min_bars * p.bar_dur)} at {target_bpm:.0f} BPM), so "
            f"--length {arrange.fmt_time(length)} was rounded up to "
            f"{arrange.fmt_time(p.duration)}"
        )
    if opts.producer:
        from .producer import apply_producer_plan
        p, note = apply_producer_plan(a, p, warnings)
        if note:
            p.note = note
    problems = arrange.validate(p)
    if problems:
        raise RuntimeError("arrangement failed validation: " + "; ".join(problems))

    step("render", f"{p.total_bars} bars, {arrange.fmt_time(p.duration)}"
                   + (f", {drum_kit.name} kit" if drum_kit else ""))
    engine = Engine(sr=a.sr, plan=p, stems=stems, chords=a.chords, semitones=semitones,
                    swing=swing, beat_multiple=tempo.beat_multiple,
                    src_bar_dur=a.bar_dur, seed=opts.seed, warp=wmap,
                    drum_kit=drum_kit, kick_reinforce=opts.kick_reinforce,
                    bass_mode=bass_mode, drums_db=opts.drums_db,
                    vocal_mode=vocal_mode)
    audio, metrics = engine.render()
    layers = engine.layers if opts.keep_layers else {}
    spans = list(engine.source_spans)
    if opts.gate:
        metrics = dict(metrics, **_gate_metrics(audio, a.sr, target_bpm, engine,
                                                spans, drum_kit is not None))

    step("write", str(out))
    paths: dict[str, Path] = {}
    # The MP3 first: it is the file a DJ plays, and the encoder's overshoot
    # decides how far the buffer has to come down to keep it off full scale.
    # The WAV, the session and the reported levels then describe that same
    # buffer and that file.
    mp3_path = out.with_suffix(".mp3") if out.suffix.lower() == ".wav" else out
    paths["mp3"], audio, mp3_peak = write_mp3_under(mp3_path, audio, a.sr)
    metrics = dict(metrics, peak_db=mp3_peak,
                   rms_db=float(20 * np.log10(max(float(np.sqrt(np.mean(np.square(audio)))), 1e-9))))
    if out.suffix.lower() == ".wav" or opts.wav:
        paths["wav"] = write_wav(out.with_suffix(".wav"), audio, a.sr)

    stem_base = out.with_suffix("")
    sess = session.build(
        p, audio, a.sr, paths.get("mp3", out), target_key.name, target_key.camelot,
        semitones,
        source={
            "file": Path(path).name,
            "bpm": round(a.grid.bpm, 2),
            "key": a.key.name,
            "camelot": a.key.camelot,
            "key_confidence": round(a.key.confidence, 3),
            "duration": round(a.duration, 2),
            "separation": stems.source_name,
            "drums": (drum_kit.name if drum_kit else "synth"),
            "bass": _bass_label(bass_mode, stems.bass_name),
            "vocal": vocal_mode,
        },
        tempo_plan=tempo.to_dict(),
    )
    # the peak a player will actually see: the decoded MP3's, not the buffer's
    sess["loudness"]["peak_db"] = round(mp3_peak, 2)
    issues = session.validate(sess)
    if issues:
        raise RuntimeError("session file failed validation: " + "; ".join(issues))
    paths["session"] = session.write(sess, f"{stem_base}.session.json")

    plan_path = Path(f"{stem_base}.plan.json")
    plan_path.write_text(json.dumps(p.to_dict(), indent=2) + "\n", encoding="utf8")
    paths["plan"] = plan_path

    return RemixResult(audio=audio, sr=a.sr, plan=p, session=sess, analysis=a,
                       tempo_plan=tempo, warp=wmap,
                       kit_name=(drum_kit.name if drum_kit else None),
                       bass_source=_bass_label(bass_mode, stems.bass_name),
                       vocal_mode=vocal_mode, semitones=semitones, target_key=target_key,
                       paths=paths, metrics=metrics, warnings=warnings,
                       layers=layers, source_spans=spans)
