"""The remix pipeline: analyse, warp onto the target grid, separate, arrange, render."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import arrange, session
from .analysis import Analysis, analyze, suggest_house_tempo
from .analysis.key import KeyEstimate, nearest_compatible, parse_key, semitone_shift
from .audio import SR, decode, write_mp3, write_wav
from .dsp import phasevocoder as PV
from .dsp.pitch import TempoPlan, plan_tempo, resample_ratio
from .house.engine import Engine, Stems
from .stems import separate
from .style import Style

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


@dataclass
class RemixResult:
    """Outputs and measurements of one remix."""

    audio: np.ndarray
    sr: int
    plan: arrange.Plan
    session: dict
    analysis: Analysis
    tempo_plan: TempoPlan
    semitones: int
    target_key: KeyEstimate
    paths: dict[str, Path]
    metrics: dict
    warnings: list[str]


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


def _neighbours(camelot: str) -> list[str]:
    from .analysis.key import camelot_neighbours
    return camelot_neighbours(camelot)


def warp_source(a: Analysis, tempo: TempoPlan, semitones: int,
                progress=None) -> np.ndarray:
    """Warp the source so every detected beat lands exactly on the target grid.

    The beat map is fed straight to the phase vocoder as a piecewise time-warp,
    so tempo drift inside the source is absorbed beat by beat instead of by one
    global rate. When a pitch shift is requested the warp targets a grid that is
    ``2**(n/12)`` times longer, and the subsequent resample brings it back to
    length while moving every partial by the interval.
    """
    x = a.clip.samples
    beats = a.grid.beats
    sr = a.sr
    ratio_pitch = 2.0 ** (semitones / 12.0)
    target_beat = (60.0 / tempo.target_bpm) * tempo.beat_multiple * ratio_pitch

    if len(beats) < 2:
        # no usable grid: fall back to a single global stretch
        y = PV.time_stretch(x, tempo.ratio / ratio_pitch)
        return resample_ratio(y, 1.0 / ratio_pitch) if semitones else y

    # extend the map a beat beyond each end so the head and tail are covered
    period = float(np.median(np.diff(beats)))
    in_times = np.concatenate([[max(0.0, beats[0] - period)], beats,
                               [beats[-1] + period, len(x) / sr + period]])
    out_times = np.arange(len(in_times)) * target_beat
    # the lead-in keeps its own duration scaled by the local ratio
    out_times = out_times - out_times[0]
    out_len = int(round(out_times[-1] * sr))

    warped = PV.warp(x, in_times, out_times, sr, out_len)
    if semitones:
        warped = resample_ratio(warped, 1.0 / ratio_pitch)
    return warped


#: The phases ``remix`` reports through ``progress``, in order. The web app
#: draws this list before the job starts, so it lives next to the calls below.
PHASES = ("analyse", "warp", "separate", "arrange", "render", "write")


def validate_options(opts: RemixOptions, out: str | Path) -> None:
    """Reject an impossible request before any work happens.

    Every check here is cheap and needs no audio, so both the CLI and the web
    app can run it up front: a bad flag used to surface either as a numpy error
    deep in the render or as a session-schema failure after a full minute of
    work, with a half-written mp3 left behind.
    """
    out = Path(out)
    if opts.target_bpm is not None and not (MIN_TARGET_BPM <= opts.target_bpm <= MAX_TARGET_BPM):
        raise ValueError(
            f"--bpm {opts.target_bpm:g} is out of range; fourfloor targets "
            f"{MIN_TARGET_BPM:g}-{MAX_TARGET_BPM:g} BPM"
        )
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
    if opts.length:
        arrange.parse_length(opts.length)          # raises with its own message
    if opts.key and opts.key.lower() != "auto":
        parse_key(opts.key)                        # ditto


def remix(path: str | Path, out: str | Path, opts: RemixOptions | None = None,
          style: Style | None = None, progress=None) -> RemixResult:
    """Turn a song into a house remix and write every output file."""
    opts = opts or RemixOptions()
    out = Path(out)
    step = progress or (lambda *_a, **_k: None)

    validate_options(opts, out)

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
    warped = warp_source(a, tempo, semitones)

    step("separate", f"{opts.stems} separation")
    stems = separate(warped, a.sr, opts.stems)

    step("arrange", f"{opts.form} form")
    length = arrange.parse_length(opts.length) if opts.length else (
        style.length if style and style.length else 270.0)
    p = arrange.plan(a, target_bpm, tempo.beat_multiple, form_name=opts.form,
                     length=length, swing=swing, has_stems=(opts.stems == "demucs"))
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

    step("render", f"{p.total_bars} bars, {arrange.fmt_time(p.duration)}")
    engine = Engine(sr=a.sr, plan=p, stems=stems, chords=a.chords, semitones=semitones,
                    swing=swing, beat_multiple=tempo.beat_multiple,
                    src_bar_dur=a.bar_dur, seed=opts.seed)
    audio, metrics = engine.render()

    step("write", str(out))
    paths: dict[str, Path] = {}
    if out.suffix.lower() == ".wav":
        paths["wav"] = write_wav(out, audio, a.sr)
        paths["mp3"] = write_mp3(out.with_suffix(".mp3"), audio, a.sr)
    else:
        paths["mp3"] = write_mp3(out, audio, a.sr)
        if opts.wav:
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
        },
        tempo_plan=tempo.to_dict(),
    )
    issues = session.validate(sess)
    if issues:
        raise RuntimeError("session file failed validation: " + "; ".join(issues))
    paths["session"] = session.write(sess, f"{stem_base}.session.json")

    plan_path = Path(f"{stem_base}.plan.json")
    plan_path.write_text(json.dumps(p.to_dict(), indent=2) + "\n", encoding="utf8")
    paths["plan"] = plan_path

    return RemixResult(audio=audio, sr=a.sr, plan=p, session=sess, analysis=a,
                       tempo_plan=tempo, semitones=semitones, target_key=target_key,
                       paths=paths, metrics=metrics, warnings=warnings)
