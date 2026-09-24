"""Remix a whole folder into one set: one tempo, a key that flows, one export.

A gig is not one remix, it is forty minutes of them that have to mix into each
other. That means three things this module does and ``fourfloor remix`` does
not:

* **one tempo for the set**, so any track can follow any other without a tempo
  fader move;
* **keys that flow** -- ``--key auto`` walks the Camelot wheel, shifting each
  track by at most two semitones onto a key that is compatible with the one
  before it, so the set never hits a clash;
* **one handoff** -- a ``set.json`` describing every track, and a
  ``rekordbox.xml`` plus ``cues.csv`` written by :mod:`fourfloor.export`.

Every track is independent: one that fails is recorded in ``set.json`` and the
batch carries on. ``--resume`` picks up where an interrupted run stopped, which
matters when the folder is thirty tracks and the render is not fast.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import ui

SCHEMA_VERSION = 1

#: How far ``--key auto`` will move a track to make it mix with the last one.
#: Beyond a couple of semitones the vocal starts to sound like a different
#: singer, which is a worse problem than a key clash a DJ can EQ around.
MAX_AUTO_SHIFT = 2

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


class BatchError(RuntimeError):
    """Something a person can fix before any rendering starts."""


# ---------------------------------------------------------------------------
# key flow
# ---------------------------------------------------------------------------

def shift_camelot(code: str, semitones: int) -> str:
    """The Camelot code ``semitones`` above ``code``, mode unchanged.

    Pitch shifting moves the tonic and leaves major/major, minor/minor, so the
    wheel letter never changes.
    """
    from .analysis.key import KeyEstimate, camelot_to_key

    pc, minor = camelot_to_key(code)
    return KeyEstimate((pc + int(semitones)) % 12, minor, 1.0).camelot


def plan_key_flow(camelots: list[str], max_shift: int = MAX_AUTO_SHIFT,
                  previous: str | None = None) -> list[dict]:
    """Choose a semitone shift per track so consecutive keys mix.

    The first track keeps its own key -- it sets the tone of the set and there
    is nothing before it to clash with. Every track after it takes the smallest
    shift within ``±max_shift`` that lands in the previous *target* key's
    Camelot neighbourhood (same code, one step round the wheel either way, or
    the relative major/minor). When nothing in range works the track is left
    alone and flagged, because a two-semitone shift that still clashes is the
    worst of both worlds.

    ``previous`` is the Camelot code of a track that already exists ahead of
    this list, which is what a resumed batch has: the first track then has
    something to mix out of and is planned like any other.

    Returns one dict per track: ``source``, ``shift``, ``camelot``, ``key`` and
    ``compatible``.
    """
    from .analysis.key import KeyEstimate, camelot_neighbours, camelot_to_key

    out: list[dict] = []
    for code in camelots:
        code = (code or "").strip().upper()
        try:
            pc, minor = camelot_to_key(code)
        except ValueError:
            out.append({"source": code, "shift": 0, "camelot": code, "key": "",
                        "compatible": False})
            continue
        if previous is None:
            shift, compatible = 0, True
        else:
            wanted = set(camelot_neighbours(previous))
            shift, compatible = 0, code in wanted
            if not compatible:
                for cand in sorted(range(-max_shift, max_shift + 1), key=abs):
                    if shift_camelot(code, cand) in wanted:
                        shift, compatible = cand, True
                        break
        target = KeyEstimate((pc + shift) % 12, minor, 1.0)
        out.append({"source": code, "shift": int(shift), "camelot": target.camelot,
                    "key": target.name, "compatible": bool(compatible)})
        previous = target.camelot
    return out


# ---------------------------------------------------------------------------
# one track
# ---------------------------------------------------------------------------

@dataclass
class TrackRow:
    """One line of ``set.json`` and one row of the terminal table."""

    source: str
    output: str | None = None
    session: str | None = None
    status: str = STATUS_FAILED
    error: str | None = None
    bpm: float | None = None
    key: str | None = None
    camelot: str | None = None
    duration: float | None = None
    semitone_shift: int | None = None
    source_bpm: float | None = None
    source_key: str | None = None
    source_camelot: str | None = None
    cues: list = field(default_factory=list)
    alignment: dict | None = None
    seconds: float = 0.0

    @property
    def name(self) -> str:
        return Path(self.source).name


def _jsonable(value):
    """numpy scalars and Paths do not survive ``json.dumps``; everything here does."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    item = getattr(value, "item", None)          # numpy scalar
    if callable(item):
        try:
            return _jsonable(item())
        except (ValueError, TypeError):
            pass
    return str(value)


def _alignment_of(metrics: dict | None) -> dict | None:
    """Whatever the engine reports about beat alignment, if it reports any.

    The audio engine owns these metric names and they move; rather than depend
    on one, take every key that is about alignment and pass it through.
    """
    if not metrics:
        return None
    picked = {k: _jsonable(v) for k, v in metrics.items()
              if "align" in k.lower() or k in ("comb_sharpness", "grid_error")}
    return picked or None


def output_for(source: Path, out_dir: Path) -> Path:
    """Where a source track's remix lands: ``<out>/<name>.house.mp3``."""
    return Path(out_dir) / f"{Path(source).stem}.house.mp3"


def plan_outputs(sources: list[Path], out_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """Every source's output path, with no two sources sharing one.

    ``Song.mp3`` and ``Song.m4a`` -- a purchase next to a download -- would
    both land on ``Song.house.mp3``, the second render overwriting the first
    (or, with ``--jobs``, two workers writing one file at once). Sources whose
    names clash carry their format in the output name instead:
    ``Song (mp3).house.mp3`` and ``Song (m4a).house.mp3``. Names are compared
    without case, as the Mac's disk compares them.
    """
    groups: dict[str, list[Path]] = {}
    for p in sources:
        groups.setdefault(output_for(p, out_dir).name.lower(), []).append(Path(p))
    taken = {name for name, group in groups.items() if len(group) == 1}
    outputs: dict[str, Path] = {}
    clashes: list[str] = []
    for name, group in groups.items():
        if len(group) == 1:
            outputs[str(group[0])] = output_for(group[0], out_dir)
            continue
        renamed = []
        for p in group:
            fmt = p.suffix.lstrip(".").lower() or "audio"
            label, n = fmt, 2
            while f"{p.stem} ({label}).house.mp3".lower() in taken:
                label, n = f"{fmt} {n}", n + 1
            dest = Path(out_dir) / f"{p.stem} ({label}).house.mp3"
            taken.add(dest.name.lower())
            outputs[str(p)] = dest
            renamed.append(dest.name)
        clashes.append(f"{', '.join(p.name for p in group)} would all be "
                       f"{output_for(group[0], out_dir).name}, so they became "
                       f"{', '.join(renamed)}")
    return outputs, clashes


#: What a render was asked for, as far as ``--resume`` cares: change any of
#: these and a finished track is no longer the track this batch wants.
RENDER_KEYS = ("bpm", "key_strategy", "stems", "kit", "bass", "form", "length",
               "swing", "seed", "vocal", "drums_db", "kick_reinforce")

#: Every fourfloor mp3 is 320 kbit/s constant bitrate: 40 000 bytes a second.
MP3_BYTES_PER_SEC = 320_000 / 8
#: An mp3 shorter than this share of its session's duration was cut off.
MIN_COMPLETE = 0.98


def stamp_path_for(out: Path) -> Path:
    """``X.house.mp3`` -> ``X.house.batch.json``: what the batch asked for."""
    return Path(out).with_suffix(".batch.json")


def plan_path_for(out: Path) -> Path:
    return Path(out).with_suffix(".plan.json")


def render_params(job: dict) -> dict:
    """The part of a job that decides what the render sounds like."""
    defaults = {"key_strategy": "lock", "stems": "hpss", "kit": None,
                "bass": "auto", "form": "club", "length": None, "swing": None,
                "seed": 0, "vocal": "auto", "drums_db": 0.0,
                "kick_reinforce": True}
    params = {k: job.get(k, defaults.get(k)) for k in RENDER_KEYS}
    params["bpm"] = float(params["bpm"])
    params["drums_db"] = float(params["drums_db"])
    params["kick_reinforce"] = bool(params["kick_reinforce"])
    return json.loads(json.dumps(params))           # the form it has on disk


def audio_bytes(path: Path) -> int:
    """Size of an mp3 without its leading ID3v2 tag.

    Tagging (``fourfloor export --format tags``) rewrites only the tag, so
    this number is the same before and after, and different for any new or
    cut-off encode.
    """
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as fh:
        head = fh.read(10)
    if len(head) == 10 and head[:3] == b"ID3":
        tag = (head[6] << 21) | (head[7] << 14) | (head[8] << 7) | head[9]
        size -= 10 + tag + (10 if head[5] & 0x10 else 0)
    return max(size, 0)


def _write_stamp(out: Path, job: dict) -> None:
    """Record, after a finished render, what it was asked for and how big it is."""
    stamp = stamp_path_for(out)
    tmp = stamp.with_name(stamp.name + ".tmp")
    data = {"generator": "fourfloor", "params": render_params(job),
            "audio_bytes": audio_bytes(out)}
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")
    os.replace(tmp, stamp)


def _legacy_mismatch(sess: dict, want: dict) -> str | None:
    """Compare a render with no stamp against the request, by its session.

    The session records the tempo, the key shift and the separation, kit,
    bass and vocal it used. Anything a session cannot vouch for (a drum trim,
    no kick reinforcement) has to be a render made before these existed, which
    used the defaults.
    """
    src = sess.get("source") or {}
    if abs(float(sess.get("bpm") or 0.0) - want["bpm"]) > 0.01:
        return f"rendered at {sess.get('bpm')} BPM, not {want['bpm']:g}"
    if want["key_strategy"] == "lock" and int(sess.get("semitone_shift") or 0):
        return (f"shifted {int(sess['semitone_shift']):+d} st, but keys are "
                "locked for this set")
    if src.get("separation") and src["separation"] != want["stems"]:
        return f"separated with {src['separation']}, not {want['stems']}"
    kit = want["kit"] or "none"
    drums = "synth" if kit.lower() == "none" else kit
    if src.get("drums") and src["drums"] != drums:
        return f"drums from {src['drums']}, not {drums}"
    if want["vocal"] != "auto" and src.get("vocal") and src["vocal"] != want["vocal"]:
        return f"vocal {src['vocal']}, not {want['vocal']}"
    label = src.get("bass")
    if want["bass"] != "auto" and label:
        fits = {"source": label in ("demucs bass", "hpss low band"),
                "sub": label.startswith("pitch-tracked sub"),
                "none": label.startswith("none"),
                "synth": label == "synth"}.get(want["bass"], True)
        if not fits:
            return f"bass {label}, not {want['bass']}"
    if want["drums_db"] != 0.0 or not want["kick_reinforce"]:
        return "no record of the drum settings it was rendered with"
    return None


def resume_check(source: Path, out_dir: Path, want: dict | None = None,
                 dest: Path | None = None) -> tuple[bool, str]:
    """Whether a finished render can stand in for this batch's, and why not.

    Finished means the mp3, its session and its plan are all there and the mp3
    is whole. The batch stamps each render it finishes with what it was asked
    for and the size of its audio; a render whose stamp matches ``want`` (see
    :func:`render_params`) is kept. A render with no stamp, or whose audio has
    changed since -- an older batch, ``fourfloor remix`` writing over it, a
    re-render interrupted mid-encode -- is judged by what its session says it
    is, and by whether its mp3 is as long as the session says.
    """
    from .export import load_session, session_path_for

    out = Path(dest) if dest is not None else output_for(source, out_dir)
    if not out.is_file() or out.stat().st_size == 0:
        return False, "no mp3"
    if not session_path_for(out).is_file():
        return False, "no session file"
    try:
        sess = load_session(out)
    except Exception as exc:                          # noqa: BLE001
        return False, str(exc)
    if not plan_path_for(out).is_file():
        return False, "no plan file (the render did not finish)"
    if want is not None and want.get("wav") and not out.with_suffix(".wav").is_file():
        return False, "no wav"
    size = audio_bytes(out)

    stamp = None
    try:
        stamp = json.loads(stamp_path_for(out).read_text(encoding="utf8"))
    except (OSError, ValueError):
        pass
    if isinstance(stamp, dict) and stamp.get("audio_bytes") == size:
        if want is None:
            return True, ""
        have = stamp.get("params") or {}
        want_p = render_params(want)
        diff = [f"{k} {have.get(k)!r} -> {want_p[k]!r}" for k in RENDER_KEYS
                if have.get(k) != want_p[k]]
        return (not diff), ("rendered with " + ", ".join(diff)) if diff else ""

    duration = float(sess.get("duration") or 0.0)
    if duration > 0 and size < MIN_COMPLETE * duration * MP3_BYTES_PER_SEC:
        return False, (f"the mp3 holds {size / MP3_BYTES_PER_SEC:.1f}s of a "
                       f"{duration:.1f}s render (cut off)")
    if want is not None:
        why = _legacy_mismatch(sess, render_params(want))
        if why:
            return False, why
    return True, ""


def already_done(source: Path, out_dir: Path, want: dict | None = None,
                 dest: Path | None = None) -> bool:
    """True when a finished render of ``source`` can be kept; see :func:`resume_check`."""
    return resume_check(source, out_dir, want, dest)[0]


def _row_from_session(source: Path, out: Path, sess: dict, status: str,
                      seconds: float = 0.0) -> TrackRow:
    from .export import session_path_for

    src = sess.get("source", {}) or {}
    return TrackRow(
        source=str(source), output=str(out), session=str(session_path_for(out)),
        status=status,
        bpm=sess.get("bpm"), key=sess.get("key"), camelot=sess.get("camelot"),
        duration=sess.get("duration"), semitone_shift=sess.get("semitone_shift"),
        source_bpm=src.get("bpm"), source_key=src.get("key"),
        source_camelot=src.get("camelot"),
        cues=[{"name": c.get("name"), "time": c.get("time"), "bar": c.get("bar"),
               "kind": c.get("kind")} for c in sess.get("cues", [])],
        seconds=round(seconds, 2),
    )


def _remix_one(job: dict) -> dict:
    """Render one track. Never raises: a failure comes back as a row.

    Module level and plain-dict in/out so it can run in a worker process.
    """
    source, out = Path(job["source"]), Path(job["output"])
    t0 = time.time()
    try:
        from .remix import RemixOptions, remix

        # Until this render finishes, whatever is at ``out`` is not a render of
        # this job: an interrupted encode must not pass for a finished one.
        stamp_path_for(out).unlink(missing_ok=True)

        opts = RemixOptions(
            target_bpm=job["bpm"], key=job.get("key"), stems=job.get("stems", "hpss"),
            form=job.get("form", "club"), length=job.get("length"),
            swing=job.get("swing"), seed=job.get("seed", 0),
            wav=bool(job.get("wav", False)), kit=job.get("kit"),
            bass=job.get("bass", "auto"), vocal=job.get("vocal", "auto"),
            drums_db=float(job.get("drums_db", 0.0)),
            kick_reinforce=bool(job.get("kick_reinforce", True)),
        )
        res = remix(source, out, opts)
        mp3 = Path(res.paths.get("mp3", out))
        try:
            _write_stamp(mp3, job)
        except OSError:
            pass                    # still a good render; resume judges it by its session
        row = _row_from_session(source, mp3, res.session, STATUS_OK, time.time() - t0)
        row.alignment = _alignment_of(getattr(res, "metrics", None))
        return asdict(row)
    except Exception as exc:                          # noqa: BLE001 - isolation is the point
        row = TrackRow(source=str(source), output=None, status=STATUS_FAILED,
                       error=f"{type(exc).__name__}: {exc}",
                       seconds=round(time.time() - t0, 2))
        return asdict(row)


# ---------------------------------------------------------------------------
# the batch
# ---------------------------------------------------------------------------

def _camelot_or_none(code: str | None) -> str | None:
    """``code`` as a Camelot code the planner can mix out of, else ``None``."""
    from .analysis.key import camelot_to_key

    code = (code or "").strip().upper()
    try:
        camelot_to_key(code)
    except ValueError:
        return None
    return code


def _source_keys(sources: list[Path], on_event) -> list[str]:
    """Camelot code of every source, for the key-flow pre-pass.

    This costs one analysis per track before any rendering, which is the price
    of knowing the whole set's keys before choosing the first shift.
    """
    from .analysis import analyze

    codes = []
    for p in sources:
        try:
            a = analyze(p, keep_audio=False)
            codes.append(a.key.camelot)
            on_event("read", {"source": str(p), "camelot": a.key.camelot,
                              "key": a.key.name, "bpm": round(a.grid.bpm, 2)})
        except Exception as exc:                      # noqa: BLE001
            codes.append("")
            on_event("read_failed", {"source": str(p), "error": str(exc)})
    return codes


def _check_options(out: Path, *, bpm, stems, kit, bass, length, form, swing, seed,
                   wav, vocal, drums_db, kick_reinforce) -> str:
    """Refuse a bad flag before any track is analysed, and name the one kit.

    Without this a typo reached every track separately: a misspelt ``--kit``
    was only noticed after each track's decode, analysis, warp and (with
    Demucs) minutes of separation, and the whole set came out empty.

    Returns the concrete kit name every track will use -- the one asked for,
    else the default right now, else ``"none"`` for the synthesised kit.
    """
    from . import kit as kit_mod
    from .remix import RemixOptions, validate_options

    opts = RemixOptions(target_bpm=float(bpm), stems=stems, form=form, length=length,
                        swing=swing, seed=seed, wav=wav, kit=kit, bass=bass,
                        vocal=vocal, drums_db=float(drums_db),
                        kick_reinforce=bool(kick_reinforce))
    try:
        validate_options(opts, out / "check.house.mp3")
    except ValueError as exc:
        raise BatchError(str(exc)) from exc
    name = kit or kit_mod.default_name() or "none"
    if name.lower() != "none":
        try:
            kit_mod.load(name)
        except FileNotFoundError as exc:
            raise BatchError(str(exc)) from exc
        except (OSError, ValueError, RuntimeError) as exc:
            raise BatchError(f"the {name!r} kit could not be read: {exc}") from exc
    return name


def run(folder: str | Path, out_dir: str | Path, *, bpm: float,
        key_strategy: str = "lock", stems: str = "hpss", kit: str | None = None,
        bass: str = "auto", jobs: int = 1, resume: bool = False,
        length: str | None = None, form: str = "club", swing: float | None = None,
        seed: int = 0, wav: bool = False, set_name: str | None = None,
        artist: str = "fourfloor", vocal: str = "auto", drums_db: float = 0.0,
        kick_reinforce: bool = True, on_event=None) -> dict:
    """Remix every track in ``folder`` at ``bpm`` and export the set.

    ``kit``, ``bass``, ``vocal``, ``drums_db`` and ``kick_reinforce`` are
    handed to every track unchanged: a set wants one drum kit and one low-end
    policy across it, not a different decision per file. Their meanings are
    :class:`~fourfloor.remix.RemixOptions`'s. With no ``kit`` the default kit
    is looked up once, here, so a kit built while the batch runs cannot change
    the drums halfway through the set.

    ``on_event(kind, payload)`` is called as things happen so a terminal (or a
    web app) can draw progress without this module knowing about either.
    """
    from .audio import find_audio

    on_event = on_event or (lambda *_a, **_k: None)
    src_dir, out = Path(folder), Path(out_dir)
    if not src_dir.is_dir():
        raise NotADirectoryError(str(src_dir))
    if key_strategy not in ("lock", "auto"):
        raise BatchError(f"key strategy must be 'lock' or 'auto', not {key_strategy!r}")
    kit = _check_options(out, bpm=bpm, stems=stems, kit=kit, bass=bass,
                         length=length, form=form, swing=swing, seed=seed, wav=wav,
                         vocal=vocal, drums_db=drums_db,
                         kick_reinforce=kick_reinforce)
    jobs = max(1, int(jobs))
    set_name = set_name or src_dir.name

    sources = find_audio(src_dir)
    if not sources:
        raise BatchError(f"no audio files in {src_dir}")
    if out.exists() and out.resolve() == src_dir.resolve():
        raise BatchError("send the remixes somewhere other than the source folder, "
                         "or the next run will remix its own output")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    on_event("start", {"tracks": len(sources), "bpm": bpm, "set": set_name,
                       "key_strategy": key_strategy, "out": str(out)})
    outputs, notes = plan_outputs(sources, out)
    for note in notes:
        on_event("renamed", {"note": note})

    # What every track is asked for; the key is added per track below.
    request = {"bpm": float(bpm), "key_strategy": key_strategy, "stems": stems,
               "form": form, "length": length, "swing": swing, "seed": seed,
               "wav": wav, "kit": kit, "bass": bass, "vocal": vocal,
               "drums_db": float(drums_db), "kick_reinforce": bool(kick_reinforce)}

    rows: dict[str, TrackRow] = {}
    pending: list[Path] = []
    for p in sources:
        if not resume:
            pending.append(p)
            continue
        dest = outputs[str(p)]
        keep, why = resume_check(p, out, request, dest=dest)
        if not keep:
            if dest.exists():
                on_event("resume_stale", {"source": str(p), "reason": why})
            pending.append(p)
            continue
        try:
            from .export import load_session
            rows[str(p)] = _row_from_session(p, dest, load_session(dest),
                                             STATUS_SKIPPED)
        except Exception as exc:                      # noqa: BLE001 - re-render it
            on_event("resume_unreadable", {"source": str(p), "error": str(exc)})
            pending.append(p)
            continue
        on_event("skipped", asdict(rows[str(p)]))

    flow: dict[str, dict] = {}
    if key_strategy == "auto" and pending:
        # A resumed batch has to mix out of what is already on disk. Walk the
        # whole set in order: a finished track is a fixed point in the chain
        # (its rendered key), and each track still to render is planned
        # against whatever really comes before it -- not against the previous
        # *pending* track, which after scattered failures is two or more
        # places back.
        codes = dict(zip((str(p) for p in pending), _source_keys(pending, on_event)))
        previous: str | None = None
        steps = []
        for p in sources:
            done_row = rows.get(str(p))
            if done_row is not None:
                previous = _camelot_or_none(done_row.camelot)
                continue
            step = plan_key_flow([codes[str(p)]], previous=previous)[0]
            flow[str(p)] = step
            steps.append(step)
            if step["key"]:              # an unreadable key leaves the chain as it was
                previous = step["camelot"]
        on_event("key_flow", {"steps": steps})

    def _target_key(p: Path) -> str | None:
        """The key to ask the engine for, or ``None`` to leave the track alone.

        A planned shift of zero is passed as ``None`` rather than as the track's
        own key: asking for the key it already has would let a disagreement
        between this pre-pass and the engine's own analysis turn into a shift
        nobody asked for.
        """
        step = flow.get(str(p))
        if not step or not step.get("shift"):
            return None
        return step.get("key") or None

    jobs_list = [
        {**request, "source": str(p), "output": str(outputs[str(p)]),
         "key": _target_key(p)}
        for p in pending
    ]

    def _record(payload: dict) -> None:
        rows[payload["source"]] = TrackRow(**payload)
        on_event("track", payload)

    if jobs > 1 and len(jobs_list) > 1:
        try:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(_remix_one, j): j for j in jobs_list}
                for fut in as_completed(futures):
                    job = futures[fut]
                    try:
                        _record(fut.result())
                    except Exception as exc:          # noqa: BLE001 - a dead worker
                        _record(asdict(TrackRow(
                            source=job["source"], status=STATUS_FAILED,
                            error=f"worker died: {type(exc).__name__}: {exc}")))
        except Exception as exc:                      # noqa: BLE001 - pool never started
            on_event("pool_failed", {"error": str(exc)})
            for j in jobs_list:
                if j["source"] not in rows:
                    _record(_remix_one(j))
    else:
        for j in jobs_list:
            _record(_remix_one(j))

    ordered = [rows[str(p)] for p in sources if str(p) in rows]
    done = [r for r in ordered if r.status in (STATUS_OK, STATUS_SKIPPED)]
    summary = {
        "total": len(ordered),
        "ok": sum(1 for r in ordered if r.status == STATUS_OK),
        "skipped": sum(1 for r in ordered if r.status == STATUS_SKIPPED),
        "failed": sum(1 for r in ordered if r.status == STATUS_FAILED),
        "set_duration": round(sum(r.duration or 0.0 for r in done), 2),
        "elapsed": round(time.time() - t0, 2),
    }

    exports: dict[str, str] = {}
    # Export exactly this set -- the tracks this run rendered or kept, in the
    # order of the source folder -- rather than every remix in the output
    # folder: an old render of a song since dropped, or of one that failed
    # tonight, must not turn up in the Rekordbox playlist.
    from . import export as export_mod
    tracks, bad = (export_mod.load_tracks([r.output for r in done if r.output],
                                          artist=artist) if done else ([], []))
    notes.extend(bad)
    if tracks:
        try:
            exports["rekordbox"] = str(export_mod.write_rekordbox(tracks, out, set_name))
            exports["csv"] = str(export_mod.write_csv(tracks, out))
        except Exception as exc:                      # noqa: BLE001 - the audio is safe
            notes.append(f"export failed: {type(exc).__name__}: {exc}")
            on_event("export_failed", {"error": str(exc)})
    elif done:
        notes.append("no session file could be read, so no rekordbox.xml or "
                     "cues.csv was written")
    else:
        notes.append("nothing rendered, so no rekordbox.xml or cues.csv was written")

    manifest = {
        "schema": SCHEMA_VERSION,
        "generator": "fourfloor",
        "set": set_name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_folder": str(src_dir),
        "out_folder": str(out),
        "bpm": float(bpm),
        "key_strategy": key_strategy,
        "stems": stems,
        "kit": kit,
        "bass": bass,
        "vocal": vocal,
        "drums_db": float(drums_db),
        "kick_reinforce": bool(kick_reinforce),
        "form": form,
        "length": length,
        "jobs": jobs,
        "resume": bool(resume),
        "tracks": [_jsonable(asdict(r)) for r in ordered],
        "key_flow": [flow[k] for k in (str(p) for p in sources) if k in flow],
        "summary": summary,
        "exports": exports,
        "notes": notes,
    }
    manifest_path = out / "set.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    manifest["manifest"] = str(manifest_path)
    on_event("done", summary)
    return manifest


# ---------------------------------------------------------------------------
# terminal report
# ---------------------------------------------------------------------------

_STATUS_MARK = {STATUS_OK: "✓", STATUS_SKIPPED: "·", STATUS_FAILED: "x"}


def _clip(text: str, n: int) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


class Reporter:
    """Live per-track lines while the batch runs, then the table and the summary.

    Deliberately the same palette and spacing as :mod:`fourfloor.ui`: the batch
    should look like the rest of the tool, not like a second program.
    """

    def __init__(self, c: ui.C, quiet: bool = False) -> None:
        self.c = c
        self.quiet = quiet
        self.total = 0
        self.seen = 0

    def __call__(self, kind: str, payload: dict) -> None:
        if self.quiet:
            return
        c = self.c
        if kind == "start":
            self.total = int(payload.get("tracks", 0))
            tempo = "{:.2f}".format(payload.get("bpm", 0.0))
            print(ui.header(c, "batch: " + str(payload.get("set", ""))))
            print()
            print(ui.kv(c, "tracks", c.bold(str(self.total))))
            print(ui.kv(c, "tempo", c.bold(tempo) + " BPM"))
            print(ui.kv(c, "keys", "shift each track onto the wheel (±2 st)"
                        if payload.get("key_strategy") == "auto"
                        else "left as they are"))
            print(ui.kv(c, "into", c.cyan(str(payload.get("out", "")))))
            print()
        elif kind == "read":
            name = _clip(Path(payload["source"]).name, 40)
            detail = "{:.2f} BPM  {} {}".format(
                payload.get("bpm", 0.0), payload.get("key", ""),
                payload.get("camelot", ""))
            print("  " + c.grey("read") + " " + name.ljust(42) + c.grey(detail))
        elif kind == "read_failed":
            print(ui.warn(c, "could not read {}: {}".format(
                Path(payload["source"]).name, payload.get("error", ""))))
        elif kind == "key_flow":
            self._key_flow(payload.get("steps", []))
        elif kind in ("track", "skipped"):
            self._track_line(payload)
        elif kind == "renamed":
            print(ui.warn(c, str(payload.get("note", ""))))
        elif kind == "resume_stale":
            print("  " + c.grey("redo") + " "
                  + _clip(Path(payload["source"]).name, 40).ljust(42)
                  + c.yellow(_clip(payload.get("reason") or "", 56)))
        elif kind == "export_failed":
            print(ui.warn(c, "export failed: " + str(payload.get("error"))))

    def _key_flow(self, steps: list[dict]) -> None:
        c = self.c
        if not steps:
            return
        if any(s.get("shift") for s in steps) or not all(s.get("compatible")
                                                         for s in steps):
            print()
            print(ui.rule(c, "key flow"))
            for s in steps:
                line = "  {} {} {}  {}".format(
                    s.get("source", "?"), c.grey("->"),
                    c.magenta(s.get("camelot", "?")),
                    c.grey("{:+d} st".format(s.get("shift", 0))))
                if not s.get("compatible"):
                    line += c.yellow("   no compatible key within ±2 st")
                print(line)
        print()

    def _track_line(self, payload: dict) -> None:
        c = self.c
        self.seen += 1
        status = payload.get("status", "?")
        mark = _STATUS_MARK.get(status, "?")
        colour = {STATUS_OK: c.green, STATUS_SKIPPED: c.grey,
                  STATUS_FAILED: c.red}.get(status, c.grey)
        name = _clip(Path(payload["source"]).name, 40)
        if payload.get("bpm"):
            detail = "{:.2f} BPM  {} {}".format(
                payload["bpm"], payload.get("key") or "?",
                payload.get("camelot") or "")
        else:
            detail = payload.get("error") or ""
        counter = "{}/{}".format(self.seen, self.total)
        print("  {} {}{}{}".format(colour(mark), c.grey(counter.ljust(7)),
                                   c.bold(name.ljust(42)),
                                   c.grey(_clip(detail, 56))))

    def report(self, manifest: dict) -> str:
        """The final table and summary, as one string."""
        from .arrange import fmt_time

        c = self.c
        s = manifest["summary"]
        lines = ["", ui.rule(c, "set"), ""]
        lines.append("  " + c.grey("#".ljust(4) + "track".ljust(34)
                                   + "status".ljust(9) + "bpm".ljust(8)
                                   + "key".ljust(9) + "length".ljust(8) + "cues"))
        for i, t in enumerate(manifest["tracks"], start=1):
            colour = {STATUS_OK: c.green, STATUS_SKIPPED: c.grey,
                      STATUS_FAILED: c.red}.get(t["status"], c.grey)
            name = _clip(Path(t["output"] or t["source"]).name, 33)
            head = "  " + str(i).ljust(4) + name.ljust(34)
            if t["status"] == STATUS_FAILED:
                lines.append(head + colour("failed".ljust(9))
                             + c.red(_clip(t.get("error") or "", 44)))
                continue
            lines.append(
                head + colour(t["status"].ljust(9))
                + "{:.2f}".format(t["bpm"] or 0.0).ljust(8)
                + "{} {}".format(t["key"] or "?", t["camelot"] or "").ljust(9)
                + fmt_time(t["duration"] or 0).ljust(8)
                + str(len(t["cues"]))
            )
        lines.append("")
        lines.append(ui.rule(c, "files"))
        for label, path in manifest.get("exports", {}).items():
            lines.append(ui.kv(c, label, c.cyan(str(path))))
        lines.append(ui.kv(c, "set.json", c.cyan(str(manifest.get("manifest", "")))))
        lines.append("")
        for note in manifest.get("notes", []):
            lines.append(ui.warn(c, note))
        if manifest.get("notes"):
            lines.append("")
        bits = [c.green("{} remixed".format(s["ok"]))]
        if s["skipped"]:
            bits.append(c.grey("{} already done".format(s["skipped"])))
        if s["failed"]:
            bits.append(c.red("{} failed".format(s["failed"])))
        lines.append("  " + "  ".join(bits) + "   "
                     + c.grey(fmt_time(s["set_duration"]) + " of music") + "   "
                     + c.grey("in {:.1f}s".format(s["elapsed"])))
        lines.append("")
        return "\n".join(lines)
