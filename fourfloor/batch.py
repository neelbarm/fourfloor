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


def already_done(source: Path, out_dir: Path) -> bool:
    """True when both the mp3 and its session file are already on disk."""
    from .export import session_path_for

    out = output_for(source, out_dir)
    return out.is_file() and out.stat().st_size > 0 and session_path_for(out).is_file()


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

        opts = RemixOptions(
            target_bpm=job["bpm"], key=job.get("key"), stems=job.get("stems", "hpss"),
            form=job.get("form", "club"), length=job.get("length"),
            swing=job.get("swing"), seed=job.get("seed", 0),
            wav=bool(job.get("wav", False)), kit=job.get("kit"),
            bass=job.get("bass", "auto"),
        )
        res = remix(source, out, opts)
        row = _row_from_session(source, Path(res.paths.get("mp3", out)), res.session,
                                STATUS_OK, time.time() - t0)
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


def run(folder: str | Path, out_dir: str | Path, *, bpm: float,
        key_strategy: str = "lock", stems: str = "hpss", kit: str | None = None,
        bass: str = "auto", jobs: int = 1, resume: bool = False,
        length: str | None = None, form: str = "club", swing: float | None = None,
        seed: int = 0, wav: bool = False, set_name: str | None = None,
        artist: str = "fourfloor", on_event=None) -> dict:
    """Remix every track in ``folder`` at ``bpm`` and export the set.

    ``kit`` and ``bass`` are handed to every track unchanged: a set wants one
    drum kit and one low-end policy across it, not a different decision per
    file. Their meanings are :class:`~fourfloor.remix.RemixOptions`'s.

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

    rows: dict[str, TrackRow] = {}
    pending: list[Path] = []
    for p in sources:
        if resume and already_done(p, out):
            dest = output_for(p, out)
            try:
                from .export import load_session
                rows[str(p)] = _row_from_session(p, dest, load_session(dest),
                                                 STATUS_SKIPPED)
            except Exception as exc:                  # noqa: BLE001 - re-render it
                on_event("resume_unreadable", {"source": str(p), "error": str(exc)})
                pending.append(p)
                continue
            on_event("skipped", asdict(rows[str(p)]))
        else:
            pending.append(p)

    flow: dict[str, dict] = {}
    if key_strategy == "auto" and pending:
        # A resumed batch has to mix out of what is already on disk: seed the
        # chain with the key of the last finished track before the first one
        # still to render.
        first = sources.index(pending[0])
        before = [rows[str(p)].camelot for p in sources[:first] if str(p) in rows]
        codes = _source_keys(pending, on_event)
        steps = plan_key_flow(codes, previous=before[-1] if before else None)
        for p, step in zip(pending, steps):
            flow[str(p)] = step
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
        {"source": str(p), "output": str(output_for(p, out)), "bpm": float(bpm),
         "key": _target_key(p), "stems": stems, "form": form, "length": length,
         "swing": swing, "seed": seed, "wav": wav, "kit": kit, "bass": bass}
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
    notes: list[str] = []
    if done:
        from . import export as export_mod
        try:
            result = export_mod.export(out, set_name, formats=("rekordbox", "csv"),
                                       out_dir=out, artist=artist)
            exports = {k: str(v) for k, v in result.files.items()}
            notes.extend(result.notes)
        except Exception as exc:                      # noqa: BLE001 - the audio is safe
            notes.append(f"export failed: {type(exc).__name__}: {exc}")
            on_event("export_failed", {"error": str(exc)})
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
