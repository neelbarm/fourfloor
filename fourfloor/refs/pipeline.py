"""One link in, one filed pair out -- and everything that can go wrong on the way.

The run, per link:

1. **Read the link.** yt-dlp's metadata probe, or the filename if what was
   pasted is a file on this machine. A set or playlist is expanded into its
   tracks.
2. **Parse the title** into artist, track and remixer (:mod:`.titles`).
3. **Fetch the remix** into ``pairs/<slug>.remix.mp3``.
4. **Search for the original** (:mod:`.search`) and download the best one or
   two candidates into a temporary folder.
5. **Verify** each candidate against the remix (:mod:`.verify`) and take the
   first that clears the accept threshold.
6. **File it.** ``pairs/<slug>.original.mp3``, a private sidecar JSON beside it
   with the sources, the match and what the remixer did with the vocal, and a
   drum kit sampled out of the remix.

Failure is per link, never per run: a dead link, a private track, a search that
finds nothing and a record with no clean four-on-the-floor in it each leave one
sentence in the ledger and the next link starts. The ledger is written after
every step, so the whole thing is resumable -- re-running the same command skips
what already finished.

Nothing is left behind that does not have to be. Candidate downloads live in a
temporary folder that is removed whether the verification accepted them or not,
Demucs stems are held in memory for the seconds it takes to measure them, and
what stays on disk is two mp3s, a small JSON, and a kit.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import fetch as fetch_mod
from . import learned, search, titles, verify
from .ledger import SETTLED, Entry, Ledger

#: Between links, and between the searches inside one: enough that forty links
#: do not read as a scraper.
POLITE_SECONDS = 1.0

#: How many of the ranked candidates are actually downloaded and listened to.
CANDIDATES = 2

#: A candidate scoring below this is not worth a download at all.
MIN_CANDIDATE_SCORE = 0.18


class PipelineError(RuntimeError):
    """One link that could not be processed, with a sentence for the user."""


@dataclass
class Options:
    """Everything the run can be told to do differently."""

    home: Path = field(default_factory=lambda: fetch_mod.REFS_HOME)
    dry_run: bool = False
    kit: bool = True
    demucs: bool = True
    candidates: int = CANDIDATES
    per_query: int = search.PER_QUERY
    pause: float = POLITE_SECONDS
    force: bool = False
    timeout: float = fetch_mod.DEFAULT_TIMEOUT
    kit_home: str | Path | None = None

    @property
    def pairs(self) -> Path:
        return Path(self.home) / "pairs"

    @property
    def remixes(self) -> Path:
        return Path(self.home) / "remixes"

    @property
    def cache(self) -> Path:
        return Path(self.home) / ".cache"

    @property
    def ledger_path(self) -> Path:
        return Path(self.home) / "ledger.json"


def _noop(*_a, **_k) -> None:
    return None


# ---------------------------------------------------------------------------
# reading what was pasted
# ---------------------------------------------------------------------------

def is_local(source: str) -> bool:
    """Whether what was pasted is a file on this machine rather than a link."""
    text = str(source or "").strip().strip('"\'')
    if text.lower().startswith(("http://", "https://")):
        return False
    try:
        return Path(text).expanduser().is_file()
    except OSError:
        return False


def read_links(path: str | Path) -> list[str]:
    """Links from a text file: one per line, ``#`` comments and blanks skipped."""
    text = Path(path).expanduser().read_text(encoding="utf8", errors="replace")
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split()[0] if line.startswith("http") else line)
    return out


def _describe(source: str, opts: Options) -> tuple[dict, str, str]:
    """``(info, title, uploader)`` for a link or a local file."""
    if is_local(source):
        path = Path(str(source).strip().strip('"\'')).expanduser()
        return {"local": str(path)}, path.stem, ""
    url = fetch_mod.check_url(source)
    info = fetch_mod.probe(url, timeout=min(opts.timeout, fetch_mod.PROBE_TIMEOUT))
    title = str(info.get("title") or "").strip()
    uploader = str(info.get("uploader") or info.get("channel")
                   or info.get("artist") or "").strip()
    return info, title, uploader


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def add(sources: list[str], opts: Options | None = None, on=None) -> list[Entry]:
    """Ingest every link (or local file), one at a time, isolating failures."""
    opts = opts or Options()
    note = on or _noop
    led = Ledger(opts.ledger_path)
    out: list[Entry] = []
    for i, source in enumerate(sources):
        if i:
            time.sleep(max(0.0, opts.pause))
        try:
            out.extend(add_one(source, opts, led, on=note))
        except KeyboardInterrupt:
            raise
        except Exception as exc:                       # one bad link, not a bad run
            entry = Entry(url=str(source), status="failed", error=_sentence(exc))
            if not opts.dry_run:
                led.put(entry)
            note("error", f"{source}: {entry.error}")
            out.append(entry)
    return out


def _sentence(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text if len(text) < 400 else text[:397] + "…"


def add_one(source: str, opts: Options, led: Ledger, on=None) -> list[Entry]:
    """Ingest one pasted thing, expanding a playlist or set into its tracks."""
    note = on or _noop
    source = str(source).strip().strip('"\'')
    existing = led.by_url(source)
    if existing is not None and existing.status in SETTLED and not opts.force:
        note("skip", f"{existing.slug or source} — already {existing.status}")
        return [existing]

    note("link", source)
    info, title, uploader = _describe(source, opts)
    if info.get("local") is None and fetch_mod.is_playlist(info):
        entries = fetch_mod.entries_of(info)
        note("step", f"a set of {len(entries)} tracks — taking them one at a time")
        out: list[Entry] = []
        for i, sub in enumerate(entries):
            link = str(sub.get("url") or sub.get("webpage_url") or "")
            if not link:
                continue
            if i:
                time.sleep(max(0.0, opts.pause))
            try:
                out.extend(add_one(link, opts, led, on=note))
            except Exception as exc:
                bad = Entry(url=link, status="failed", error=_sentence(exc))
                led.put(bad)
                note("error", f"{link}: {bad.error}")
                out.append(bad)
        return out
    return [_ingest(source, info, title, uploader, opts, led, note)]


def _ingest(source: str, info: dict, title: str, uploader: str,
            opts: Options, led: Ledger, note) -> Entry:
    parsed = titles.parse(title, uploader)
    note("step", f"parsed: {parsed.label()}")

    entry = led.by_url(source) or Entry(url=source)
    entry.status = "pending"
    entry.title, entry.uploader = title, uploader
    entry.site = "local file" if info.get("local") else fetch_mod.site_of(source)
    entry.artist, entry.track = parsed.artist, parsed.track
    entry.remixer, entry.kind = parsed.remixer, parsed.kind
    entry.error = ""
    if not entry.slug:
        entry.slug = led.free_slug(parsed.slug(), taken=_slugs_on_disk(opts))

    # --- the candidates --------------------------------------------------
    note("step", "looking for the original")
    candidates = search.find_original(
        parsed, per_query=opts.per_query, pause=opts.pause,
        on_note=lambda text: note("step", text))
    entry.candidates = [c.to_dict() for c in candidates[:5]]

    if opts.dry_run:
        best = candidates[0] if candidates else None
        entry.status = "dry-run"
        entry.original_url = best.url if best else ""
        entry.original_title = best.title if best else ""
        entry.score = best.score if best else 0.0
        note("dry", _dry_line(entry, parsed, best))
        return entry

    led.put(entry)

    # --- the remix -------------------------------------------------------
    remix_path = Path(entry.remix_path) if entry.remix_path else None
    if remix_path is None or not remix_path.is_file():
        remix_path = _get_remix(source, info, entry.slug, opts, note)
        entry.remix_path = str(remix_path)
        led.put(entry)

    # --- verify, best candidate first ------------------------------------
    accepted = None
    best_seen: tuple[verify.Match, search.Candidate, Path] | None = None
    tmp = Path(tempfile.mkdtemp(prefix="fourfloor-refs-"))
    try:
        for cand in candidates[:max(1, opts.candidates)]:
            if cand.score < MIN_CANDIDATE_SCORE:
                continue
            note("step", f"candidate: {cand.title[:60]} ({cand.score:.2f})")
            try:
                got = fetch_mod.fetch(cand.url, tmp, name=f"cand-{len(list(tmp.iterdir()))}",
                                      timeout=opts.timeout)
            except fetch_mod.FetchError as exc:
                note("warn", f"could not fetch that candidate: {exc}")
                continue
            try:
                match, orig_fp, remix_fp = _compare(got.path, remix_path, opts, note)
            except verify.VerifyError as exc:
                note("warn", f"could not compare that candidate: {exc}")
                continue
            note("step", f"score {match.score:.3f} (margin {match.margin:.3f}) "
                         f"— {match.verdict}")
            if best_seen is None or match.score > best_seen[0].score:
                best_seen = (match, cand, got.path)
            if match.verdict == "match":
                accepted = (match, cand, got.path, orig_fp, remix_fp)
                break

        if accepted is not None:
            match, cand, path, orig_fp, remix_fp = accepted
            _file_pair(entry, match, cand, path, orig_fp, remix_fp, opts, note)
        elif best_seen is not None and best_seen[0].verdict == "needs_review":
            match, cand, _path = best_seen
            entry.status = "needs_review"
            _stamp_match(entry, match, cand)
            note("warn", f"{entry.slug}: {match.score:.3f} is between the "
                         f"thresholds — `fourfloor refs review` to decide")
        else:
            _keep_standalone(entry, opts, note, best_seen)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if opts.kit and entry.status in ("done", "standalone"):
        entry.kit = _build_kit(Path(entry.remix_path), entry.slug, opts, note)
    led.put(entry)
    return entry


def _dry_line(entry: Entry, parsed: titles.Parsed, best) -> str:
    would = (f"would fetch: {best.title} ({best.score:.2f}) {best.url}"
             if best else "found nothing worth fetching")
    return f"{parsed.label()}  ->  {entry.slug}\n    {would}"


def _slugs_on_disk(opts: Options) -> set[str]:
    """Names already taken by files in the reference folder."""
    out: set[str] = set()
    for folder, suffixes in ((opts.pairs, (".remix", ".original")),
                             (opts.remixes, ("",))):
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            stem = path.stem
            for suffix in suffixes:
                if suffix and stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
            out.add(stem)
    return out


def _get_remix(source: str, info: dict, slug: str, opts: Options, note) -> Path:
    """The remix itself, in ``pairs/<slug>.remix.mp3``."""
    opts.pairs.mkdir(parents=True, exist_ok=True)
    local = info.get("local")
    if local:
        target = fetch_mod.free_path(opts.pairs, f"{slug}.remix",
                                     Path(local).suffix.lower() or ".mp3")
        note("step", f"copying {Path(local).name}")
        shutil.copy2(local, target)
        return target
    note("step", "downloading the remix")
    got = fetch_mod.fetch(source, opts.pairs, name=slug, kind="remix",
                          timeout=opts.timeout, info=info)
    return got.path


def _compare(original: Path, remix: Path, opts: Options, note):
    """Fingerprint both and score them. The remix's features are cached."""
    remix_fp = verify.fingerprint(remix, kind="remix", demucs=opts.demucs,
                                  cache_dir=opts.cache,
                                  progress=lambda *a: note("step", " ".join(map(str, a))))
    orig_fp = verify.fingerprint(original, kind="original", demucs=opts.demucs,
                                 progress=lambda *a: note("step", " ".join(map(str, a))))
    return verify.compare(orig_fp, remix_fp), orig_fp, remix_fp


def _stamp_match(entry: Entry, match: verify.Match, cand: search.Candidate) -> None:
    entry.score = round(float(match.score), 4)
    entry.verdict = match.verdict
    entry.method = match.method
    entry.tempo_ratio = round(float(match.tempo_ratio), 4)
    entry.semitones = int(match.semitones)
    entry.bpm_original = round(float(match.bpm_original), 2)
    entry.bpm_remix = round(float(match.bpm_remix), 2)
    entry.original_url = cand.url
    entry.original_title = cand.title


def _file_pair(entry: Entry, match, cand, path: Path, orig_fp, remix_fp,
               opts: Options, note) -> None:
    """Move the verified original into place and write the sidecar."""
    target = fetch_mod.free_path(opts.pairs, f"{entry.slug}.original", ".mp3")
    shutil.move(str(path), str(target))
    entry.original_path = str(target)
    entry.status = "done"
    _stamp_match(entry, match, cand)

    vocal = learned.treatment_of(orig_fp, remix_fp, match.tempo_ratio)
    vocal["bpm_remix"] = round(float(match.bpm_remix), 2)
    entry.lattice = vocal["original_lattice"]
    entry.treatment = vocal["treatment"]
    sidecar = {
        "schema": 1,
        "slug": entry.slug,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "remix": {"url": entry.url, "title": entry.title,
                  "uploader": entry.uploader, "site": entry.site,
                  "file": Path(entry.remix_path).name if entry.remix_path else "",
                  "artist": entry.artist, "track": entry.track,
                  "remixer": entry.remixer, "kind": entry.kind},
        "original": {"url": cand.url, "title": cand.title,
                     "uploader": cand.uploader, "file": target.name,
                     "search_query": cand.query,
                     "rank_score": round(float(cand.score), 4)},
        "match": match.to_dict(),
        "vocal": vocal,
        "excerpts": {"original": orig_fp.to_meta(), "remix": remix_fp.to_meta()},
    }
    (opts.pairs / f"{entry.slug}.json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf8")
    note("ok", f"{entry.slug}: {cand.title[:52]} — {match.score:.3f}")


def _keep_standalone(entry: Entry, opts: Options, note, best_seen=None) -> None:
    """No original: the remix still goes to ``remixes/`` and still makes a kit."""
    remix = Path(entry.remix_path) if entry.remix_path else None
    if remix is not None and remix.is_file() and remix.parent == opts.pairs:
        target = fetch_mod.free_path(opts.remixes, entry.slug, remix.suffix or ".mp3")
        shutil.move(str(remix), str(target))
        entry.remix_path = str(target)
    entry.status = "standalone"
    if best_seen is not None:
        entry.score = round(float(best_seen[0].score), 4)
        entry.verdict = best_seen[0].verdict
        entry.original_title = best_seen[1].title
        entry.original_url = best_seen[1].url
    note("warn", f"{entry.slug}: no original verified — kept as a standalone remix")


def _build_kit(remix: Path, slug: str, opts: Options, note) -> str:
    """Sample a kit out of the remix, or say why there is none."""
    from .. import kit as kit_mod

    name = slug if kit_mod.NAME_RE.match(slug) else kit_mod.slugify(slug)
    try:
        note("step", f"building a kit from {remix.name}")
        built = kit_mod.build(remix, name=name, home=opts.kit_home)
    except KeyboardInterrupt:
        raise
    except (RuntimeError, ValueError, OSError) as exc:
        note("warn", f"no kit from this one: {_sentence(exc)}")
        return ""
    note("ok", f"kit {built.name} ({built.bars} bars, score {built.score:.3f})")
    return built.name


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

def accept(slug: str, opts: Options | None = None, on=None,
           url: str | None = None) -> Entry:
    """File a ``needs_review`` pair: fetch its candidate again and keep it.

    The candidate's download was thrown away when the run finished -- keeping
    two mp3s per undecided link would fill a disk -- so accepting fetches it
    once more and re-measures, which also means the sidecar says what the
    numbers were at the moment it was filed.
    """
    opts = opts or Options()
    note = on or _noop
    led = Ledger(opts.ledger_path)
    entry = led.by_slug(slug)
    if entry is None:
        raise PipelineError(f"nothing called {slug!r} has been ingested")
    link = url or entry.original_url
    if not link:
        raise PipelineError(f"{slug} has no candidate to accept")
    remix = Path(entry.remix_path)
    if not remix.is_file():
        raise PipelineError(f"the remix for {slug} is not where the ledger says it is")

    tmp = Path(tempfile.mkdtemp(prefix="fourfloor-refs-"))
    try:
        note("step", f"fetching {link}")
        got = fetch_mod.fetch(link, tmp, name="accepted", timeout=opts.timeout)
        match, orig_fp, remix_fp = _compare(got.path, remix, opts, note)
        cand = search.Candidate(url=link, title=got.title or entry.original_title,
                                uploader=got.uploader, duration=got.duration,
                                score=entry.score)
        _file_pair(entry, match, cand, got.path, orig_fp, remix_fp, opts, note)
        entry.verdict = "accepted by hand"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if opts.kit and not entry.kit:
        entry.kit = _build_kit(remix, entry.slug, opts, note)
    led.put(entry)
    return entry


def reject(slug: str, opts: Options | None = None, on=None) -> Entry:
    """Say no: the remix becomes a standalone reference, kit and all."""
    opts = opts or Options()
    note = on or _noop
    led = Ledger(opts.ledger_path)
    entry = led.by_slug(slug)
    if entry is None:
        raise PipelineError(f"nothing called {slug!r} has been ingested")
    _keep_standalone(entry, opts, note)
    entry.verdict = "rejected by hand"
    entry.original_url = entry.original_title = ""
    if opts.kit and not entry.kit and entry.remix_path:
        entry.kit = _build_kit(Path(entry.remix_path), entry.slug, opts, note)
    led.put(entry)
    return entry


# ---------------------------------------------------------------------------
# learning
# ---------------------------------------------------------------------------

def sidecars(opts: Options) -> list[dict]:
    """Every per-pair JSON in the reference folder, newest last."""
    out = []
    if not opts.pairs.is_dir():
        return out
    for path in sorted(opts.pairs.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("vocal"):
            out.append(data)
    return out


def learn(opts: Options | None = None, on=None, repo_out: str | Path | None = None,
          home_out: str | Path | None = None) -> dict:
    """Re-measure every reference and write the profile the engine reads.

    Two files come out. ``~/.fourfloor/style.json`` is the private one: the
    aggregate numbers plus the vocal table, and it is what the engine picks up
    as its default style. The second is the anonymised copy for the repository
    -- the same aggregates with the per-file rows stripped out, so what gets
    committed is arithmetic rather than anybody's record collection.
    """
    from ..audio import find_audio
    from ..style import learn as style_learn

    opts = opts or Options()
    note = on or _noop
    files: list[Path] = []
    for folder in (opts.pairs, opts.remixes):
        if folder.is_dir():
            # the *remix* halves only. The style being learned is the one to
            # imitate, and the originals are pop records at pop tempi -- letting
            # them into the median would teach the engine to aim at 120 because
            # half its references are not house at all.
            files.extend(f for f in find_audio(folder)
                         if not f.name.lower().endswith((".original.mp3",
                                                         ".original.wav",
                                                         ".original.m4a",
                                                         ".original.flac")))
    if not files:
        raise PipelineError(f"no reference audio in {opts.home}")

    style = style_learn(opts.home, progress=lambda *a: note("step", " ".join(map(str, a))),
                        files=files)
    rows = [dict(s["vocal"], slug=s.get("slug", "")) for s in sidecars(opts)]
    table = learned.table(rows)

    private = style.to_dict()
    private["vocal"] = table
    private["n_pairs"] = table["n_pairs"]
    ratios = [float(r["tempo_ratio"]) for r in rows if r.get("tempo_ratio")]
    if ratios:
        import numpy as np
        private["tempo_ratio"] = round(float(np.median(ratios)), 4)
    home_path = Path(home_out) if home_out else learned.style_path(opts.kit_home)
    home_path.parent.mkdir(parents=True, exist_ok=True)
    home_path.write_text(json.dumps(private, indent=2) + "\n", encoding="utf8")
    learned.forget()
    note("ok", f"style profile: {home_path}")

    public = None
    if repo_out:
        public = dict(private)
        public.pop("per_track", None)
        # the table's cells are counts and medians; the slugs never travel
        public["vocal"] = {k: v for k, v in table.items()}
        repo_path = Path(repo_out)
        merged = {}
        if repo_path.is_file():
            try:
                merged = json.loads(repo_path.read_text(encoding="utf8"))
            except (OSError, ValueError):
                merged = {}
        merged.update(public)
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        repo_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf8")
        note("ok", f"anonymised aggregate: {repo_path}")
        public = merged
    return {"style": private, "public": public, "table": table,
            "n_files": len(files), "n_pairs": table["n_pairs"]}
