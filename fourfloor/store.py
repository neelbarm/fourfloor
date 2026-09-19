"""The on-disk library behind the web app: uploaded sources and finished remixes.

Everything lives under ``~/.fourfloor`` (override with ``--home`` or
``FOURFLOOR_HOME``)::

    sources/<id>/source.mp3        the uploaded file, renamed by us
    sources/<id>/meta.json         display name, size, analysis
    remixes/<id>/remix.mp3         + .wav, .session.json, .plan.json, wave.json
    remixes/<id>/meta.json         title, bpm, key, length, options

**No path is ever built from something a browser sent.** Ids are generated
here and must match :data:`ID_RE` to be looked up; the file a download can name
must be one of :data:`REMIX_FILES`. A user's filename is kept only as text in
``meta.json``, for display, and never touches the filesystem.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
from pathlib import Path

#: Ids are our own hex, so anything else -- ``..``, a slash, a NUL -- is simply
#: not an id and is rejected before it can reach the filesystem.
ID_RE = re.compile(r"^[0-9a-f]{16}$")

#: What a browser may upload. ffmpeg reads more, but a short list is a clearer
#: promise and a smaller surface.
UPLOAD_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".aiff", ".aif"}

#: The only names ``GET /api/remixes/<id>/<file>`` can resolve.
REMIX_FILES: dict[str, str] = {
    "remix.mp3": "audio/mpeg",
    "remix.wav": "audio/wav",
    "remix.session.json": "application/json",
    "remix.plan.json": "application/json",
}

MAX_UPLOAD_BYTES = 200 * 1024 * 1024

#: Content types for the source audio the compare view plays back.
AUDIO_TYPES: dict[str, str] = {
    ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav",
    ".flac": "audio/flac", ".aiff": "audio/aiff", ".aif": "audio/aiff",
}

#: Where a human-made remix of a source may be sitting, for A/B against ours.
#: ``<name>.remix.<ext>`` beside the learning pairs; override with
#: ``FOURFLOOR_REFS``.
REFS_ENV = "FOURFLOOR_REFS"


class NotFound(LookupError):
    """No such id in the library."""


def new_id() -> str:
    """A fresh 16-hex-character id."""
    return secrets.token_hex(8)


def safe_id(value: str) -> str:
    """Return ``value`` if it is one of our ids, else raise.

    This is the only gate between a URL path segment and a directory name.
    """
    if not isinstance(value, str) or not ID_RE.match(value):
        raise NotFound(f"not an id: {value!r}")
    return value


def safe_ext(filename: str) -> str:
    """The extension we will give an upload, from the name the browser sent.

    Only the suffix is read, and only to pick from :data:`UPLOAD_EXTS` -- the
    result is one of our own constants, never the user's text.
    """
    ext = Path(str(filename or "")).suffix.lower()
    if ext not in UPLOAD_EXTS:
        raise ValueError(
            f"{ext or 'that file'} is not audio fourfloor reads. "
            "Drop an mp3, m4a, wav, flac or aiff."
        )
    return ext


def display_name(filename: str) -> str:
    """A filename reduced to something safe to show and store as text."""
    name = Path(str(filename or "")).name.replace("\x00", "")
    name = re.sub(r"[\r\n\t]", " ", name).strip()
    return (name or "untitled")[:120]


def title_of(filename: str) -> str:
    """A human title for the library row."""
    stem = Path(display_name(filename)).stem.replace("_", " ").replace("-", " ")
    return stem.strip()[:80] or "Untitled"


def slug(text: str) -> str:
    """A name reduced to letters and digits, for comparing two spellings.

    Only ever used to compare two names to each other -- never to build a path.
    """
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")
    tmp.replace(path)
    return path


class Library:
    """Uploaded sources and finished remixes on disk."""

    def __init__(self, home: str | os.PathLike | None = None) -> None:
        root = home or os.environ.get("FOURFLOOR_HOME") or (Path.home() / ".fourfloor")
        self.home = Path(root).expanduser()
        self.sources = self.home / "sources"
        self.remixes = self.home / "remixes"
        self.uploads = self.home / "uploads"
        for d in (self.sources, self.remixes, self.uploads):
            d.mkdir(parents=True, exist_ok=True)

    # -- sources ----------------------------------------------------------

    def source_dir(self, sid: str) -> Path:
        d = self.sources / safe_id(sid)
        if not d.is_dir():
            raise NotFound(f"no such source: {sid}")
        return d

    def create_source(self, filename: str) -> tuple[str, Path]:
        """Reserve an id and the path the upload will be moved to."""
        ext = safe_ext(filename)
        sid = new_id()
        d = self.sources / sid
        d.mkdir(parents=True, exist_ok=True)
        return sid, d / f"source{ext}"

    def save_source_meta(self, sid: str, meta: dict) -> dict:
        _write_json(self.source_dir(sid) / "meta.json", meta)
        return meta

    def source_meta(self, sid: str) -> dict:
        meta = _read_json(self.source_dir(sid) / "meta.json")
        if not meta:
            raise NotFound(f"no such source: {sid}")
        return meta

    def source_audio(self, sid: str) -> Path:
        d = self.source_dir(sid)
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in UPLOAD_EXTS and f.is_file():
                return f
        raise NotFound(f"source {sid} has no audio")

    def source_file(self, sid: str) -> tuple[Path, str]:
        """The source's audio and its content type, for the compare player."""
        path = self.source_audio(sid)
        return path, AUDIO_TYPES.get(path.suffix.lower(), "application/octet-stream")

    # -- the reference pairs folder ---------------------------------------

    def refs_pairs(self) -> Path:
        root = os.environ.get(REFS_ENV) or (Path.home() / "Music" / "house-refs")
        return Path(root).expanduser() / "pairs"

    def reference_pair(self, name: str) -> tuple[Path, str] | None:
        """A human remix of *name* in the pairs folder, or ``None``.

        ``name`` is text out of a filename somebody typed, so it never becomes
        a path. The folder is listed, each candidate's own stem is reduced to a
        slug, and the match is slug against slug -- the path that comes back is
        one the directory handed us, checked to still be inside it.
        """
        want = slug(name)
        if not want:
            return None
        folder = self.refs_pairs()
        if not folder.is_dir():
            return None
        root = folder.resolve()
        for f in sorted(folder.iterdir()):
            if not f.is_file() or ".remix." not in f.name:
                continue
            if f.suffix.lower() not in AUDIO_TYPES:
                continue
            if slug(f.name.split(".remix.")[0]) != want:
                continue
            resolved = f.resolve()
            if root not in resolved.parents:            # a symlink out of the tree
                continue
            return resolved, AUDIO_TYPES[f.suffix.lower()]
        return None

    # -- remixes ----------------------------------------------------------

    def create_remix(self) -> tuple[str, Path]:
        rid = new_id()
        d = self.remixes / rid
        d.mkdir(parents=True, exist_ok=True)
        return rid, d

    def remix_dir(self, rid: str) -> Path:
        d = self.remixes / safe_id(rid)
        if not d.is_dir():
            raise NotFound(f"no such remix: {rid}")
        return d

    def save_remix_meta(self, rid: str, meta: dict) -> dict:
        meta = dict(meta, id=rid)
        meta.setdefault("created", time.time())
        _write_json(self.remix_dir(rid) / "meta.json", meta)
        return meta

    def remix_meta(self, rid: str) -> dict:
        meta = _read_json(self.remix_dir(rid) / "meta.json")
        if not meta:
            raise NotFound(f"no such remix: {rid}")
        return meta

    def remix_file(self, rid: str, name: str) -> tuple[Path, str]:
        """Resolve one downloadable file of a remix, by allowlist."""
        if name not in REMIX_FILES:
            raise NotFound(f"no such file: {name!r}")
        path = self.remix_dir(rid) / name
        if not path.is_file():
            raise NotFound(f"{name} was not written for this remix")
        return path, REMIX_FILES[name]

    def list_remixes(self) -> list[dict]:
        """Every finished remix, newest first."""
        rows = []
        for d in self.remixes.iterdir():
            if not d.is_dir() or not ID_RE.match(d.name):
                continue
            meta = _read_json(d / "meta.json")
            if meta.get("id") and (d / "remix.mp3").is_file():
                meta["files"] = sorted(n for n in REMIX_FILES if (d / n).is_file())
                rows.append(meta)
        rows.sort(key=lambda m: m.get("created", 0), reverse=True)
        return rows

    def delete_remix(self, rid: str) -> None:
        shutil.rmtree(self.remix_dir(rid), ignore_errors=True)

    # -- housekeeping -----------------------------------------------------

    def sweep_uploads(self, older_than: float = 3600.0) -> int:
        """Delete abandoned upload temporaries; returns how many went."""
        now = time.time()
        gone = 0
        for f in self.uploads.glob("*.part"):
            try:
                if now - f.stat().st_mtime > older_than:
                    f.unlink()
                    gone += 1
            except OSError:
                pass
        return gone
