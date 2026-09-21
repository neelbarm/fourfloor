"""Take a finished remix to the booth: Rekordbox XML, ID3 tags, cue sheets.

A remix is only useful at a gig if the DJ software already knows its tempo, its
key and where the drop is. ``*.session.json`` has all of that -- this module is
the translator between that file and the three things a DJ actually loads:

* **rekordbox.xml** -- the ``DJ_PLAYLISTS`` document Rekordbox imports under
  *Preferences > Advanced > rekordbox xml*. One ``TRACK`` per remix with a
  ``TEMPO`` beat-grid anchor and ``POSITION_MARK`` hot cues A-H, gathered into a
  playlist node named after the set.
* **ID3 tags on the mp3 itself** -- TBPM, TKEY, TIT2, TPE1, a COMM cue list and
  the ``TXXX:INITIALKEY`` frame Rekordbox and Serato both read.
* **cues.csv** -- a plain sheet for anything else (Traktor, a spreadsheet, a
  printout taped to the mixer).

Every format here is written with the standard library alone -- there is no
mutagen in this project and ffmpeg ``-metadata`` would remux the file -- and the
ID3 writer only ever replaces the tag at the head of the mp3: the MPEG frames
are copied through byte for byte, so the audio cannot be damaged by tagging.
:func:`probe` re-checks that with ffprobe anyway.

Serato's hot cues live in a ``Serato Markers2`` GEOB frame whose format is not
published by Serato; the encoder here follows the community reverse-engineering
(see :data:`SERATO_REFERENCE`) and is round-tripped by the test suite, but it
has **not** been confirmed against a real Serato DJ install. Treat
``--format serato`` as best effort and check one track before a gig.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import struct
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import quote

#: Where the Serato GEOB layout below comes from. Serato publishes no spec.
SERATO_REFERENCE = "https://github.com/Holzhaus/serato-tags (reverse-engineered)"

#: Rekordbox writes an mp3's sample rate into the collection; every fourfloor
#: render is 44.1 kHz stereo, and the session file does not carry it.
DEFAULT_SR = 44100
DEFAULT_BITRATE = 320

#: The suffix ``--suffix`` appends to the title frame.
TITLE_SUFFIX = " (fourfloor house remix)"

GENERATOR = "fourfloor"

#: A Rekordbox deck shows eight hot cue buttons, A to H.
MAX_HOT_CUES = 8
HOT_CUE_LETTERS = "ABCDEFGH"

#: Hot cue colours, by section kind, in Rekordbox's Red/Green/Blue attributes.
#: These are the stock Rekordbox swatches, so an imported set looks native.
CUE_COLORS: dict[str, tuple[int, int, int]] = {
    "intro": (40, 226, 20),         # green: safe to mix in
    "build": (224, 100, 27),        # orange: something is coming
    "drop": (222, 68, 207),         # pink: the moment
    "breakdown": (48, 90, 255),     # blue: the air pocket
    "outro": (195, 175, 4),         # yellow: start mixing out
}
DEFAULT_CUE_COLOR = (40, 226, 20)

FORMATS = ("rekordbox", "serato", "csv", "tags")
#: What ``fourfloor export`` does when no ``--format`` is given: the two files
#: that cannot damage anything. ``tags`` and ``serato`` rewrite the mp3, so they
#: are opt-in.
DEFAULT_FORMATS = ("rekordbox", "csv")

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aiff", ".aif", ".flac"}


class ExportError(RuntimeError):
    """Something a person can fix: a missing session file, an unreadable mp3."""


# ---------------------------------------------------------------------------
# the session file, as the exporters want to see it
# ---------------------------------------------------------------------------

def session_path_for(audio: str | Path) -> Path:
    """The ``*.session.json`` that belongs beside ``audio``.

    ``lofi-7.house.mp3`` -> ``lofi-7.house.session.json``.
    """
    audio = Path(audio)
    return audio.with_suffix(".session.json")


def load_session(audio: str | Path) -> dict:
    """Read the session file beside ``audio``, with a message that says where."""
    path = session_path_for(audio)
    if not path.is_file():
        raise ExportError(
            f"no session file beside {Path(audio).name} (expected {path.name}); "
            f"`fourfloor remix` writes one next to every output"
        )
    try:
        data = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"{path.name} is not readable session JSON: {exc}") from exc
    if not isinstance(data, dict) or "cues" not in data:
        raise ExportError(f"{path.name} does not look like a fourfloor session file")
    return data


@dataclass
class Cue:
    """One hot cue: what it is called, where it is, which bar it lands on."""

    name: str
    seconds: float
    bar: int
    kind: str

    @property
    def color(self) -> tuple[int, int, int]:
        return CUE_COLORS.get(self.kind, DEFAULT_CUE_COLOR)

    @property
    def millis(self) -> int:
        return int(round(self.seconds * 1000.0))


@dataclass
class Track:
    """A remix plus everything the exporters need to describe it."""

    path: Path
    session: dict
    title: str
    artist: str = GENERATOR
    genre: str = "House"

    @property
    def bpm(self) -> float:
        return float(self.session.get("bpm", 0.0))

    @property
    def first_downbeat(self) -> float:
        return float(self.session.get("first_downbeat_sec", 0.0))

    @property
    def key(self) -> str:
        return str(self.session.get("key", ""))

    @property
    def camelot(self) -> str:
        return str(self.session.get("camelot", ""))

    @property
    def duration(self) -> float:
        return float(self.session.get("duration", 0.0))

    @property
    def cues(self) -> list[Cue]:
        """Every structural cue in the session, ``end`` included, in time order."""
        out = [
            Cue(name=str(c.get("name", c.get("kind", "cue"))),
                seconds=float(c.get("time", 0.0)),
                bar=int(c.get("bar", 0)),
                kind=str(c.get("kind", "")))
            for c in self.session.get("cues", [])
        ]
        return sorted(out, key=lambda c: c.seconds)

    @property
    def hot_cues(self) -> list[Cue]:
        """The cues that get a button: everything but ``end``, capped at eight."""
        return [c for c in self.cues if c.kind != "end"][:MAX_HOT_CUES]

    def comment(self) -> str:
        """The one-line cue list that goes in COMM and in Rekordbox's Comments."""
        bits = [f"{self.bpm:.2f} BPM", f"{self.key} {self.camelot}".strip()]
        bits += [f"{HOT_CUE_LETTERS[i]} {c.name} {fmt_clock(c.seconds)}"
                 for i, c in enumerate(self.hot_cues)]
        return " | ".join(b for b in bits if b)


def fmt_clock(seconds: float) -> str:
    """Seconds as ``M:SS``, the way a cue sheet reads."""
    m, s = divmod(int(round(float(seconds))), 60)
    return f"{m}:{s:02d}"


_HOUSE_STEM = re.compile(r"\.(house|remix)$", re.IGNORECASE)


def title_from_path(path: Path) -> str:
    """A human title from a filename: ``Some Song.house.mp3`` -> ``Some Song``."""
    return _HOUSE_STEM.sub("", path.stem).strip() or path.stem


def collect(target: str | Path, artist: str = GENERATOR,
            suffix: bool = False) -> list[Track]:
    """Gather tracks from one audio file or every remix in a folder.

    A folder is scanned for audio files that have a session file beside them;
    anything else in the folder is ignored rather than refused, because an
    output folder also holds ``.wav``, ``.plan.json`` and ``preview.html``.
    """
    target = Path(target)
    if target.is_dir():
        candidates = sorted(
            p for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
            and not p.name.startswith(".") and session_path_for(p).is_file()
        )
        # a remix is written as both .mp3 and .wav; the mp3 is the one a DJ loads
        stems = {p.with_suffix("") for p in candidates if p.suffix.lower() == ".mp3"}
        candidates = [p for p in candidates
                      if p.suffix.lower() == ".mp3" or p.with_suffix("") not in stems]
        if not candidates:
            raise ExportError(
                f"no remixes with a session file in {target} -- "
                f"`fourfloor batch` or `fourfloor remix` writes one beside each output"
            )
    elif target.is_file():
        candidates = [target]
    else:
        raise FileNotFoundError(str(target))

    tracks = []
    for p in candidates:
        sess = load_session(p)
        name = title_from_path(p)
        tracks.append(Track(path=p, session=sess,
                            title=name + (TITLE_SUFFIX if suffix else ""),
                            artist=artist))
    return tracks


# ---------------------------------------------------------------------------
# Rekordbox XML
# ---------------------------------------------------------------------------

def file_url(path: str | Path) -> str:
    """The ``Location`` string Rekordbox wants: a percent-encoded file URL.

    Rekordbox writes ``file://localhost`` followed by the absolute path with
    every reserved byte percent-encoded -- spaces become ``%20`` and non-ASCII
    becomes its UTF-8 bytes. Getting this wrong is the single most common reason
    an imported collection shows every track in red as "file missing".
    """
    p = Path(path).expanduser()
    p = p if p.is_absolute() else (Path.cwd() / p)
    # resolve() would follow symlinks out from under the person's own library
    text = os.path.normpath(str(p))
    return "file://localhost" + quote(text, safe="/")


def _track_element(track: Track, track_id: int) -> ET.Element:
    el = ET.Element("TRACK", {
        "TrackID": str(track_id),
        "Name": track.title,
        "Artist": track.artist,
        "Composer": "",
        "Album": "",
        "Grouping": "",
        "Genre": track.genre,
        "Kind": "MP3 File" if track.path.suffix.lower() == ".mp3" else "WAV File",
        "Size": str(track.path.stat().st_size if track.path.is_file() else 0),
        "TotalTime": str(int(round(track.duration))),
        "DiscNumber": "0",
        "TrackNumber": str(track_id),
        "Year": str(date.today().year),
        "AverageBpm": f"{track.bpm:.2f}",
        "DateAdded": date.today().isoformat(),
        "BitRate": str(DEFAULT_BITRATE),
        "SampleRate": str(DEFAULT_SR),
        "Comments": track.comment(),
        "PlayCount": "0",
        "Rating": "0",
        "Location": file_url(track.path),
        "Remixer": GENERATOR,
        "Tonality": track.key,
        "Label": "",
        "Mix": "",
    })
    # The beat grid. The remix is synthesised on an exact grid, so one anchor at
    # the first downbeat describes the whole track -- no drifting second marker.
    ET.SubElement(el, "TEMPO", {
        "Inizio": f"{track.first_downbeat:.3f}",
        "Bpm": f"{track.bpm:.2f}",
        "Metro": "4/4",
        "Battito": "1",
    })
    # Memory cue at the first downbeat: Num="-1" is how Rekordbox marks a
    # memory cue rather than one of the eight hot cue buttons.
    ET.SubElement(el, "POSITION_MARK", {
        "Name": "first downbeat",
        "Type": "0",
        "Start": f"{track.first_downbeat:.3f}",
        "Num": "-1",
    })
    for i, cue in enumerate(track.hot_cues):
        r, g, b = cue.color
        ET.SubElement(el, "POSITION_MARK", {
            "Name": cue.name,
            "Type": "0",                       # 0 = cue point (1 = fade-in, 4 = loop)
            "Start": f"{cue.seconds:.3f}",
            "Num": str(i),                     # 0 = hot cue A
            "Red": str(r), "Green": str(g), "Blue": str(b),
        })
    return el


def rekordbox_xml(tracks: list[Track], set_name: str, version: str = "0.1.0") -> str:
    """Build the whole ``DJ_PLAYLISTS`` document as a string."""
    root = ET.Element("DJ_PLAYLISTS", {"Version": "1.0.0"})
    ET.SubElement(root, "PRODUCT", {
        "Name": GENERATOR, "Version": version, "Company": GENERATOR,
    })
    collection = ET.SubElement(root, "COLLECTION", {"Entries": str(len(tracks))})
    for i, t in enumerate(tracks, start=1):
        collection.append(_track_element(t, i))

    playlists = ET.SubElement(root, "PLAYLISTS")
    rootnode = ET.SubElement(playlists, "NODE", {
        "Type": "0", "Name": "ROOT", "Count": "1",
    })
    node = ET.SubElement(rootnode, "NODE", {
        "Name": set_name, "Type": "1", "KeyType": "0", "Entries": str(len(tracks)),
    })
    for i in range(1, len(tracks) + 1):
        ET.SubElement(node, "TRACK", {"Key": str(i)})

    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def write_rekordbox(tracks: list[Track], out_dir: str | Path, set_name: str,
                    filename: str = "rekordbox.xml") -> Path:
    """Write the collection to ``<out_dir>/rekordbox.xml`` and return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(rekordbox_xml(tracks, set_name), encoding="utf8")
    return path


# ---------------------------------------------------------------------------
# cues.csv
# ---------------------------------------------------------------------------

CSV_HEADER = ("file", "cue", "seconds", "bar", "time", "kind", "hot_cue")


def cues_csv(tracks: list[Track]) -> str:
    """The generic cue sheet: one row per cue, every track in one file."""
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_HEADER)
    for t in tracks:
        cues = t.cues
        # the properties rebuild their Cue objects on every access, so pair the
        # hot cue letters to positions in this one list rather than to identity
        hot = 0
        for c in cues:
            letter = ""
            if c.kind != "end" and hot < MAX_HOT_CUES:
                letter = HOT_CUE_LETTERS[hot]
                hot += 1
            w.writerow([t.path.name, c.name, f"{c.seconds:.3f}", c.bar,
                        fmt_clock(c.seconds), c.kind, letter])
    return buf.getvalue()


def write_csv(tracks: list[Track], out_dir: str | Path,
              filename: str = "cues.csv") -> Path:
    """Write the cue sheet to ``<out_dir>/cues.csv`` and return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(cues_csv(tracks), encoding="utf8")
    return path


# ---------------------------------------------------------------------------
# ID3v2.3
# ---------------------------------------------------------------------------
#
# Written by hand rather than with mutagen, which is not a dependency of this
# project, and rather than with ffmpeg ``-metadata``, which would re-encode or
# at least remux the audio. The only bytes that move are the tag at the head of
# the file; every MPEG frame after it is copied verbatim.

def _synchsafe(n: int) -> bytes:
    if not 0 <= n < (1 << 28):
        raise ValueError(f"tag too large for a synchsafe size: {n}")
    return bytes(((n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F))


def _unsynchsafe(b: bytes) -> int:
    return (b[0] << 21) | (b[1] << 14) | (b[2] << 7) | b[3]


def _encode_text(s: str) -> tuple[int, bytes, bytes]:
    """``(encoding byte, encoded text, terminator)`` for an ID3v2.3 string.

    ISO-8859-1 when it fits -- it is what every reader handles without thinking
    -- and UTF-16 with a byte order mark when it does not, which is the only
    other encoding ID3v2.3 allows.
    """
    try:
        return 0x00, s.encode("latin-1"), b"\x00"
    except UnicodeEncodeError:
        return 0x01, b"\xff\xfe" + s.encode("utf-16-le"), b"\x00\x00"


def _decode_text(enc: int, data: bytes) -> str:
    if enc == 0x00:
        return data.decode("latin-1")
    if enc == 0x01:
        return data.decode("utf-16", errors="replace")
    if enc == 0x02:
        return data.decode("utf-16-be", errors="replace")
    return data.decode("utf8", errors="replace")


def _split_term(enc: int, data: bytes) -> tuple[bytes, bytes]:
    """Split ``data`` at the first terminator for ``enc``; returns (head, rest)."""
    if enc in (0x01, 0x02):
        for i in range(0, len(data) - 1, 2):
            if data[i] == 0 and data[i + 1] == 0:
                return data[:i], data[i + 2:]
        return data, b""
    i = data.find(b"\x00")
    return (data, b"") if i < 0 else (data[:i], data[i + 1:])


def _frame(frame_id: str, body: bytes) -> bytes:
    return frame_id.encode("ascii") + struct.pack(">I", len(body)) + b"\x00\x00" + body


def text_frame(frame_id: str, value: str) -> bytes:
    """A ``T***`` text frame -- TIT2, TBPM, TKEY and the rest."""
    enc, data, _ = _encode_text(value)
    return _frame(frame_id, bytes([enc]) + data)


def comment_frame(value: str, description: str = "", lang: str = "eng") -> bytes:
    """A ``COMM`` frame: language, a short description, then the comment."""
    enc, data, term = _encode_text(value)
    denc, ddata, _ = _encode_text(description)
    if denc != enc:                       # both strings share one encoding byte
        enc, data, term = 0x01, b"\xff\xfe" + value.encode("utf-16-le"), b"\x00\x00"
        ddata = b"\xff\xfe" + description.encode("utf-16-le")
    return _frame("COMM", bytes([enc]) + lang.encode("ascii")[:3] + ddata + term + data)


def txxx_frame(description: str, value: str) -> bytes:
    """A ``TXXX`` user-defined text frame, keyed by its description."""
    enc, data, term = _encode_text(value)
    denc, ddata, _ = _encode_text(description)
    if denc != enc:
        enc, data, term = 0x01, b"\xff\xfe" + value.encode("utf-16-le"), b"\x00\x00"
        ddata = b"\xff\xfe" + description.encode("utf-16-le")
    return _frame("TXXX", bytes([enc]) + ddata + term + data)


def geob_frame(description: str, data: bytes,
               mime: str = "application/octet-stream", filename: str = "") -> bytes:
    """A General Encapsulated Object frame -- how Serato hides its cue blobs."""
    body = (b"\x00" + mime.encode("latin-1") + b"\x00"
            + filename.encode("latin-1") + b"\x00"
            + description.encode("latin-1") + b"\x00" + data)
    return _frame("GEOB", body)


def id3v23_tag(frames: list[bytes], padding: int = 1024) -> bytes:
    """Wrap encoded frames in an ID3v2.3 header with some padding to grow into."""
    body = b"".join(frames) + b"\x00" * padding
    return b"ID3\x03\x00\x00" + _synchsafe(len(body)) + body


def strip_id3(raw: bytes) -> bytes:
    """Return ``raw`` with any leading ID3v2 tag removed."""
    if len(raw) >= 10 and raw[:3] == b"ID3":
        size = _unsynchsafe(raw[6:10])
        footer = 10 if (raw[5] & 0x10) else 0   # ID3v2.4 footer, if present
        return raw[10 + size + footer:]
    return raw


def read_id3(path: str | Path) -> dict[str, object]:
    """Parse the ID3v2.3 frames this module writes back out again.

    Text frames come back as strings, ``TXXX``/``COMM`` as ``{description:
    value}`` maps and ``GEOB`` as ``{description: bytes}``. Good enough to
    verify what we wrote; not a general ID3 reader.
    """
    raw = Path(path).read_bytes()
    out: dict[str, object] = {"TXXX": {}, "COMM": {}, "GEOB": {}}
    if len(raw) < 10 or raw[:3] != b"ID3":
        return out
    size = _unsynchsafe(raw[6:10])
    body, pos = raw[10:10 + size], 0
    while pos + 10 <= len(body):
        fid = body[pos:pos + 4]
        if not fid.strip(b"\x00"):
            break                                  # padding
        flen = struct.unpack(">I", body[pos + 4:pos + 8])[0]
        data = body[pos + 10:pos + 10 + flen]
        pos += 10 + flen
        name = fid.decode("latin-1")
        if name == "TXXX":
            enc = data[0]
            desc, val = _split_term(enc, data[1:])
            out["TXXX"][_decode_text(enc, desc)] = _decode_text(enc, val).rstrip("\x00")
        elif name == "COMM":
            enc = data[0]
            desc, val = _split_term(enc, data[4:])
            out["COMM"][_decode_text(enc, desc)] = _decode_text(enc, val).rstrip("\x00")
        elif name == "GEOB":
            enc = data[0]
            _mime, rest = _split_term(0x00, data[1:])
            _fn, rest = _split_term(enc, rest)
            desc, obj = _split_term(enc, rest)
            out["GEOB"][_decode_text(enc, desc)] = obj
        elif name.startswith("T"):
            out[name] = _decode_text(data[0], data[1:]).rstrip("\x00")
    return out


# ---------------------------------------------------------------------------
# Serato Markers2
# ---------------------------------------------------------------------------
#
# Serato stores hot cues in a GEOB frame called "Serato Markers2". The layout
# below is the community reverse-engineering (SERATO_REFERENCE); Serato has
# never published it. It round-trips through the decoder in this module and
# through the test suite, but it has not been loaded into Serato DJ Pro here --
# see the module docstring.

SERATO_MARKERS2 = "Serato Markers2"
_B64_LINE = 72


def _serato_entry(name: str, body: bytes) -> bytes:
    return name.encode("ascii") + b"\x00" + struct.pack(">I", len(body)) + body


def serato_markers2_payload(cues: list[Cue],
                            track_color: tuple[int, int, int] = (255, 255, 255),
                            bpm_lock: bool = False) -> bytes:
    """The *decoded* Markers2 payload: a version header then typed entries."""
    out = [b"\x01\x01"]
    out.append(_serato_entry("COLOR", b"\x00" + bytes(track_color)))
    for i, cue in enumerate(cues[:MAX_HOT_CUES]):
        r, g, b = cue.color
        body = (b"\x00"                                  # field marker
                + bytes([i])                             # hot cue index, 0 = A
                + struct.pack(">I", cue.millis)          # position, milliseconds
                + b"\x00"
                + bytes([r, g, b])                       # cue colour
                + b"\x00\x00"
                + cue.name.encode("utf8") + b"\x00")     # label
        out.append(_serato_entry("CUE", body))
    out.append(_serato_entry("BPMLOCK", b"\x01" if bpm_lock else b"\x00"))
    out.append(b"\x00")                                  # end of entries
    return b"".join(out)


def serato_markers2(cues: list[Cue], **kw) -> bytes:
    """The GEOB object bytes: a ``01 01`` header then line-wrapped base64."""
    blob = base64.b64encode(serato_markers2_payload(cues, **kw))
    lines = [blob[i:i + _B64_LINE] for i in range(0, len(blob), _B64_LINE)]
    return b"\x01\x01" + b"\n".join(lines) + b"\x00"


def parse_serato_markers2(data: bytes) -> list[dict]:
    """Decode what :func:`serato_markers2` wrote.

    Used by the tests, and by anyone who wants to check what is actually in a
    tagged file without opening Serato.
    """
    if not data.startswith(b"\x01\x01"):
        raise ExportError("not a Serato Markers2 object (bad version header)")
    b64 = data[2:].split(b"\x00", 1)[0].replace(b"\n", b"").replace(b"\r", b"")
    pad = (-len(b64)) % 4
    payload = base64.b64decode(b64 + b"=" * pad)
    if not payload.startswith(b"\x01\x01"):
        raise ExportError("Serato Markers2 payload has an unexpected version")
    pos, out = 2, []
    while pos < len(payload):
        end = payload.find(b"\x00", pos)
        if end < 0 or end == pos:
            break                                        # terminator
        name = payload[pos:end].decode("ascii", "replace")
        pos = end + 1
        if pos + 4 > len(payload):
            break
        length = struct.unpack(">I", payload[pos:pos + 4])[0]
        body = payload[pos + 4:pos + 4 + length]
        pos += 4 + length
        if name == "CUE" and len(body) >= 12:
            out.append({
                "type": "CUE",
                "index": body[1],
                "millis": struct.unpack(">I", body[2:6])[0],
                "color": tuple(body[7:10]),
                "name": body[12:].split(b"\x00", 1)[0].decode("utf8", "replace"),
            })
        elif name == "COLOR" and len(body) >= 4:
            out.append({"type": "COLOR", "color": tuple(body[1:4])})
        elif name == "BPMLOCK" and body:
            out.append({"type": "BPMLOCK", "locked": bool(body[0])})
    return out


# ---------------------------------------------------------------------------
# tagging an mp3 in place
# ---------------------------------------------------------------------------

def probe(path: str | Path) -> dict | None:
    """Duration, codec, sample rate and channels via ffprobe, or ``None``.

    Returning ``None`` when ffprobe is missing keeps tagging usable on a machine
    without it; the verification step simply does not run.
    """
    from .audio import _tool             # same resolution order as the decoder

    try:
        proc = subprocess.run(
            [_tool("ffprobe"), "-v", "error", "-show_entries",
             "format=duration:stream=codec_name,sample_rate,channels",
             "-select_streams", "a:0", "-of", "json", str(path)],
            capture_output=True, check=False, timeout=60,
        )
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout.decode("utf8", "replace"))
    except json.JSONDecodeError:
        return None
    stream = (data.get("streams") or [{}])[0]
    dur = (data.get("format") or {}).get("duration")
    return {
        "duration": round(float(dur), 3) if dur else None,
        "codec_name": stream.get("codec_name"),
        "sample_rate": stream.get("sample_rate"),
        "channels": stream.get("channels"),
    }


def _same_stream(before: dict | None, after: dict | None) -> bool:
    if not before or not after:
        return True
    for k in ("codec_name", "sample_rate", "channels"):
        if before.get(k) != after.get(k):
            return False
    a, b = before.get("duration"), after.get("duration")
    if a is None or b is None:
        return True
    return abs(a - b) <= 0.05


def track_frames(track: Track, *, serato: bool = False) -> list[bytes]:
    """Every ID3 frame fourfloor writes for one remix, in tag order."""
    frames = [
        text_frame("TIT2", track.title),
        text_frame("TPE1", track.artist),
        text_frame("TALB", track.session.get("source", {}).get("file", "") or track.title),
        text_frame("TCON", track.genre),
        text_frame("TBPM", f"{track.bpm:.2f}"),
        text_frame("TKEY", track.key),
        text_frame("TLEN", str(int(round(track.duration * 1000)))),
        text_frame("TENC", GENERATOR),
        # Rekordbox and Serato both read INITIALKEY from a TXXX frame; TKEY
        # alone is ignored by several versions of both.
        txxx_frame("INITIALKEY", track.key),
        txxx_frame("CAMELOT", track.camelot),
        txxx_frame("FOURFLOOR_SESSION", session_path_for(track.path).name),
        comment_frame(track.comment()),
    ]
    if serato:
        frames.append(geob_frame(SERATO_MARKERS2, serato_markers2(track.hot_cues)))
    return frames


def tag_file(track: Track, *, serato: bool = False, verify: bool = True) -> dict:
    """Write fourfloor's ID3v2.3 tag onto ``track.path``, audio untouched.

    The existing tag is replaced, the MPEG frames are copied through unchanged,
    and the result is written to a sibling temp file and moved into place, so an
    interrupted run cannot leave a half-written mp3 behind.
    """
    path = Path(track.path)
    if path.suffix.lower() != ".mp3":
        raise ExportError(f"can only tag mp3 files, not {path.suffix} ({path.name})")
    before = probe(path) if verify else None

    raw = path.read_bytes()
    audio = strip_id3(raw)
    tag = id3v23_tag(track_frames(track, serato=serato))
    tmp = path.with_name(path.name + ".fourfloor-tmp")
    try:
        tmp.write_bytes(tag + audio)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()

    after = probe(path) if verify else None
    if verify and not _same_stream(before, after):
        raise ExportError(
            f"tagging changed the audio stream of {path.name} "
            f"({before} -> {after}); the file has been left as written, "
            f"re-render it before the gig"
        )
    return {"file": str(path), "bytes": len(tag), "serato": serato,
            "probe": after or before}


# ---------------------------------------------------------------------------
# the front door
# ---------------------------------------------------------------------------

@dataclass
class ExportResult:
    """What an export produced, for the report and for ``--json``."""

    set_name: str
    tracks: list[Track] = field(default_factory=list)
    files: dict[str, Path] = field(default_factory=dict)
    tagged: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "set": self.set_name,
            "tracks": [
                {"file": str(t.path), "title": t.title, "artist": t.artist,
                 "bpm": t.bpm, "key": t.key, "camelot": t.camelot,
                 "duration": t.duration,
                 "hot_cues": [{"letter": HOT_CUE_LETTERS[i], "name": c.name,
                               "seconds": c.seconds, "bar": c.bar}
                              for i, c in enumerate(t.hot_cues)]}
                for t in self.tracks
            ],
            "files": {k: str(v) for k, v in self.files.items()},
            "tagged": [{k: v for k, v in d.items() if k != "probe"}
                       for d in self.tagged],
            "notes": self.notes,
        }


def normalise_formats(formats) -> list[str]:
    """Expand ``all``, drop duplicates, keep a stable order, refuse nonsense."""
    if not formats:
        return list(DEFAULT_FORMATS)
    wanted: list[str] = []
    for f in formats:
        for part in str(f).split(","):
            part = part.strip().lower()
            if not part:
                continue
            if part == "all":
                wanted.extend(FORMATS)
            elif part in FORMATS:
                wanted.append(part)
            else:
                raise ExportError(
                    f"unknown export format {part!r}; choose from "
                    f"{', '.join(FORMATS)} or 'all'"
                )
    return [f for i, f in enumerate(wanted) if f not in wanted[:i]]


def export(target: str | Path, set_name: str, formats=None,
           out_dir: str | Path | None = None, artist: str = GENERATOR,
           suffix: bool = False, verify: bool = True) -> ExportResult:
    """Export one remix or a folder of them into DJ-ready files.

    ``out_dir`` receives ``rekordbox.xml`` and ``cues.csv``; the ``tags`` and
    ``serato`` formats always write into the mp3 itself, wherever it lives.
    """
    wanted = normalise_formats(formats)
    tracks = collect(target, artist=artist, suffix=suffix)
    base = Path(target)
    out = Path(out_dir) if out_dir else (base if base.is_dir() else base.parent)
    res = ExportResult(set_name=set_name, tracks=tracks)

    if "rekordbox" in wanted:
        res.files["rekordbox"] = write_rekordbox(tracks, out, set_name)
    if "csv" in wanted:
        res.files["csv"] = write_csv(tracks, out)
    if "tags" in wanted or "serato" in wanted:
        want_serato = "serato" in wanted
        for t in tracks:
            res.tagged.append(tag_file(t, serato=want_serato, verify=verify))
        if want_serato:
            res.notes.append(
                "Serato cues were written to a 'Serato Markers2' GEOB frame from "
                "the public reverse-engineered format; Serato publishes no spec, "
                "so check one track in Serato DJ before relying on it at a gig"
            )
    return res
