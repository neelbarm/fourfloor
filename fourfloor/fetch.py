"""Pull audio off a link -- YouTube, YouTube Music, SoundCloud, anything yt-dlp reads.

The point is to remove the boring step. Instead of finding an mp3, downloading
it, and dragging it in, you paste the link and fourfloor does the rest:
``fourfloor fetch <url>``, ``fourfloor inspect --url <url>``, or the link field
on the app's drop screen.

How it works, and why:

* **yt-dlp is a binary we shell out to, not a dependency.** It ships its own
  release cadence against sites that change weekly; pinning it in
  ``pyproject.toml`` would age badly and drag ffmpeg-flavoured extras into a
  library whose job is DSP. If it is not on PATH we say so in one sentence.
* **The metadata comes first, the bytes second.** A ``--dump-single-json``
  probe gives the title, uploader and duration, so the filename is chosen by
  :func:`sanitize` from text we control -- yt-dlp's own ``-o`` template never
  sees the user's words. The download lands in a temporary folder beside the
  destination and is moved into place afterwards, so a failed or half-finished
  fetch cannot leave a plausible-looking mp3 in your reference folder.
* **Nothing is ever overwritten.** A name that exists gets ``-2``, ``-3``, and
  so on. Reference folders are the user's own library; clobbering a file there
  would be the worst thing this module could do.
* **Only http(s), never a local address.** The web app hands this module a URL
  typed into a browser, so it must not be usable to make the server read
  ``file:///etc/passwd`` or probe ``169.254.169.254``.

Downloading from YouTube is against YouTube's terms of service. This is here
for private, local analysis of tracks you already have the right to use.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlsplit

#: How long one track may take before we give up on it.
DEFAULT_TIMEOUT = 900.0
#: The metadata probe is a single API call; it should never take this long.
PROBE_TIMEOUT = 90.0

#: Where reference material lives, by the convention `fourfloor learn` reads.
REFS_HOME = Path.home() / "Music" / "house-refs"
REMIXES_DIR = REFS_HOME / "remixes"
PAIRS_DIR = REFS_HOME / "pairs"

#: The two halves of a pair: ``<name>.original.mp3`` and ``<name>.remix.mp3``.
KINDS = ("original", "remix")

#: Characters a filename may keep. Everything else becomes a space.
_KEEP = re.compile(r"[^\w \-.,'()&+]", re.UNICODE)
_SPACES = re.compile(r"\s+")
_PERCENT = re.compile(r"\[download\]\s+([0-9]+(?:\.[0-9]+)?)%")

#: Progress callback: ``progress(percent, note)`` with percent in 0..100.
Progress = Callable[[float, str], None]


class FetchError(RuntimeError):
    """A link we could not turn into audio, with a sentence for the user."""


# ---------------------------------------------------------------------------
# what we got
# ---------------------------------------------------------------------------

@dataclass
class Fetched:
    """One downloaded track: the file, and what the site said about it.

    It is also a :class:`os.PathLike`, so ``open(fetched)`` and
    ``analyze(fetched)`` work as if it were the path it wraps.
    """

    path: Path
    title: str = ""
    uploader: str = ""
    duration: float = 0.0
    url: str = ""
    site: str = ""
    playlist: str = ""
    bytes: int = 0
    _extra: dict = field(default_factory=dict, repr=False)

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    def to_dict(self) -> dict:
        return {
            "path": str(self.path), "name": self.path.name, "title": self.title,
            "uploader": self.uploader, "duration": round(float(self.duration or 0.0), 2),
            "url": self.url, "site": self.site, "playlist": self.playlist,
            "bytes": self.bytes,
        }


# ---------------------------------------------------------------------------
# the binary
# ---------------------------------------------------------------------------

def ytdlp() -> str:
    """The yt-dlp executable, or a :class:`FetchError` that says how to get it."""
    override = os.environ.get("FOURFLOOR_YTDLP") or ""
    if override:
        found = shutil.which(override) or (override if Path(override).is_file() else "")
    else:
        found = shutil.which("yt-dlp") or ""
    if not found:
        raise FetchError(
            "yt-dlp is not on your PATH, and fourfloor needs it to read a link. "
            "Install it with `brew install yt-dlp` (or `pipx install yt-dlp`) "
            "and try again.")
    return found


def available() -> bool:
    """Whether links can be fetched at all on this machine."""
    try:
        ytdlp()
    except FetchError:
        return False
    return True


# ---------------------------------------------------------------------------
# urls
# ---------------------------------------------------------------------------

_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "broadcasthost"}


def check_url(url: str) -> str:
    """Return ``url`` if it is a public http(s) link, else raise.

    The app lets a browser post a URL, so this is a real boundary: ``file:``,
    ``ftp:``, a bare path, a loopback address or anything on a private network
    is refused before yt-dlp is started.
    """
    raw = (url or "").strip()
    if not raw:
        raise FetchError("paste a link first.")
    if len(raw) > 2048:
        raise FetchError("that link is implausibly long.")
    if any(ch in raw for ch in "\r\n\t\x00"):
        raise FetchError("that link has control characters in it.")
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        named = f" -- it will not touch {scheme}:" if scheme else ""
        raise FetchError(f"fourfloor only fetches http and https links{named}")
    host = (parts.hostname or "").strip(".").lower()
    if not host:
        raise FetchError("that link has no host in it.")
    if host in _LOCAL_NAMES or host.endswith(".local") or host.endswith(".internal"):
        raise FetchError("that is a local address, not a link to a track.")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if "." not in host:                       # a bare name is a machine on this LAN
            raise FetchError("that is a local address, not a link to a track.") from None
    else:
        if (ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise FetchError("that is a local address, not a link to a track.")
    return raw


def site_of(url: str) -> str:
    """A short label for where a link points, for the report."""
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    return {
        "youtu.be": "youtube", "youtube.com": "youtube",
        "music.youtube.com": "youtube music", "m.youtube.com": "youtube",
        "soundcloud.com": "soundcloud", "on.soundcloud.com": "soundcloud",
        "m.soundcloud.com": "soundcloud", "bandcamp.com": "bandcamp",
        "vimeo.com": "vimeo", "mixcloud.com": "mixcloud",
    }.get(host, host or "link")


# ---------------------------------------------------------------------------
# names on disk
# ---------------------------------------------------------------------------

def sanitize(text: str, fallback: str = "track") -> str:
    """Reduce a title to something safe, readable and short enough to type."""
    name = _KEEP.sub(" ", str(text or "")).replace("/", " ")
    name = _SPACES.sub(" ", name).strip(" .-_")
    name = name[:80].strip(" .-_")
    return name or fallback


def free_path(directory: Path, stem: str, suffix: str = ".mp3") -> Path:
    """``<dir>/<stem><suffix>``, with ``-2``, ``-3``… if that name is taken."""
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}{suffix}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{n}{suffix}"
        n += 1
        if n > 999:                                # pathological; stop politely
            raise FetchError(f"there are already 999 files called {stem} in "
                             f"{directory}.")
    return candidate


def pair_stem(name: str, kind: str | None) -> str:
    """The stem of one half of a pair: ``<name>.original`` / ``<name>.remix``."""
    stem = sanitize(name)
    if kind is None:
        return stem
    if kind not in KINDS:
        raise FetchError(f"--as takes {' or '.join(KINDS)}, not {kind!r}")
    return f"{stem}.{kind}"


def resolve_dest(where: str | os.PathLike | None) -> Path:
    """Turn ``remixes``, ``pairs`` or a folder into a path, without creating it."""
    if where is None or where == "" or where == "remixes":
        return REMIXES_DIR
    if where == "pairs":
        return PAIRS_DIR
    return Path(str(where)).expanduser()


# ---------------------------------------------------------------------------
# running yt-dlp
# ---------------------------------------------------------------------------

def _friendly(stderr: str, url: str) -> str:
    """Turn yt-dlp's last words into one sentence a person can act on."""
    lines = [ln.strip() for ln in stderr.strip().splitlines() if ln.strip()]
    err = next((ln for ln in reversed(lines) if ln.startswith("ERROR:")), "")
    text = err.removeprefix("ERROR:").strip() or (lines[-1] if lines else "")
    low = text.lower()
    if "unsupported url" in low or "is not a valid url" in low:
        return f"yt-dlp does not know that site: {url}"
    if "private" in low or "sign in" in low or "login" in low or "members-only" in low:
        return ("that track is private or needs a sign-in, so yt-dlp could not "
                "read it.")
    if "video unavailable" in low or "not available" in low or "removed" in low:
        return "that track is not available any more."
    if "ffmpeg" in low or "ffprobe" in low:
        return ("yt-dlp needs ffmpeg to make an mp3 and could not find it. "
                "Install it with `brew install ffmpeg`.")
    if "urlopen error" in low or "getaddrinfo" in low or "network" in low \
            or "connection" in low:
        return "that link could not be reached -- check the network and try again."
    return f"yt-dlp could not fetch that link: {text or 'it failed with no message'}"


def _run(cmd: list[str], timeout: float,
         on_line: Callable[[str], None] | None = None) -> tuple[int, str]:
    """Run yt-dlp, streaming merged output through ``on_line``. Returns (code, log)."""
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, encoding="utf8", errors="replace",
            bufsize=1)
    except OSError as exc:                         # the binary vanished mid-run
        raise FetchError(f"could not start yt-dlp: {exc}") from None

    timed_out = threading.Event()

    def _kill() -> None:
        timed_out.set()
        proc.kill()

    watchdog = threading.Timer(timeout, _kill)
    watchdog.daemon = True
    watchdog.start()
    log: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if len(log) < 400:
                log.append(line)
            if on_line is not None:
                on_line(line)
        code = proc.wait()
    finally:
        watchdog.cancel()
        if proc.poll() is None:                    # we left early somehow
            proc.kill()
    if timed_out.is_set():
        raise FetchError(f"that download was still going after "
                         f"{int(timeout)}s, so fourfloor gave up on it.")
    return code, "\n".join(log)


def probe(url: str, timeout: float = PROBE_TIMEOUT) -> dict:
    """Ask the site what is at ``url`` without downloading anything."""
    url = check_url(url)
    cmd = [ytdlp(), "--dump-single-json", "--flat-playlist", "--no-warnings",
           "--no-color", "--ignore-config", "--socket-timeout", "20", url]
    code, log = _run(cmd, timeout)
    if code != 0:
        raise FetchError(_friendly(log, url))
    for line in log.splitlines():                  # the JSON is the last full line
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise FetchError(f"yt-dlp said nothing about {url}")


def entries_of(info: dict) -> list[dict]:
    """The tracks of a playlist or set, or ``[]`` if it is a single track."""
    if str(info.get("_type") or "") != "playlist":
        return []
    out = []
    for e in info.get("entries") or []:
        if not isinstance(e, dict):
            continue
        link = e.get("url") or e.get("webpage_url") or e.get("original_url")
        if link:
            out.append(e)
    return out


def is_playlist(info: dict) -> bool:
    """Whether a probe found a playlist or SoundCloud set with more than one track."""
    return len(entries_of(info)) > 1


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def _download(url: str, into: Path, progress: Progress | None,
              timeout: float, label: str) -> Path:
    """Run the download into ``into`` and return the mp3 it produced."""
    cmd = [ytdlp(), "--extract-audio", "--audio-format", "mp3", "--audio-quality", "0",
           "--no-playlist", "--newline", "--no-color", "--no-warnings",
           "--ignore-config", "--no-mtime", "--socket-timeout", "30",
           "--retries", "3", "-o", str(into / "audio.%(ext)s"), url]

    last = -1.0

    def on_line(line: str) -> None:
        nonlocal last
        if progress is None:
            return
        m = _PERCENT.search(line)
        if m:
            pct = min(100.0, float(m.group(1)))
            if pct >= last + 0.5 or pct >= 100.0:
                last = pct
                progress(pct, label)
        elif "[ExtractAudio]" in line:
            progress(100.0, "converting to mp3")

    code, log = _run(cmd, timeout, on_line)
    if code != 0:
        raise FetchError(_friendly(log, url))
    mp3s = sorted(into.glob("*.mp3"))
    if not mp3s:
        leftovers = sorted(p for p in into.iterdir() if p.is_file())
        if not leftovers:
            raise FetchError(f"yt-dlp finished but wrote nothing for {url}")
        raise FetchError(
            f"yt-dlp wrote {leftovers[0].name}, not an mp3 -- ffmpeg is probably "
            f"missing. Install it with `brew install ffmpeg`.")
    return mp3s[0]


def fetch(url: str, dest_dir: str | os.PathLike, name: str | None = None,
          kind: str | None = None, progress: Progress | None = None,
          timeout: float = DEFAULT_TIMEOUT, info: dict | None = None) -> Fetched:
    """Download one track to ``dest_dir`` and return where it landed.

    ``name`` overrides the title for the filename; ``kind`` is ``original`` or
    ``remix`` and writes the ``<name>.<kind>.mp3`` half of a learning pair.
    ``progress`` is called as ``progress(percent, note)``. The return value is
    a :class:`Fetched`, which is usable anywhere a path is.
    """
    url = check_url(url)
    dest = Path(dest_dir).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    if info is None:
        if progress is not None:
            progress(0.0, "reading the link")
        info = probe(url, timeout=min(timeout, PROBE_TIMEOUT))
        if is_playlist(info):
            raise FetchError(
                f"{url} is a playlist or set of {len(entries_of(info))} tracks -- "
                f"use fetch_all() (the CLI does this for you).")

    title = str(info.get("title") or "").strip()
    uploader = str(info.get("uploader") or info.get("channel")
                   or info.get("artist") or "").strip()
    duration = float(info.get("duration") or 0.0)
    stem = pair_stem(name, kind) if name else pair_stem(sanitize(title), kind)
    label = f"{title or url}"

    tmp = Path(tempfile.mkdtemp(prefix=".fourfloor-dl-", dir=dest))
    try:
        got = _download(url, tmp, progress, timeout, label)
        target = free_path(dest, stem)
        shutil.move(str(got), str(target))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if progress is not None:
        progress(100.0, "saved")
    return Fetched(
        path=target, title=title or target.stem, uploader=uploader,
        duration=duration, url=str(info.get("webpage_url") or url),
        site=site_of(url), bytes=target.stat().st_size,
        _extra={"id": info.get("id"), "extractor": info.get("extractor_key")})


def fetch_all(url: str, dest_dir: str | os.PathLike, name: str | None = None,
              kind: str | None = None, progress: Progress | None = None,
              timeout: float = DEFAULT_TIMEOUT) -> list[Fetched]:
    """Fetch a link, expanding a playlist or SoundCloud set into every track."""
    url = check_url(url)
    if progress is not None:
        progress(0.0, "reading the link")
    info = probe(url, timeout=min(timeout, PROBE_TIMEOUT))
    if not is_playlist(info):
        return [fetch(url, dest_dir, name=name, kind=kind, progress=progress,
                      timeout=timeout, info=info)]

    entries = entries_of(info)
    playlist = str(info.get("title") or "").strip()
    out: list[Fetched] = []
    for i, entry in enumerate(entries, start=1):
        link = str(entry.get("url") or entry.get("webpage_url"))
        stem_name = f"{sanitize(name)}-{i:02d}" if name else None
        note = f"{i}/{len(entries)}"

        def step(pct: float, detail: str, note=note) -> None:
            if progress is not None:
                progress(pct, f"{note} · {detail}")

        got = fetch(link, dest_dir, name=stem_name, kind=kind,
                    progress=step if progress else None, timeout=timeout,
                    info=None if entry.get("duration") is None else dict(entry))
        got.playlist = playlist
        out.append(got)
    return out


def fetch_pair(original_url: str, remix_url: str, name: str,
               dest_dir: str | os.PathLike | None = None,
               progress: Progress | None = None,
               timeout: float = DEFAULT_TIMEOUT) -> list[Fetched]:
    """Fetch both halves of a learning pair under one name."""
    dest = Path(dest_dir) if dest_dir is not None else PAIRS_DIR
    out = []
    for kind, link in zip(KINDS, (original_url, remix_url)):
        def step(pct: float, detail: str, kind=kind) -> None:
            if progress is not None:
                progress(pct, f"{kind} · {detail}")
        out.append(fetch(link, dest, name=name, kind=kind,
                         progress=step if progress else None, timeout=timeout))
    return out


def fetch_to_temp(url: str, progress: Progress | None = None,
                  timeout: float = DEFAULT_TIMEOUT) -> tuple[Fetched, Callable[[], None]]:
    """Fetch into a throwaway folder; returns the track and a cleanup callable.

    ``inspect --url`` and ``remix --url`` use this: the download is scratch, and
    the caller decides when it stops mattering.
    """
    tmp = Path(tempfile.mkdtemp(prefix="fourfloor-link-"))
    try:
        got = fetch(url, tmp, progress=progress, timeout=timeout)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return got, lambda: shutil.rmtree(tmp, ignore_errors=True)


def total_bytes(items: Iterable[Fetched]) -> int:
    return sum(int(i.bytes or 0) for i in items)


def fmt_bytes(n: int) -> str:
    mb = n / (1024 * 1024)
    return f"{mb:.1f} MB" if mb >= 1 else f"{n / 1024:.0f} KB"
