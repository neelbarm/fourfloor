"""``fourfloor serve``: the local web app.

Stdlib only -- :class:`~http.server.ThreadingHTTPServer` with a hand-rolled
router, a single background worker for the remix queue and Server-Sent Events
for progress. There is no framework, no build step and no runtime dependency
the CLI does not already have.

The rules it holds to:

* **127.0.0.1 only**, plus a ``Host`` check, so a page on the internet cannot
  reach the API through a rebound DNS name.
* **Options go through the CLI parser.** A request is rendered as the flags a
  person would have typed and handed to :func:`fourfloor.cli.parse_remix_args`,
  so the app cannot ask for anything ``fourfloor remix`` would refuse.
* **No path comes from the browser.** Ids are minted in :mod:`fourfloor.store`
  and downloadable names are an allowlist.
* **A pasted link is a URL, not a fetch instruction.**
  :func:`fourfloor.fetch.check_url` refuses anything that is not public http(s)
  before yt-dlp is started, so the link field cannot be used to read a local
  file or knock on a machine inside the network.
"""

from __future__ import annotations

import json
import mimetypes
import os
import socket
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import cli, store, ui
from .arrange import FORMS, fmt_time
from .jobs import JobQueue, PhaseTimer
from .remix import MAX_TARGET_BPM, MIN_TARGET_BPM, PHASES
from .store import MAX_UPLOAD_BYTES, Library

WEB_DIR = Path(__file__).resolve().parent / "web"
BPM_PRESETS = (120.0, 124.0, 126.0, 128.0, 132.0)
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml",
                ".json": "application/json", ".ico": "image/x-icon"}
SSE_TIMEOUT = 3600.0
CSP = ("default-src 'self'; media-src 'self' blob:; img-src 'self' data: blob:; "
       "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
       "form-action 'none'; base-uri 'none'; frame-ancestors 'none'")


class HttpError(Exception):
    """An error with a status code, rendered as JSON."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------------------
# style profiles
# ---------------------------------------------------------------------------

def style_dirs(lib: Library) -> list[Path]:
    """Where a ``styles/`` folder might be, most local first."""
    here = Path(__file__).resolve().parent.parent
    seen, out = set(), []
    for d in (Path.cwd() / "styles", here / "styles", lib.home / "styles"):
        r = d.resolve() if d.exists() else d
        if d.is_dir() and r not in seen:
            seen.add(r)
            out.append(d)
    return out


def list_styles(lib: Library) -> list[dict]:
    """Every style profile the app will let you pick, by name."""
    out: dict[str, dict] = {}
    for d in style_dirs(lib):
        for f in sorted(d.glob("*.json")):
            if f.stem in out:
                continue
            try:
                data = json.loads(f.read_text(encoding="utf8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or "bpm" not in data:
                continue
            out[f.stem] = {
                "name": f.stem,
                "label": f.stem.replace("-", " ").replace("_", " "),
                "bpm": data.get("bpm"),
                "swing": data.get("swing"),
                "length": data.get("length"),
                "refs": data.get("n_refs", 0),
                "path": str(f),
            }
    return list(out.values())


def resolve_style(lib: Library, name: str) -> str:
    """Turn a style *name* from the browser into a path we chose."""
    for s in list_styles(lib):
        if s["name"] == name:
            return s["path"]
    raise HttpError(400, f"no style profile called {name!r}")


# ---------------------------------------------------------------------------
# the app
# ---------------------------------------------------------------------------

class App:
    """Everything the handlers need: the library, the queue, the config."""

    def __init__(self, lib: Library) -> None:
        self.lib = lib
        self.queue = JobQueue()
        self.lib.sweep_uploads()

    # -- config -----------------------------------------------------------

    def config(self) -> dict:
        from . import fetch as fetch_mod
        from .stems import demucs_available

        # the page shows this path, so show it the way a person writes it
        home = self.lib.home
        try:
            shown = "~/" + str(home.relative_to(Path.home()))
        except ValueError:
            shown = str(home)
        return {
            "version": cli.VERSION,
            "forms": list(FORMS),
            "bpm_presets": list(BPM_PRESETS),
            "bpm_range": [MIN_TARGET_BPM, MAX_TARGET_BPM],
            "phases": list(PHASES),
            "styles": [{k: v for k, v in s.items() if k != "path"}
                       for s in list_styles(self.lib)],
            "demucs": demucs_available(),
            "fetch": fetch_mod.available(),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "upload_exts": sorted(store.UPLOAD_EXTS),
            "queue_depth": self.queue.depth(),
            "home": shown,
        }

    # -- upload -----------------------------------------------------------

    def upload(self, part_name: str, filename: str, tmp_path: Path) -> dict:
        """Take a spooled upload into the library and analyse it."""
        from . import multipart
        from .analysis import analyze, suggest_house_tempo
        from .preview import waveform_of

        try:
            sid, target = self.lib.create_source(filename)
        except ValueError as exc:
            raise HttpError(400, str(exc)) from None
        part = multipart.Part(name=part_name, filename=filename, path=tmp_path)
        multipart.move(part, target)

        try:
            a = analyze(target)
        except Exception as exc:                      # noqa: BLE001
            raise HttpError(400, f"could not read that audio: {exc}") from None

        meta = {
            "id": sid,
            "name": store.display_name(filename),
            "title": store.title_of(filename),
            "created": time.time(),
            "bytes": target.stat().st_size,
            "analysis": a.to_dict(),
            "suggested_bpm": suggest_house_tempo(a.grid.bpm),
            "wave": waveform_of(a.clip.mono, 900) if a.clip is not None else [],
        }
        self.lib.save_source_meta(sid, meta)
        return meta

    # -- fetch ------------------------------------------------------------

    def start_fetch(self, payload: dict) -> dict:
        """Queue a download of a pasted link. It ends where an upload ends.

        The job runs on the same single worker as the remixes, so a link pasted
        while a remix is rendering waits its turn and says so -- one queue, one
        event stream, one thing to reason about. Its result is exactly the
        object ``POST /api/upload`` returns, so the page joins the normal path
        at the analysis and lands on the controls screen.
        """
        from . import fetch as fetch_mod

        try:
            url = fetch_mod.check_url(str(payload.get("url") or ""))
        except fetch_mod.FetchError as exc:
            raise HttpError(400, str(exc)) from None
        if not fetch_mod.available():
            raise HttpError(400, "yt-dlp is not installed on this machine, so "
                                 "fourfloor cannot read a link. `brew install yt-dlp`")
        jid = store.new_id()
        job = self.queue.submit(jid, lambda j: self._run_fetch(j, url),
                                label=fetch_mod.site_of(url))
        return {"job": jid, "url": url, "site": fetch_mod.site_of(url),
                "queued": self.queue.depth(), "state": job.state}

    def _run_fetch(self, job, url: str) -> dict:
        import shutil
        import tempfile

        from . import fetch as fetch_mod

        job.emit("phase", name="fetch", detail="reading the link")
        info = fetch_mod.probe(url)
        if fetch_mod.is_playlist(info):
            n = len(fetch_mod.entries_of(info))
            raise fetch_mod.FetchError(
                f"that link is a set of {n} tracks. The app takes one at a time -- "
                f"paste a single track, or run `fourfloor fetch` in a terminal.")
        title = str(info.get("title") or "").strip()
        job.emit("phase", name="fetch", detail=title or url)

        seen = -1.0

        def progress(pct: float, note: str = "") -> None:
            nonlocal seen
            if pct >= seen + 1.0 or pct >= 100.0:
                seen = pct
                job.emit("progress", percent=round(pct, 1), detail=note or title)

        tmp = Path(tempfile.mkdtemp(prefix="link-", dir=self.lib.uploads))
        try:
            got = fetch_mod.fetch(url, tmp, progress=progress, info=info)
            job.emit("phase_done", name="fetch", elapsed=round(time.time() - job.started, 2))
            job.emit("phase", name="analyse", detail=got.title or got.path.stem)
            meta = self.upload("file", got.path.name, got.path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        meta = self.lib.save_source_meta(meta["id"], {**meta, "link": got.to_dict()})
        job.emit("phase_done", name="analyse",
                 elapsed=round(time.time() - job.started, 2))
        return meta

    # -- remix ------------------------------------------------------------

    def start_remix(self, payload: dict) -> dict:
        lib = self.lib
        source_id = store.safe_id(str(payload.get("source", "")))
        src_meta = lib.source_meta(source_id)
        src_path = lib.source_audio(source_id)

        rid, out_dir = lib.create_remix()
        out = out_dir / "remix.mp3"
        argv = ["remix", str(src_path), "-o", str(out)]

        def flag(name: str, value) -> None:
            if value is None or value == "" or value is False:
                return
            argv.extend([name, str(value)])

        flag("--bpm", payload.get("bpm"))
        key = payload.get("key")
        match = payload.get("compatible_with")
        if match:
            match_path = lib.source_audio(store.safe_id(str(match)))
            argv.extend(["--compatible-with", str(match_path)])
        elif key and str(key).lower() != "keep":
            flag("--key", key)
        flag("--stems", payload.get("stems"))
        flag("--form", payload.get("form"))
        flag("--length", payload.get("length"))
        flag("--swing", payload.get("swing"))
        flag("--seed", payload.get("seed"))
        style_name = payload.get("style")
        if style_name:
            argv.extend(["--style", resolve_style(lib, str(style_name))])

        try:
            args = cli.parse_remix_args(argv)
            opts = cli.remix_options(args)
            from .remix import validate_options
            validate_options(opts, out)
        except (cli.CliError, ValueError) as exc:
            lib.delete_remix(rid)
            raise HttpError(400, str(exc)) from None
        if opts.stems == "demucs":
            from .stems import demucs_available
            if not demucs_available():
                lib.delete_remix(rid)
                raise HttpError(400, "demucs is not installed in this environment; "
                                     "use the built-in HPSS separation.")

        style = None
        if args.style:
            from .style import Style
            try:
                style = Style.load(args.style)
            except (OSError, ValueError) as exc:
                lib.delete_remix(rid)
                raise HttpError(400, str(exc)) from None

        title = src_meta.get("title") or "Untitled"
        job = self.queue.submit(
            rid, lambda j: self._run_remix(j, rid, out_dir, src_path, src_meta,
                                           opts, style, payload),
            label=title)
        return {"job": rid, "remix": rid, "title": title,
                "queued": self.queue.depth(), "state": job.state}

    def _run_remix(self, job, rid: str, out_dir: Path, src_path: Path,
                   src_meta: dict, opts, style, payload: dict) -> dict:
        from .preview import waveform_of
        from .remix import remix

        timer = PhaseTimer(job)
        try:
            res = remix(src_path, out_dir / "remix.mp3", opts, style=style,
                        progress=timer)
        finally:
            timer.close()

        wave = waveform_of(res.audio, 1400)
        (out_dir / "wave.json").write_text(json.dumps(wave), encoding="utf8")
        meta = self.lib.save_remix_meta(rid, {
            "title": src_meta.get("title") or "Untitled",
            "source_id": src_meta.get("id"),
            "source_name": src_meta.get("name"),
            "created": time.time(),
            "bpm": res.session["bpm"],
            "key": res.session["key"],
            "camelot": res.session["camelot"],
            "semitone_shift": res.session["semitone_shift"],
            "duration": res.session["duration"],
            "length": fmt_time(res.session["duration"]),
            "bars": res.session["bars"],
            "form": res.plan.form,
            "stems": opts.stems,
            "warnings": list(res.warnings),
            "note": res.plan.note,
            "options": {
                "bpm": payload.get("bpm"), "key": payload.get("key"),
                "compatible_with": payload.get("compatible_with"),
                "stems": opts.stems, "form": opts.form, "length": opts.length,
                "swing": opts.swing, "style": payload.get("style"),
                "seed": opts.seed,
            },
            "files": sorted(n for n in store.REMIX_FILES if (out_dir / n).is_file()),
        })
        return meta

    # -- library ----------------------------------------------------------

    def remix_detail(self, rid: str) -> dict:
        d = self.lib.remix_dir(rid)
        meta = self.lib.remix_meta(rid)
        session = json.loads((d / "remix.session.json").read_text(encoding="utf8"))
        wave_file = d / "wave.json"
        wave = json.loads(wave_file.read_text(encoding="utf8")) if wave_file.is_file() else []
        source = {}
        if meta.get("source_id"):
            try:
                src = self.lib.source_meta(meta["source_id"])
                source = {k: src[k] for k in ("id", "name", "title", "analysis",
                                              "suggested_bpm", "wave") if k in src}
            except store.NotFound:
                source = {}
        return {"meta": meta, "session": session, "wave": wave, "source": source}

    def close(self) -> None:
        self.queue.close()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    """The whole router. ``app`` is attached to the server."""

    server_version = f"fourfloor/{cli.VERSION}"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> App:
        return self.server.app                        # type: ignore[attr-defined]

    # -- plumbing ---------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:   # noqa: D102
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _host_is_local(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("localhost", "127.0.0.1", "::1", "")

    def _send(self, status: int, body: bytes = b"", ctype: str = "application/json",
              extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data).encode("utf8"))

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message, "status": status}, status)

    # -- routing ----------------------------------------------------------

    def do_GET(self) -> None:                         # noqa: N802
        self._route("GET")

    def do_HEAD(self) -> None:                        # noqa: N802
        self._route("GET")

    def do_POST(self) -> None:                        # noqa: N802
        self._route("POST")

    def do_DELETE(self) -> None:                      # noqa: N802
        self._route("DELETE")

    def _route(self, method: str) -> None:
        if not self._host_is_local():
            self._error(403, "fourfloor serve only answers to localhost")
            return
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        try:
            if path.startswith("/api/"):
                self._api(method, path, query)
            elif method == "GET":
                self._static(path)
            else:
                self._error(405, f"{method} is not allowed here")
        except HttpError as exc:
            self._error(exc.status, exc.message)
        except store.NotFound as exc:
            self._error(404, str(exc))
        except BrokenPipeError:
            pass                                       # the browser hung up
        except Exception as exc:                       # noqa: BLE001
            self._error(500, f"{exc.__class__.__name__}: {exc}")

    def _api(self, method: str, path: str, query: dict) -> None:
        parts = [p for p in path.split("/") if p][1:]   # drop "api"
        app = self.app

        if parts == ["config"] and method == "GET":
            self._json(app.config())
        elif parts == ["upload"] and method == "POST":
            self._upload()
        elif parts == ["fetch"] and method == "POST":
            self._json(app.start_fetch(self._body_json()))
        elif parts == ["remix"] and method == "POST":
            self._json(app.start_remix(self._body_json()))
        elif len(parts) == 3 and parts[0] == "jobs" and parts[2] == "events" \
                and method == "GET":
            self._events(parts[1], query)
        elif len(parts) == 2 and parts[0] == "jobs" and method == "GET":
            job = app.queue.get(store.safe_id(parts[1]))
            if job is None:
                raise HttpError(404, "no such job")
            self._json({**job.to_dict(), "events": list(job.events)})
        elif parts == ["remixes"] and method == "GET":
            self._json({"remixes": app.lib.list_remixes()})
        elif len(parts) == 2 and parts[0] == "remixes" and method == "GET":
            self._json(app.remix_detail(parts[1]))
        elif len(parts) == 2 and parts[0] == "remixes" and method == "DELETE":
            app.lib.delete_remix(parts[1])
            self._json({"deleted": parts[1]})
        elif len(parts) == 3 and parts[0] == "remixes" and method == "GET":
            self._download(parts[1], parts[2], "download" in query)
        else:
            raise HttpError(404, f"no route for {method} {path}")

    # -- endpoints --------------------------------------------------------

    def _body_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 1024 * 1024:
            raise HttpError(413, "that request body is far too large")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HttpError(400, f"body is not JSON: {exc}") from None
        if not isinstance(data, dict):
            raise HttpError(400, "body must be a JSON object")
        return data

    def _upload(self) -> None:
        from . import multipart

        ctype = self.headers.get("Content-Type") or ""
        length = self.headers.get("Content-Length")
        try:
            parts = multipart.parse(
                self.rfile, ctype, int(length) if length else None,
                self.app.lib.uploads, MAX_UPLOAD_BYTES)
        except multipart.PayloadTooLarge as exc:
            raise HttpError(413, str(exc)) from None
        except multipart.MultipartError as exc:
            raise HttpError(400, str(exc)) from None

        try:
            file_part = next((p for p in parts if p.is_file and p.name == "file"), None)
            if file_part is None or file_part.path is None:
                raise HttpError(400, "no file was attached to that upload")
            if file_part.size == 0:
                raise HttpError(400, "that file is empty")
            meta = self.app.upload(file_part.name, file_part.filename or "",
                                   file_part.path)
        finally:
            multipart.cleanup(parts)
        self._json(meta)

    def _events(self, job_id: str, query: dict) -> None:
        job = self.app.queue.get(store.safe_id(job_id))
        if job is None:
            raise HttpError(404, "no such job")
        try:
            start = int((query.get("from") or ["0"])[0])
        except ValueError:
            start = 0
        last = self.headers.get("Last-Event-ID")
        if last and last.isdigit():
            start = int(last) + 1

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.close_connection = True

        index = start
        try:
            self.wfile.write(b"retry: 2000\n\n")
            for event in job.follow(start, timeout=SSE_TIMEOUT):
                if event is None:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                payload = json.dumps(event).encode("utf8")
                self.wfile.write(b"id: %d\ndata: %s\n\n" % (index, payload))
                self.wfile.flush()
                index += 1
            self.wfile.write(b"event: end\ndata: {}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _download(self, rid: str, name: str, as_download: bool) -> None:
        path, ctype = self.app.lib.remix_file(rid, name)
        meta = self.app.lib.remix_meta(rid)
        stem = store.display_name(meta.get("title") or "remix").replace('"', "")
        suffix = name.split("remix", 1)[1] or ".mp3"
        extra = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
        if as_download:
            extra["Content-Disposition"] = (
                f'attachment; filename="{stem} (fourfloor){suffix}"')
        self._send_file(path, ctype, extra)

    def _send_file(self, path: Path, ctype: str, extra: dict) -> None:
        """Send a file, honouring a single ``Range`` so audio can seek."""
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes=") and "," not in rng:
            first, _, last = rng[6:].partition("-")
            try:
                if first:
                    start = int(first)
                    end = int(last) if last else size - 1
                elif last:                                   # suffix range
                    start = max(0, size - int(last))
            except ValueError:
                start, end = 0, size - 1
            else:
                end = min(end, size - 1)
                if start > end or start >= size:
                    self._send(416, b"", "text/plain",
                               {"Content-Range": f"bytes */{size}"})
                    return
                status = HTTPStatus.PARTIAL_CONTENT
                extra = {**extra, "Content-Range": f"bytes {start}-{end}/{size}"}

        n = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(n))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as fh:
            fh.seek(start)
            left = n
            while left > 0:
                chunk = fh.read(min(256 * 1024, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _static(self, path: str) -> None:
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        if "/" in name or name.startswith("."):
            raise HttpError(404, "no such page")
        target = WEB_DIR / name
        if not target.is_file():
            raise HttpError(404, "no such page")
        ctype = STATIC_TYPES.get(target.suffix.lower()) \
            or mimetypes.guess_type(name)[0] or "application/octet-stream"
        body = target.read_bytes()
        self._send(200, body, ctype, {"Cache-Control": "no-cache"})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    verbose = False

    def __init__(self, addr, app: App, verbose: bool = False) -> None:
        super().__init__(addr, Handler)
        self.app = app
        self.verbose = verbose

    def handle_error(self, request, client_address) -> None:
        """Don't print a stack trace because a browser hung up.

        An <audio> element opens a connection, reads what it wants and resets
        the socket; so does every reload during an SSE stream. socketserver's
        default is to dump a traceback for each one, which makes a working app
        look like it is failing.
        """
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError,
                            ConnectionAbortedError, TimeoutError)):
            return
        if self.verbose:
            super().handle_error(request, client_address)
        else:
            print(f"  request failed: {exc.__class__.__name__}: {exc}", file=sys.stderr)


def make_server(port: int = 4444, home: str | os.PathLike | None = None,
                verbose: bool = False) -> Server:
    """Build a server bound to 127.0.0.1 (``port=0`` picks a free one)."""
    return Server(("127.0.0.1", port), App(Library(home)), verbose=verbose)


def serve(port: int = 4444, open_browser: bool = False,
          home: str | os.PathLike | None = None, c: ui.C | None = None) -> int:
    """Run the app until interrupted. Returns a process exit code."""
    c = c or ui.C(ui.supports_color())
    try:
        srv = make_server(port, home, verbose=bool(os.environ.get("FOURFLOOR_DEBUG")))
    except OSError as exc:
        print(ui.error(c, f"could not listen on port {port}: {exc}"))
        print(f"  something else is probably using it; try "
              f"`fourfloor serve --port {port + 1}`")
        return 1

    url = f"http://127.0.0.1:{srv.server_address[1]}"
    print(ui.header(c, "serve"))
    print()
    print(ui.kv(c, "app", c.cyan(url)))
    print(ui.kv(c, "library", c.grey(str(srv.app.lib.home))))
    print(ui.kv(c, "stop", c.grey("control-c")))
    print()
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.4), webbrowser.open(url)),
                         daemon=True).start()
    try:
        srv.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\n" + ui.kv(c, "stopped", c.grey("bye")))
    finally:
        srv.shutdown()
        srv.server_close()
        srv.app.close()
    return 0


def free_port() -> int:
    """An unused localhost port, for tests and ``--port 0``."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
