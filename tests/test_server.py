"""The web app: multipart parsing, id/path safety, the job queue, and HTTP.

The end-to-end test at the bottom drives the real server the way the browser
does -- upload the fixture, start a remix, consume the SSE stream to the end,
download the mp3 -- and is the one that would catch a break in the wiring
between any two of the pieces above.
"""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

from fourfloor import jobs, multipart, store
from fourfloor.server import make_server

# ---------------------------------------------------------------------------
# multipart
# ---------------------------------------------------------------------------

BOUNDARY = "----fourfloorTestBoundary"


def body_for(parts: list[tuple[str, str | None, bytes]], boundary: str = BOUNDARY) -> bytes:
    """Build a multipart body: (name, filename or None, content)."""
    out = []
    for name, filename, content in parts:
        head = f'Content-Disposition: form-data; name="{name}"'
        if filename is not None:
            head += f'; filename="{filename}"'
        head += "\r\nContent-Type: " + ("audio/mpeg" if filename else "text/plain")
        out.append(f"--{boundary}\r\n{head}\r\n\r\n".encode() + content + b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out)


def parse_body(body: bytes, tmp_path: Path, limit: int = 8 << 20,
               boundary: str = BOUNDARY, length: int | None = -1):
    return multipart.parse(
        io.BytesIO(body), f"multipart/form-data; boundary={boundary}",
        len(body) if length == -1 else length, tmp_path, limit)


def test_multipart_splits_files_from_fields(tmp_path: Path) -> None:
    body = body_for([("file", "a song.mp3", b"\x00\x01audio"), ("form", None, b"club")])
    parts = parse_body(body, tmp_path)
    assert [p.name for p in parts] == ["file", "form"]
    f, field = parts
    assert f.is_file and f.filename == "a song.mp3" and f.content_type == "audio/mpeg"
    assert f.path.read_bytes() == b"\x00\x01audio"
    assert not field.is_file and field.text == "club"


def test_multipart_streams_a_body_larger_than_one_chunk(tmp_path: Path) -> None:
    """The regression that shipped broken: a file bigger than one read.

    The preamble scan used to drop the buffer it had just filled, boundary and
    all, and then find only the closing boundary -- which parses as a payload
    with no parts at all rather than as an error.
    """
    payload = bytes(range(256)) * (multipart.CHUNK // 128)      # 2 chunks' worth
    body = body_for([("file", "big.wav", payload)])
    parts = parse_body(body, tmp_path)
    assert len(parts) == 1
    assert parts[0].size == len(payload)
    assert parts[0].path.read_bytes() == payload


def test_multipart_keeps_content_that_looks_like_a_boundary(tmp_path: Path) -> None:
    payload = b"x\r\n--" + BOUNDARY[:-3].encode() + b"\r\nnot a boundary\r\n"
    parts = parse_body(body_for([("file", "t.wav", payload)]), tmp_path)
    assert parts[0].path.read_bytes() == payload


@pytest.mark.parametrize("filename, expect", [
    ('plain.mp3', 'plain.mp3'),
    ('../../etc/passwd', '../../etc/passwd'),      # kept verbatim: it is data
    ('semi;colon.mp3', 'semi;colon.mp3'),
])
def test_multipart_reads_awkward_filenames(tmp_path: Path, filename, expect) -> None:
    parts = parse_body(body_for([("file", filename, b"x")]), tmp_path)
    assert parts[0].filename == expect


def test_multipart_rejects_a_missing_boundary(tmp_path: Path) -> None:
    with pytest.raises(multipart.MultipartError):
        multipart.parse(io.BytesIO(b"hello"), "multipart/form-data", 5, tmp_path, 1 << 20)
    with pytest.raises(multipart.MultipartError):
        multipart.parse(io.BytesIO(b"hello"), "application/json", 5, tmp_path, 1 << 20)


def test_multipart_enforces_the_limit_while_reading(tmp_path: Path) -> None:
    """The cap holds whether or not the headers are honest about the size."""
    body = body_for([("file", "big.wav", b"z" * 4096)])
    with pytest.raises(multipart.PayloadTooLarge):           # caught mid-read
        parse_body(body, tmp_path, limit=1024)
    with pytest.raises(multipart.PayloadTooLarge):           # refused up front
        parse_body(body, tmp_path, limit=1024, length=1 << 30)
    with pytest.raises(multipart.PayloadTooLarge):           # no header at all
        parse_body(body, tmp_path, limit=1024, length=None)
    assert not list(tmp_path.glob("*.part")), "temporaries survived a failed parse"


def test_multipart_caps_a_plain_field(tmp_path: Path) -> None:
    with pytest.raises(multipart.PayloadTooLarge):
        parse_body(body_for([("note", None, b"n" * (multipart.MAX_FIELD_BYTES + 10))]),
                   tmp_path)


def test_multipart_rejects_a_truncated_part(tmp_path: Path) -> None:
    body = body_for([("file", "t.wav", b"data")])[:-40]
    with pytest.raises(multipart.MultipartError):
        parse_body(body, tmp_path, length=None)
    assert not list(tmp_path.glob("*.part"))


# ---------------------------------------------------------------------------
# ids and paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "..", "../../etc/passwd", "a/b", "", "ZZZZZZZZZZZZZZZZ", "0123456789abcde",
    "0123456789abcdef0", "0123456789ABCDEF", ".", "\x00" * 16, None, 7,
])
def test_safe_id_rejects_anything_we_did_not_mint(bad) -> None:
    with pytest.raises(store.NotFound):
        store.safe_id(bad)


def test_safe_id_accepts_a_minted_id() -> None:
    for _ in range(50):
        assert store.safe_id(store.new_id())


@pytest.mark.parametrize("bad", ["meta.json", "../meta.json", "wave.json", "remix.MP3",
                                 "remix.mp3/../meta.json", "/etc/passwd", ""])
def test_remix_file_is_an_allowlist(tmp_path: Path, bad) -> None:
    lib = store.Library(tmp_path)
    rid, d = lib.create_remix()
    (d / "remix.mp3").write_bytes(b"x")
    (d / "meta.json").write_text("{}")
    with pytest.raises(store.NotFound):
        lib.remix_file(rid, bad)
    assert lib.remix_file(rid, "remix.mp3")[0] == d / "remix.mp3"


@pytest.mark.parametrize("name, ok", [
    ("song.mp3", True), ("SONG.MP3", True), ("a.m4a", True), ("a.flac", True),
    ("a.aiff", True), ("a.exe", False), ("noext", False), ("a.mp3.exe", False),
    ("a.ogg", False),
])
def test_upload_extensions_are_an_allowlist(name, ok) -> None:
    if ok:
        assert store.safe_ext(name) in store.UPLOAD_EXTS
    else:
        with pytest.raises(ValueError):
            store.safe_ext(name)


def test_the_upload_path_is_ours_not_the_browsers(tmp_path: Path) -> None:
    lib = store.Library(tmp_path)
    sid, target = lib.create_source("../../../evil.mp3")
    assert target.name == "source.mp3"
    assert target.parent == lib.sources / sid
    assert lib.sources in target.parents


def test_display_name_is_safe_to_show(tmp_path: Path) -> None:
    assert store.display_name("../../secret/song.mp3") == "song.mp3"
    assert "\n" not in store.display_name("a\nb.mp3")
    assert store.display_name("") == "untitled"
    assert len(store.display_name("x" * 400)) <= 120


# ---------------------------------------------------------------------------
# the job queue
# ---------------------------------------------------------------------------

@pytest.fixture
def queue():
    q = jobs.JobQueue()
    yield q
    q.close()


def test_jobs_run_one_at_a_time_in_order(queue) -> None:
    live, order, lock = [], [], threading.Lock()

    def work(name):
        def fn(job):
            with lock:
                live.append(name)
                assert len(live) == 1, f"{live} ran at once"
            time.sleep(0.05)
            with lock:
                live.remove(name)
                order.append(name)
            return name
        return fn

    ids = [store.new_id() for _ in range(4)]
    for i, jid in enumerate(ids):
        queue.submit(jid, work(i))
    for jid in ids:
        assert wait_for(queue.get(jid))
    assert order == [0, 1, 2, 3]


def wait_for(job, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.is_finished:
            return True
        time.sleep(0.01)
    return False


def test_a_follower_that_arrives_late_still_sees_every_event(queue) -> None:
    def fn(job):
        for name in ("analyse", "render"):
            job.emit("phase", name=name)
        return {"ok": True}

    job = queue.submit(store.new_id(), fn)
    assert wait_for(job)
    seen = [e for e in job.follow(0, timeout=2) if e]
    assert [e["type"] for e in seen] == ["queued", "start", "phase", "phase", "done"]
    assert seen[-1]["result"] == {"ok": True}
    # and a follower can resume from an offset
    assert [e["type"] for e in job.follow(3, timeout=2) if e] == ["phase", "done"]


def test_a_follower_watching_live_work_finishes_with_the_job(queue) -> None:
    gate = threading.Event()
    job = queue.submit(store.new_id(), lambda j: (gate.wait(5), "done")[1])
    events = []

    def watch():
        events.extend(e for e in job.follow(0, timeout=10) if e)

    t = threading.Thread(target=watch)
    t.start()
    time.sleep(0.2)
    gate.set()
    t.join(10)
    assert not t.is_alive(), "follow() did not return when the job finished"
    assert events[-1]["type"] == "done"


def test_a_failing_job_reports_the_message(queue) -> None:
    def fn(job):
        raise ValueError("--bpm 900 is out of range")

    job = queue.submit(store.new_id(), fn)
    assert wait_for(job)
    assert job.state == jobs.FAILED
    last = [e for e in job.follow(0, timeout=2) if e][-1]
    assert last["type"] == "error"
    assert "out of range" in last["message"]


def test_phase_timer_closes_each_phase_it_opened() -> None:
    job = jobs.Job(id="x")
    job.started = time.time()
    timer = jobs.PhaseTimer(job)
    timer("analyse", "decoding")
    timer("render", "96 bars")
    timer.close()
    kinds = [(e["type"], e.get("name")) for e in job.events]
    assert kinds == [("phase", "analyse"), ("phase_done", "analyse"),
                     ("phase", "render"), ("phase_done", "render")]
    assert all("elapsed" in e for e in job.events if e["type"] == "phase_done")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app(tmp_path_factory):
    """A live server on a throwaway library, shared by the HTTP tests."""
    srv = make_server(0, tmp_path_factory.mktemp("fourfloor-home"))
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    yield Client(f"http://127.0.0.1:{srv.server_address[1]}")
    srv.shutdown()
    srv.server_close()
    srv.app.close()


class Client:
    """The few HTTP moves the tests need, with errors as values."""

    def __init__(self, base: str) -> None:
        self.base = base

    def request(self, path: str, method: str = "GET", data: bytes | None = None,
                headers: dict | None = None, timeout: float = 120.0):
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers), exc.read()

    def json(self, path: str, method: str = "GET", payload=None, **kw):
        data = json.dumps(payload).encode() if payload is not None else None
        head = {"Content-Type": "application/json"} if data else {}
        status, _, body = self.request(path, method, data, head, **kw)
        return status, json.loads(body or b"null")

    def upload(self, filename: str, content: bytes, field: str = "file"):
        boundary = uuid.uuid4().hex
        body = body_for([(field, filename, content)], boundary)
        status, _, out = self.request(
            "/api/upload", "POST", body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=300)
        return status, json.loads(out)

    def events(self, job_id: str, timeout: float = 300.0):
        """Consume an SSE stream to its end, returning the events."""
        req = urllib.request.Request(f"{self.base}/api/jobs/{job_id}/events")
        out = []
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                if line.startswith(b"data: "):
                    event = json.loads(line[6:])
                    if "type" in event:
                        out.append(event)
        return out


def test_the_page_and_its_assets_are_served(app) -> None:
    for path, needle in [("/", b"<title>fourfloor</title>"), ("/app.js", b"startPlayer"),
                         ("/app.css", b"--pink"), ("/favicon.svg", b"<svg")]:
        status, headers, body = app.request(path)
        assert status == 200, path
        assert needle in body, path
        assert "nosniff" in headers.get("X-Content-Type-Options", "")
    assert b"cdn" not in app.request("/")[2].lower()


@pytest.mark.parametrize("path", [
    "/../pyproject.toml", "/..%2fpyproject.toml", "/web/app.js", "/.gitignore",
    "/nope.html", "/app.js/../../pyproject.toml",
])
def test_static_serving_cannot_escape_the_web_folder(app, path) -> None:
    assert app.request(path)[0] == 404


def test_only_localhost_hosts_are_answered(app) -> None:
    status, _, body = app.request("/api/config", headers={"Host": "evil.example.com"})
    assert status == 403
    assert b"localhost" in body


def test_config_describes_what_the_cli_allows(app) -> None:
    status, cfg = app.json("/api/config")
    assert status == 200
    from fourfloor.arrange import FORMS
    from fourfloor.remix import PHASES
    assert cfg["forms"] == list(FORMS)
    assert cfg["phases"] == list(PHASES)
    assert cfg["bpm_range"] == [60.0, 200.0]
    assert cfg["max_upload_mb"] == 200
    assert isinstance(cfg["demucs"], bool)


def test_an_upload_that_is_not_audio_is_refused_kindly(app) -> None:
    status, out = app.upload("notes.txt", b"hello")
    assert status == 400
    assert "mp3" in out["error"]
    status, out = app.upload("song.mp3", b"")
    assert status == 400


def test_upload_without_a_file_part_is_refused(app) -> None:
    status, out = app.upload("song.mp3", b"x", field="something-else")
    assert status == 400
    assert "no file" in out["error"]


def test_unknown_routes_answer_json(app) -> None:
    status, out = app.json("/api/nope")
    assert status == 404 and "error" in out


# -- the whole thing --------------------------------------------------------

TARGET_BPM = 124.0


@pytest.fixture(scope="module")
def uploaded(app, fixture_path):
    status, meta = app.upload("lofi-7.mp3", Path(fixture_path).read_bytes())
    assert status == 200, meta
    return meta


def test_upload_returns_an_id_and_the_analysis(uploaded) -> None:
    assert store.safe_id(uploaded["id"])
    a = uploaded["analysis"]
    assert 60 < a["tempo"]["bpm"] < 200
    assert a["key"]["camelot"] and a["sections"]
    assert 124 <= uploaded["suggested_bpm"] <= 132
    assert len(uploaded["wave"]) == 900
    assert uploaded["name"] == "lofi-7.mp3"


@pytest.mark.parametrize("payload, needle", [
    ({"bpm": "900"}, "out of range"),
    ({"bpm": "banana"}, "invalid float"),
    ({"form": "disco"}, "invalid choice"),
    ({"key": "H#"}, "unrecognised key"),
    ({"length": "0"}, "greater than zero"),
    ({"style": "no-such-style"}, "no style profile"),
    ({"swing": "9"}, "out of range"),
])
def test_the_web_app_cannot_ask_for_what_the_cli_would_refuse(app, uploaded, payload,
                                                              needle) -> None:
    status, out = app.json("/api/remix", "POST", {"source": uploaded["id"], **payload})
    assert status == 400, out
    assert needle in out["error"], out


def test_a_remix_of_an_unknown_source_is_a_404(app) -> None:
    for bad in ("../../etc", store.new_id()):
        status, out = app.json("/api/remix", "POST", {"source": bad})
        assert status == 404, out


def test_remix_end_to_end_over_http(app, uploaded, tmp_path) -> None:
    """Upload → remix → SSE → download, exactly as the browser does it."""
    status, started = app.json("/api/remix", "POST", {
        "source": uploaded["id"], "bpm": str(TARGET_BPM), "length": "2:00",
        "form": "radio", "stems": "hpss",
    })
    assert status == 200, started
    rid = started["job"]

    events = app.events(rid)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "queued" and kinds[-1] == "done", kinds
    assert "error" not in kinds, [e for e in events if e["type"] == "error"]
    from fourfloor.remix import PHASES
    assert [e["name"] for e in events if e["type"] == "phase"] == list(PHASES)
    assert [e["name"] for e in events if e["type"] == "phase_done"] == list(PHASES)
    assert all(e["elapsed"] >= 0 for e in events if e["type"] == "phase_done")

    result = events[-1]["result"]
    assert result["id"] == rid
    assert result["bpm"] == pytest.approx(TARGET_BPM, abs=0.01)

    # the mp3 is real audio of the length the session promises
    status, headers, mp3 = app.request(f"/api/remixes/{rid}/remix.mp3")
    assert status == 200 and headers["Content-Type"] == "audio/mpeg"
    assert len(mp3) > 50_000
    out = tmp_path / "downloaded.mp3"
    out.write_bytes(mp3)

    status, _, raw = app.request(f"/api/remixes/{rid}/remix.session.json")
    session = json.loads(raw)
    assert session["bpm"] == 124.0
    assert session["first_downbeat_sec"] == 0.0
    from fourfloor import session as session_mod
    assert session_mod.validate(session) == []

    from fourfloor.audio import decode
    assert decode(out).duration == pytest.approx(session["duration"], abs=0.25)

    # …and the page's own view of it
    status, detail = app.json(f"/api/remixes/{rid}")
    assert status == 200
    assert len(detail["wave"]) == 1400
    assert detail["session"]["bars"] == session["bars"]
    assert detail["source"]["id"] == uploaded["id"]
    assert detail["meta"]["files"] == ["remix.mp3", "remix.plan.json",
                                       "remix.session.json", "remix.wav"]

    status, listing = app.json("/api/remixes")
    assert rid in [r["id"] for r in listing["remixes"]]


def test_audio_is_served_with_ranges_so_a_player_can_seek(app) -> None:
    rid = json.loads(app.request("/api/remixes")[2])["remixes"][0]["id"]
    status, headers, body = app.request(f"/api/remixes/{rid}/remix.mp3",
                                        headers={"Range": "bytes=10-109"})
    assert status == 206
    assert len(body) == 100
    assert headers["Content-Range"].startswith("bytes 10-109/")
    status, headers, _ = app.request(f"/api/remixes/{rid}/remix.mp3",
                                     headers={"Range": "bytes=999999999-"})
    assert status == 416


def test_a_download_is_named_after_the_track(app) -> None:
    rid = json.loads(app.request("/api/remixes")[2])["remixes"][0]["id"]
    _, headers, _ = app.request(f"/api/remixes/{rid}/remix.mp3?download=1")
    assert "attachment" in headers["Content-Disposition"]
    assert "fourfloor" in headers["Content-Disposition"]


def test_deleting_a_remix_removes_it_from_the_library(app) -> None:
    rid = json.loads(app.request("/api/remixes")[2])["remixes"][0]["id"]
    status, out = app.json(f"/api/remixes/{rid}", "DELETE")
    assert status == 200 and out["deleted"] == rid
    assert rid not in [r["id"] for r in json.loads(
        app.request("/api/remixes")[2])["remixes"]]
    assert app.json(f"/api/remixes/{rid}")[0] == 404
    assert app.json(f"/api/remixes/{rid}", "DELETE")[0] == 404
