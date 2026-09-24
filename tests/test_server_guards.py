"""The web server's edges: names, lengths, other sites, and what is left on disk.

Nothing here renders audio. Remixes that need to exist are written straight
into a throwaway library, and the one remix that runs has its renderer
replaced, so this file is quick enough to run on every change to the server.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from urllib.parse import unquote

import pytest

from fourfloor import server, store
from fourfloor.server import App, make_server
from tests.test_server import Client

UNICODE_TITLE = "Beyoncé – Don’t Stop Жизнь 東京"


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    s = make_server(0, tmp_path_factory.mktemp("guard-home"))
    thread = threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    yield s
    s.shutdown()
    s.server_close()
    s.app.close()


@pytest.fixture(scope="module")
def client(srv):
    return Client(f"http://127.0.0.1:{srv.server_address[1]}")


def fake_remix(lib: store.Library, title: str, body: bytes = b"ID3 fake mp3 body") -> str:
    """A finished remix on disk, as the library lists one, without rendering."""
    rid, d = lib.create_remix()
    (d / "remix.mp3").write_bytes(body)
    lib.save_remix_meta(rid, {"title": title, "created": time.time()})
    return rid


def raw_http(port: int, request: bytes, timeout: float = 5.0) -> bytes:
    """Send bytes, read until the server closes: what is really on the wire."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(request)
        out = b""
        try:
            while chunk := sock.recv(65536):
                out += chunk
        except (ConnectionResetError, TimeoutError, socket.timeout):
            pass
        return out


# ---------------------------------------------------------------------------
# download names
# ---------------------------------------------------------------------------

def test_a_download_of_a_title_that_is_not_latin1_is_the_mp3(srv, client) -> None:
    """A curly apostrophe used to send a 200 audio/mpeg whose body was JSON."""
    body = b"ID3" + bytes(range(256)) * 8
    rid = fake_remix(srv.app.lib, UNICODE_TITLE, body)
    status, headers, got = client.request(f"/api/remixes/{rid}/remix.mp3?download=1")
    assert status == 200
    assert headers["Content-Type"] == "audio/mpeg"
    assert got == body
    cd = headers["Content-Disposition"]
    assert cd.startswith("attachment;")
    cd.encode("latin-1")                                  # the header is sendable
    star = cd.split("filename*=UTF-8''", 1)[1]
    assert unquote(star) == f"{UNICODE_TITLE} (fourfloor).mp3"
    plain = cd.split('filename="', 1)[1].split('"', 1)[0]
    assert plain.isascii() and plain.endswith(" (fourfloor).mp3")
    assert "Don't Stop" in plain and "Beyonce" in plain


def test_the_ascii_name_never_carries_a_quote_or_control_character() -> None:
    cd = server.content_disposition('a\\b";\x7f’x', ".wav")
    plain = cd.split('filename="', 1)[1].rsplit('"', 1)[0]
    assert '"' not in plain and "\\" not in plain and "\x7f" not in plain
    assert server.content_disposition("東京", ".mp3").startswith(
        'attachment; filename="remix (fourfloor).mp3"')


def test_an_error_after_the_headers_began_does_not_send_a_second_response(
        srv, monkeypatch) -> None:
    """The client must see a failure, never a 200 with an error for a body."""
    def half_then_boom(self, rid):
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        raise RuntimeError("boom")

    monkeypatch.setattr(server.Handler, "_reference", half_then_boom)
    rid = store.new_id()
    got = raw_http(srv.server_address[1],
                   f"GET /api/remixes/{rid}/reference HTTP/1.1\r\n"
                   f"Host: 127.0.0.1\r\n\r\n".encode())
    assert b"200 OK" not in got
    assert got.count(b"HTTP/1.") <= 1
    assert b"boom" not in got


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("length", ["-1", "banana", "1e3", " -5"])
def test_a_bad_content_length_is_a_400_not_a_read_to_eof(srv, length) -> None:
    head = (f"POST /api/fetch HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            f"Content-Type: application/json\r\nContent-Length: {length}\r\n\r\n")
    # the socket stays open after the body: a read(-1) would hang until timeout
    with socket.create_connection(("127.0.0.1", srv.server_address[1]),
                                  timeout=5) as sock:
        sock.sendall(head.encode() + b'{"url": "https://example.com/"}')
        start = time.time()
        got = sock.recv(65536)
    assert time.time() - start < 4
    assert got.startswith(b"HTTP/1.1 400"), got[:80]
    assert b"Content-Length" in got


def test_the_json_routes_insist_on_json(client) -> None:
    for path in ("/api/fetch", "/api/remix",
                 f"/api/remixes/{store.new_id()}/feedback"):
        status, _, body = client.request(
            path, "POST", b'{"url": "https://example.com/a"}',
            {"Content-Type": "text/plain;charset=UTF-8"})
        assert status == 415, (path, body)


# ---------------------------------------------------------------------------
# other sites
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("headers", [
    {"Origin": "https://evil.example.com"},
    {"Origin": "null"},
    {"Origin": "http://127.0.0.1:1"},                   # right host, wrong port
    {"Sec-Fetch-Site": "cross-site"},
    {"Sec-Fetch-Site": "same-site"},
])
def test_another_site_cannot_change_anything(client, headers) -> None:
    for path, body, ctype in [
        ("/api/fetch", b'{"url": "https://example.com/a"}', "application/json"),
        ("/api/remix", b"{}", "application/json"),
        ("/api/upload", b"--x--\r\n", "multipart/form-data; boundary=x"),
    ]:
        status, _, out = client.request(path, "POST", body,
                                        {"Content-Type": ctype, **headers})
        assert status == 403, (path, headers, out)
    rid = fake_remix(client_lib(client), "keep me")
    status, _, _ = client.request(f"/api/remixes/{rid}", "DELETE", headers=headers)
    assert status == 403
    assert rid in [r["id"] for r in client.json("/api/remixes")[1]["remixes"]]


def client_lib(client) -> store.Library:
    return client._lib


@pytest.fixture(scope="module", autouse=True)
def _attach_lib(srv, client):
    client._lib = srv.app.lib


def test_the_page_itself_is_still_let_in(srv, client) -> None:
    port = srv.server_address[1]
    for origin, host in [(f"http://127.0.0.1:{port}", f"127.0.0.1:{port}"),
                         (f"http://localhost:{port}", f"localhost:{port}")]:
        status, _, out = client.request(
            "/api/fetch", "POST", b'{"url": "file:///etc/passwd"}',
            {"Content-Type": "application/json", "Origin": origin, "Host": host,
             "Sec-Fetch-Site": "same-origin"})
        assert status == 400, out                        # past the guard, to check_url
        assert b"http" in out


def test_the_queue_has_a_bound(tmp_path, monkeypatch) -> None:
    app = App(store.Library(tmp_path / "home"))
    try:
        monkeypatch.setattr(app.queue, "depth", lambda: server.MAX_PENDING)
        with pytest.raises(server.HttpError) as exc:
            app._room_in_queue()
        assert exc.value.status == 429
    finally:
        app.close()


# ---------------------------------------------------------------------------
# what a failure leaves on disk
# ---------------------------------------------------------------------------

def test_an_upload_that_will_not_decode_leaves_nothing_behind(srv, client) -> None:
    lib = srv.app.lib
    before = set(lib.sources.iterdir())
    for _ in range(3):
        status, out = client.upload("broken.mp3", b"this is not audio at all" * 200)
        assert status == 400, out
        assert "could not read" in out["error"]
    assert set(lib.sources.iterdir()) == before
    assert not list(lib.uploads.glob("*.part"))


def test_a_remix_that_fails_part_way_leaves_no_folder(tmp_path, monkeypatch) -> None:
    import fourfloor.remix as remix_mod

    lib = store.Library(tmp_path / "home")
    sid, target = lib.create_source("song.mp3")
    target.write_bytes(b"not decoded by the fake renderer")
    lib.save_source_meta(sid, {"id": sid, "name": "song.mp3", "title": "song"})

    def half_a_render(src, out, opts, style=None, progress=None):
        progress("render", "")
        Path(out).write_bytes(b"a full-length mp3")
        raise ValueError("session failed validation")

    monkeypatch.setattr(remix_mod, "remix", half_a_render)
    app = App(lib)
    try:
        started = app.start_remix({"source": sid, "stems": "hpss"})
        job = app.queue.get(started["job"])
        deadline = time.time() + 10
        while not job.is_finished and time.time() < deadline:
            time.sleep(0.02)
        assert job.state == "failed"
        assert "validation" in job.error
        assert not (lib.remixes / started["remix"]).exists()
    finally:
        app.close()


def _age(path: Path, seconds: float) -> None:
    old = time.time() - seconds
    for p in [path, *(path.iterdir() if path.is_dir() else [])]:
        os.utime(p, (old, old))


def test_startup_sweeps_folders_nothing_lists(tmp_path) -> None:
    lib = store.Library(tmp_path / "home")
    # orphans: a source that never analysed, a render cut off by a crash
    sid, src = lib.create_source("x.mp3")
    src.write_bytes(b"x" * 1000)
    rid, d = lib.create_remix()
    (d / "remix.mp3").write_bytes(b"y" * 1000)
    # keepers: a real remix, and a young folder another server may be writing
    keep = fake_remix(lib, "kept")
    young, yd = lib.create_remix()
    (yd / "remix.mp3").write_bytes(b"z")
    for p in (lib.sources / sid, d, lib.remixes / keep):
        _age(p, 7 * 3600)
    App(lib).close()
    assert not (lib.sources / sid).exists()
    assert not d.exists()
    assert (lib.remixes / keep).exists()
    assert yd.exists()


def test_stale_fetch_folders_are_swept_too(tmp_path) -> None:
    lib = store.Library(tmp_path / "home")
    stale = lib.uploads / "link-abc123"
    (stale / ".fourfloor-dl-x").mkdir(parents=True)
    (stale / ".fourfloor-dl-x" / "a.webm.part").write_bytes(b"x" * 100)
    fresh = lib.uploads / "link-def456"
    fresh.mkdir()
    _age(stale / ".fourfloor-dl-x", 2 * 3600)
    _age(stale, 2 * 3600)
    assert lib.sweep_uploads() == 1
    assert not stale.exists()
    assert fresh.exists()


def test_meta_is_flushed_before_it_replaces_the_old_one(tmp_path, monkeypatch) -> None:
    synced = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real(fd)))
    store._write_json(tmp_path / "meta.json", {"id": "x"})
    assert synced
    assert json.loads((tmp_path / "meta.json").read_text()) == {"id": "x"}


# ---------------------------------------------------------------------------
# links
# ---------------------------------------------------------------------------

class _Job:
    started = time.time()

    def emit(self, *a, **kw) -> None:
        pass


def test_a_link_that_lands_somewhere_local_is_refused_before_download(
        tmp_path, monkeypatch) -> None:
    from fourfloor import fetch as fetch_mod

    looked_up, downloaded = [], []
    monkeypatch.setattr(fetch_mod, "check_resolves", lambda url: looked_up.append(url))
    monkeypatch.setattr(fetch_mod, "probe", lambda url: {
        "title": "t", "webpage_url": "http://127.0.0.1:4444/api/config"})
    monkeypatch.setattr(fetch_mod, "fetch", lambda *a, **kw: downloaded.append(a))
    app = App(store.Library(tmp_path / "home"))
    try:
        with pytest.raises(fetch_mod.FetchError) as exc:
            app._run_fetch(_Job(), "https://example.com/redirects")
        assert "local address" in str(exc.value)
        assert looked_up == ["https://example.com/redirects"]
        assert downloaded == []
    finally:
        app.close()
