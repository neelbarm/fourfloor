"""The feedback endpoints, driven over real HTTP the way the page drives them.

One module-scoped server and one rendered remix, because rendering is the slow
part; every test here is a request against that.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from fourfloor import feedback, store
from fourfloor.server import make_server
from test_server import Client               # noqa: E402 - pythonpath has tests/


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    srv = make_server(0, tmp_path_factory.mktemp("fourfloor-feedback-home"))
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    yield Client(f"http://127.0.0.1:{srv.server_address[1]}")
    srv.shutdown()
    srv.server_close()
    srv.app.close()


@pytest.fixture(scope="module")
def built(app, fixture_path):
    """One source and two remixes of it, so A/B has something to compare."""
    status, source = app.upload("lofi-7.mp3", Path(fixture_path).read_bytes())
    assert status == 200, source
    ids = []
    for bpm in ("124", "126"):
        status, started = app.json("/api/remix", "POST", {
            "source": source["id"], "bpm": bpm, "length": "2:00",
            "form": "radio", "stems": "hpss"})
        assert status == 200, started
        events = app.events(started["job"])
        assert events[-1]["type"] == "done", events[-1]
        ids.append(started["job"])
    return source, ids


# ---------------------------------------------------------------------------
# GET/POST /api/remixes/<id>/feedback
# ---------------------------------------------------------------------------

def test_a_fresh_remix_has_an_empty_feedback_document(app, built) -> None:
    _, (rid, _) = built
    status, data = app.json(f"/api/remixes/{rid}/feedback")
    assert status == 200
    assert data["markers"] == [] and data["votes"] == []
    assert data["rating"] == {"stars": None, "verdict": "", "at": 0.0}


def test_posting_a_marker_stores_it_beside_the_render(app, built, tmp_path) -> None:
    _, (rid, other) = built
    status, detail = app.json(f"/api/remixes/{rid}")
    duration = detail["session"]["duration"]

    status, data = app.json(f"/api/remixes/{rid}/feedback", "POST", {
        "marker": {"time": duration / 2, "category": "drums-fake",
                   "note": "hats sound like a preset"}})
    assert status == 200, data
    assert len(data["markers"]) == 1
    marker = data["markers"][0]
    assert marker["category"] == "drums-fake"
    assert marker["bar"] > 0, "the bar is worked out from the session grid"
    assert marker["slot"], "and the slot from the plan"

    # …and it is on disk, in the remix's own folder
    home = Path(app.json("/api/config")[1]["home"]).expanduser()
    folder = store.Library(home).remix_dir(rid)
    assert json.loads((folder / "feedback.json").read_text())["markers"] == \
        data["markers"]


def test_a_marker_reaches_the_session_file_the_dj_app_downloads(app, built) -> None:
    _, (rid, _) = built
    app.json(f"/api/remixes/{rid}/feedback", "POST", {
        "marker": {"time": 3.0, "category": "too-quiet", "note": "kick vanishes"}})
    _, _, raw = app.request(f"/api/remixes/{rid}/remix.session.json")
    session = json.loads(raw)
    notes = [f for f in session.get("feedback", []) if f["category"] == "too-quiet"]
    assert notes and notes[-1]["note"] == "kick vanishes"

    from fourfloor import session as session_mod
    assert session_mod.validate(session) == []


def test_a_rating_and_a_verdict_round_trip(app, built) -> None:
    _, (rid, _) = built
    status, data = app.json(f"/api/remixes/{rid}/feedback", "POST",
                            {"stars": 4, "verdict": "the drop lands"})
    assert status == 200
    assert data["rating"]["stars"] == 4
    assert data["rating"]["verdict"] == "the drop lands"
    # re-rating appends rather than replaces
    status, data = app.json(f"/api/remixes/{rid}/feedback", "POST", {"stars": 2})
    assert data["rating"]["stars"] == 2
    assert data["rating"]["verdict"] == "the drop lands"
    assert len(data["ratings"]) >= 2


def test_a_vote_records_the_other_side(app, built) -> None:
    _, (rid, other) = built
    status, data = app.json(f"/api/remixes/{rid}/feedback", "POST", {
        "vote": {"other": other, "prefer": "b", "reason": "126 breathes",
                 "other_label": "take 2"}})
    assert status == 200
    vote = data["votes"][-1]
    assert vote["other"] == other and vote["winner"] == "other"
    assert vote["reason"] == "126 breathes"


@pytest.mark.parametrize("payload, needle", [
    ({"marker": {"time": 1, "category": "sounds-bad"}}, "not a feedback category"),
    ({"marker": {"time": "soon", "category": "good"}}, "time in seconds"),
    ({"marker": {"time": -1, "category": "good"}}, "time in seconds"),
    ({"stars": 9}, "1 to 5"),
    ({"stars": "lots"}, "whole number"),
    ({"vote": {"other": "x", "prefer": "maybe"}}, "'a' or 'b'"),
    ({}, "nothing to save"),
    ({"note": "just talking"}, "nothing to save"),
])
def test_the_page_cannot_store_something_the_digest_could_not_group(
        app, built, payload, needle) -> None:
    _, (rid, _) = built
    status, out = app.json(f"/api/remixes/{rid}/feedback", "POST", payload)
    assert status == 400, out
    assert needle in out["error"], out


@pytest.mark.parametrize("bad", ["../../etc", "nope", "0" * 16])
def test_feedback_for_an_id_we_never_minted_is_a_404(app, bad) -> None:
    assert app.json(f"/api/remixes/{bad}/feedback")[0] == 404
    assert app.json(f"/api/remixes/{bad}/feedback", "POST",
                    {"stars": 3})[0] == 404


def test_feedback_is_not_a_downloadable_file_name(app, built) -> None:
    """The download allowlist is still the allowlist; the route is separate."""
    _, (rid, _) = built
    assert app.request(f"/api/remixes/{rid}/feedback.json")[0] == 404
    assert app.request(f"/api/remixes/{rid}/meta.json")[0] == 404


# ---------------------------------------------------------------------------
# GET /api/feedback — what the engineers read
# ---------------------------------------------------------------------------

def test_the_whole_library_of_notes_comes_back_at_once(app, built) -> None:
    _, (rid, other) = built
    app.json(f"/api/remixes/{other}/feedback", "POST", {
        "marker": {"time": 2.0, "category": "boring", "note": "same 8 bars"},
        "stars": 3})
    status, data = app.json("/api/feedback")
    assert status == 200
    ids = [r["id"] for r in data["remixes"]]
    assert rid in ids and other in ids
    row = next(r for r in data["remixes"] if r["id"] == other)
    assert row["rating"]["stars"] == 3
    assert row["feedback"]["markers"][0]["note"] == "same 8 bars"
    assert [c["code"] for c in data["categories"]] == list(feedback.CATEGORIES)
    assert "listening notes" in data["digest"]


def test_the_digest_is_available_as_plain_text_for_a_terminal(app, built) -> None:
    _, (rid, _) = built
    status, headers, body = app.request("/api/feedback?text=1")
    assert status == 200
    assert headers["Content-Type"].startswith("text/plain")
    text = body.decode("utf8")
    assert "listening notes" in text and rid in text
    assert "Drums fake" in text


def test_the_config_tells_the_page_which_chips_to_draw(app) -> None:
    _, cfg = app.json("/api/config")
    assert [c["code"] for c in cfg["feedback_categories"]] == list(feedback.CATEGORIES)


# ---------------------------------------------------------------------------
# what A/B can play
# ---------------------------------------------------------------------------

def test_a_remix_lists_what_it_can_be_compared_against(app, built) -> None:
    source, (rid, other) = built
    status, detail = app.json(f"/api/remixes/{rid}")
    assert status == 200
    partners = detail["partners"]
    kinds = [p["kind"] for p in partners]
    assert "source" in kinds, "you can always A/B against the track you dropped"
    assert other in [p["id"] for p in partners if p["kind"] == "remix"]
    assert rid not in [p["id"] for p in partners], "not against itself"
    assert all(p["same_source"] for p in partners), "both takes share a source"
    src = next(p for p in partners if p["kind"] == "source")
    assert src["url"] == f"/api/sources/{source['id']}/audio"


def test_the_source_audio_is_served_for_the_compare_player(app, built) -> None:
    source, _ = built
    status, headers, body = app.request(f"/api/sources/{source['id']}/audio")
    assert status == 200
    assert headers["Content-Type"] == "audio/mpeg"
    assert len(body) > 10_000
    # and it seeks, because the compare transport scrubs
    status, headers, chunk = app.request(f"/api/sources/{source['id']}/audio",
                                         headers={"Range": "bytes=0-99"})
    assert status == 206 and len(chunk) == 100


@pytest.mark.parametrize("bad", ["../../etc", "nope", "0" * 16])
def test_source_audio_is_only_reachable_by_an_id_we_minted(app, bad) -> None:
    assert app.request(f"/api/sources/{bad}/audio")[0] == 404


def test_no_reference_remix_on_disk_is_an_honest_404(app, built) -> None:
    _, (rid, _) = built
    status, out = app.json(f"/api/remixes/{rid}/reference")
    assert status == 404
    assert "reference" in out["error"]


def test_a_reference_remix_on_disk_is_offered_and_played(app, built, tmp_path,
                                                         monkeypatch) -> None:
    """With a pairs folder present, the human remix joins the partner list."""
    source, (rid, _) = built
    pairs = tmp_path / "house-refs" / "pairs"
    pairs.mkdir(parents=True)
    (pairs / f"{source['title']}.remix.mp3").write_bytes(b"\xff\xfb" + b"\x00" * 4096)
    monkeypatch.setenv("FOURFLOOR_REFS", str(tmp_path / "house-refs"))

    _, detail = app.json(f"/api/remixes/{rid}")
    ref = next((p for p in detail["partners"] if p["kind"] == "reference"), None)
    assert ref is not None, [p["kind"] for p in detail["partners"]]
    assert ref["url"] == f"/api/remixes/{rid}/reference"

    status, headers, body = app.request(ref["url"])
    assert status == 200 and headers["Content-Type"] == "audio/mpeg"
    assert len(body) == 4098
