"""``fourfloor refs`` from the command line, and what it prints.

The reports are strings built out of a dozen fields that may or may not be
there, which is exactly the kind of code that breaks silently, so every one of
them is rendered here at least once.

The last test in this file is the only one in the suite that talks to YouTube.
It is skipped unless ``FOURFLOOR_NET_TESTS=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fourfloor import ui
from fourfloor.cli import main
from fourfloor.refs import report
from fourfloor.refs.ledger import Entry, Ledger

C = ui.C(False)


def an_entry(**fields) -> Entry:
    base = {"url": "https://www.youtube.com/watch?v=x", "slug": "artist-track-remixer",
            "status": "done", "title": "Artist - Track (Remixer Remix)",
            "artist": "Artist", "track": "Track", "remixer": "Remixer",
            "score": 0.61, "bpm_original": 146.0, "bpm_remix": 130.0,
            "semitones": -1, "lattice": "triplet", "treatment": "chopped",
            "kit": "artist-track-remixer", "original_title": "Artist - Track"}
    base.update(fields)
    return Entry(**base)


# ---------------------------------------------------------------------------
# the reports
# ---------------------------------------------------------------------------

def test_the_table_says_what_happened_to_every_link() -> None:
    text = report.list_report(
        [an_entry(), an_entry(slug="b", status="standalone", kit="b"),
         an_entry(slug="c", status="needs_review", score=0.33),
         an_entry(slug="d", status="failed", error="that track is private")],
        C, home=Path("/tmp/refs"))
    assert "artist-track-remixer" in text
    assert "146 -> 130 BPM" in text
    assert "-1 st" in text
    assert "triplet" in text and "chopped" in text
    assert "that track is private" in text
    assert "4 links" in text


def test_an_empty_folder_says_how_to_start() -> None:
    assert "refs add" in report.list_report([], C)


def test_the_review_shows_the_candidates_and_the_two_commands() -> None:
    entry = an_entry(status="needs_review", score=0.33, candidates=[
        {"title": "Artist - Track [Official Audio]", "uploader": "Artist",
         "duration": 214.0, "score": 0.92,
         "url": "https://www.youtube.com/watch?v=one"},
        {"title": "Artist - Track (Live)", "uploader": "Someone",
         "duration": 260.0, "score": 0.41,
         "url": "https://www.youtube.com/watch?v=two"}])
    text = report.review_report([entry], C)
    assert "https://www.youtube.com/watch?v=one" in text
    assert "rank 0.92" in text
    assert "refs accept artist-track-remixer" in text
    assert "refs reject artist-track-remixer" in text


def test_nothing_to_review_says_so() -> None:
    assert "nothing is waiting on you" in report.review_report([], C)


def test_the_learned_table_prints_its_cells() -> None:
    text = report.table_report({
        "n_files": 9, "table": {"n_pairs": 3, "straight_lock": 0.64, "cells": {
            "straight/continuous": {"n": 2, "tempo_ratio": 0.9, "vocal_kept": 0.85,
                                    "straight_fit": 0.78, "bpm": 128.0},
            "triplet/chopped": {"n": 1, "tempo_ratio": 0.88, "vocal_kept": 0.4,
                                "straight_fit": 0.52, "bpm": 130.0}}},
        "style": {"bpm": 129.5, "tempo_ratio": 0.9}}, C)
    assert "3 verified pairs" in text
    assert "straight -> continuous" in text
    assert "triplet -> chopped" in text
    assert "0.640" in text


def test_a_run_prints_one_line_per_event(capsys) -> None:
    run = report.Run(C)
    for kind, text in (("link", "https://x.example/1"), ("step", "parsed: A - B"),
                       ("ok", "filed"), ("warn", "no kit"), ("skip", "already done"),
                       ("dry", "would fetch"), ("error", "it broke")):
        run(kind, text)
    out = capsys.readouterr().out
    for needle in ("https://x.example/1", "parsed: A - B", "filed", "no kit",
                   "already done", "would fetch", "it broke"):
        assert needle in out


def test_a_quiet_run_prints_nothing(capsys) -> None:
    run = report.Run(C, quiet=True)
    run("link", "https://x.example/1")
    run("ok", "filed")
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def test_listing_an_empty_reference_folder_is_not_an_error(tmp_path, capsys) -> None:
    assert main(["refs", "list", "--refs", str(tmp_path)]) == 0
    assert "none yet" in capsys.readouterr().out


def test_the_list_can_be_json(tmp_path, capsys) -> None:
    Ledger(tmp_path / "ledger.json").put(an_entry())
    assert main(["refs", "list", "--refs", str(tmp_path), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["slug"] == "artist-track-remixer"


def test_review_shows_only_what_is_undecided(tmp_path, capsys) -> None:
    led = Ledger(tmp_path / "ledger.json")
    led.put(an_entry())
    led.put(an_entry(url="https://y.example/2", slug="waiting",
                     status="needs_review"))
    assert main(["refs", "review", "--refs", str(tmp_path), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["slug"] for r in rows] == ["waiting"]


def test_adding_nothing_is_refused_with_a_sentence(tmp_path, capsys) -> None:
    assert main(["refs", "add", "--refs", str(tmp_path)]) == 1
    assert "give me some links" in capsys.readouterr().err


def test_accepting_a_name_that_is_not_there_is_refused(tmp_path, capsys) -> None:
    assert main(["refs", "accept", "nope", "--refs", str(tmp_path)]) == 1
    assert "nope" in capsys.readouterr().err


def test_a_dry_run_from_the_command_line_fetches_nothing(tmp_path, capsys,
                                                         monkeypatch) -> None:
    from fourfloor.refs import pipeline, search

    monkeypatch.setattr(search, "find_original", lambda parsed, **k: [
        search.Candidate(url="https://www.youtube.com/watch?v=orig",
                         title="Somebody - The Original", duration=200.0, score=0.9)])
    monkeypatch.setattr(pipeline.fetch_mod, "fetch", lambda *a, **k: pytest.fail(
        "a dry run must not download anything"))
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "lofi-7.mp3"
    source = tmp_path / "Somebody - The Original (Neel Remix).mp3"
    source.write_bytes(fixture.read_bytes())

    assert main(["refs", "add", str(source), "--refs", str(tmp_path / "refs"),
                 "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would fetch" in out
    assert not (tmp_path / "refs" / "pairs").exists()


# ---------------------------------------------------------------------------
# the real thing, once, when asked
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("FOURFLOOR_NET_TESTS") != "1",
                    reason="set FOURFLOOR_NET_TESTS=1 to search YouTube for real")
def test_the_search_finds_a_real_record_on_youtube() -> None:
    """One live search, no download: the ranking has to put the record first."""
    from fourfloor.refs import search, titles

    parsed = titles.parse("Gwen Stefani - The Sweet Escape (BOSEP Remix) FREE DL")
    found = search.find_original(parsed, per_query=5, pause=0.5)
    assert found, "the search came back empty"
    best = found[0]
    assert "sweet escape" in best.title.lower()
    assert best.score > 0.6
    assert 120.0 < best.duration < 360.0
    assert "bosep" not in best.title.lower()
