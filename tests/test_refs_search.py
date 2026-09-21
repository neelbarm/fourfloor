"""Finding the original: what the search asks for, and how it ranks the answers.

No network. The one test that exercises the yt-dlp command line replaces the
runner with a function that records the arguments and hands back the JSON the
real binary would have printed, which is the whole contract this module has
with it.
"""

from __future__ import annotations

import json

import pytest

from fourfloor.refs import search as S
from fourfloor.refs import titles as T


def cand(title: str, uploader: str = "", duration: float = 200.0,
         url: str = "") -> S.Candidate:
    return S.Candidate(url=url or f"https://www.youtube.com/watch?v={abs(hash(title)) % 10**9}",
                       title=title, uploader=uploader, duration=duration)


WANTED = T.parse("Don Toliver - E85 (Kosuk Remix) [FREE DOWNLOAD]")


# ---------------------------------------------------------------------------
# comparing titles
# ---------------------------------------------------------------------------

def test_a_title_matches_itself() -> None:
    assert S.title_similarity("Gwen Stefani The Sweet Escape",
                              "Gwen Stefani - The Sweet Escape") == pytest.approx(1.0)


def test_the_junk_is_stripped_from_both_sides_before_comparing() -> None:
    plain = S.title_similarity("Gwen Stefani The Sweet Escape",
                               "Gwen Stefani - The Sweet Escape")
    dressed = S.title_similarity(
        "Gwen Stefani The Sweet Escape",
        "Gwen Stefani - The Sweet Escape (Official Music Video) HQ #throwback")
    assert dressed == pytest.approx(plain, abs=0.01)


def test_a_different_song_by_the_same_artist_is_not_the_same_title() -> None:
    assert S.title_similarity("Don Toliver E85", "Don Toliver - Body") < 0.75


@pytest.mark.parametrize("seconds, expect", [
    (0.0, 0.6),        # unknown duration is neither rewarded nor punished
    (30.0, 0.1),
    (120.0, 1.0),
    (215.0, 1.0),
    (360.0, 1.0),
    (3600.0, 0.11),
])
def test_only_a_song_length_is_plausible(seconds, expect) -> None:
    assert S.duration_fit(seconds) == pytest.approx(expect, abs=0.06)


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------

def test_the_official_upload_beats_everything_else() -> None:
    ranked = S.rank(WANTED, [
        cand("Don Toliver - E85 (Kosuk Extended Remix) FREE DOWNLOAD"),
        cand("Don Toliver - E85 [Official Audio]"),
        cand("Don Toliver - E85 (sped up)"),
        cand("Don Toliver E85 1 hour loop", duration=3600.0),
    ])
    assert ranked[0].title == "Don Toliver - E85 [Official Audio]"
    assert ranked[0].score > 0.8


@pytest.mark.parametrize("title, word", [
    ("Don Toliver - E85 (Some Guy Remix)", "remix"),
    ("Don Toliver - E85 (Bootleg)", "bootleg"),
    ("Don Toliver - E85 sped up", "sped up"),
    ("Don Toliver - E85 slowed + reverb", "slowed"),
    ("Don Toliver - E85 nightcore", "nightcore"),
    ("Don Toliver - E85 8d audio", "8d"),
    ("Don Toliver - E85 cover", "cover"),
    ("Don Toliver - E85 live at Coachella", "live"),
    ("Don Toliver - E85 karaoke", "karaoke"),
    ("Don Toliver - E85 instrumental", "instrumental"),
    ("Don Toliver - E85 x Something mashup", "mashup"),
])
def test_the_words_that_give_away_an_edit_cost_it(title, word) -> None:
    clean = S.score_candidate(WANTED, cand("Don Toliver - E85"))
    dirty = S.score_candidate(WANTED, cand(title))
    assert word in dirty.penalties
    assert dirty.score < clean.score


def test_a_candidate_credited_to_the_remixer_is_the_remix_we_already_have() -> None:
    got = S.score_candidate(WANTED, cand("Don Toliver - E85 (Kosuk Edit)"))
    assert any("Kosuk" in p for p in got.penalties)
    assert got.score < 0.2


def test_a_topic_channel_is_the_labels_own_upload() -> None:
    plain = S.score_candidate(WANTED, cand("E85", uploader="somebody"))
    topic = S.score_candidate(WANTED, cand("E85", uploader="Don Toliver - Topic"))
    assert "topic channel" in topic.bonuses
    assert topic.score > plain.score


def test_the_artists_own_channel_helps_a_little_and_no_more() -> None:
    """It broke the ranking once by helping a lot: everything an artist ever
    released is on his channel, and only one of them is the right record."""
    right = S.score_candidate(WANTED, cand("Don Toliver - E85 [Official Audio]",
                                           uploader="Stranger"))
    wrong = S.score_candidate(WANTED, cand("Don Toliver - Body [Official Visualizer]",
                                           uploader="Don Toliver"))
    assert "artist's own channel" in wrong.bonuses
    assert right.score > wrong.score


def test_ranking_drops_the_same_link_twice() -> None:
    one = cand("Don Toliver - E85", url="https://youtu.be/abc")
    two = cand("Don Toliver - E85 [Official Audio]", url="https://youtu.be/abc")
    assert len(S.rank(WANTED, [one, two])) == 1


def test_an_hour_long_upload_loses_to_a_three_minute_one() -> None:
    ranked = S.rank(WANTED, [cand("Don Toliver - E85", duration=3600.0),
                             cand("Don Toliver - E85", duration=190.0,
                                  url="https://youtu.be/x")])
    assert ranked[0].duration == 190.0


# ---------------------------------------------------------------------------
# the search itself
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_ytdlp(monkeypatch):
    """Record the command lines, answer with a playlist of search results."""
    from fourfloor import fetch as fetch_mod

    seen: list[list[str]] = []

    def fake_run(cmd, timeout, on_line=None):
        seen.append(list(cmd))
        query = cmd[-1].split(":", 1)[1]
        payload = {"_type": "playlist", "title": query, "entries": [
            {"id": "aaa", "title": f"{query} [Official Audio]", "duration": 200,
             "uploader": "Somebody", "url": "https://www.youtube.com/watch?v=aaa"},
            {"id": "bbb", "title": f"{query} (Some Guy Remix)", "duration": 260,
             "uploader": "Some Guy", "url": "https://www.youtube.com/watch?v=bbb"},
        ]}
        return 0, json.dumps(payload)

    monkeypatch.setattr(fetch_mod, "ytdlp", lambda: "/usr/bin/true")
    monkeypatch.setattr(fetch_mod, "_run", fake_run)
    return seen


def test_the_search_asks_yt_dlp_for_n_results(fake_ytdlp) -> None:
    S._entries("Don Toliver E85", 5, 30.0)
    assert fake_ytdlp[0][-1] == "ytsearch5:Don Toliver E85"
    assert "--flat-playlist" in fake_ytdlp[0]
    assert "--dump-single-json" in fake_ytdlp[0]


def test_a_query_with_newlines_in_it_cannot_become_a_second_argument(fake_ytdlp) -> None:
    S._entries("Don Toliver\nE85\r--rm-cache-dir", 5, 30.0)
    assert fake_ytdlp[0][-1] == "ytsearch5:Don Toliver E85 --rm-cache-dir"
    assert len([a for a in fake_ytdlp[0] if a.startswith("ytsearch")]) == 1


def test_a_convincing_first_query_stops_the_search_early(fake_ytdlp, monkeypatch) -> None:
    parsed = T.parse("Gwen Stefani - The Sweet Escape (BOSEP Remix)")
    found = S.find_original(parsed, pause=0.0)
    assert found[0].score >= 0.8
    assert len(fake_ytdlp) == 1              # it did not need the other queries


def test_a_search_that_finds_nothing_convincing_tries_every_query(monkeypatch) -> None:
    from fourfloor import fetch as fetch_mod

    seen: list[str] = []

    def fake_run(cmd, timeout, on_line=None):
        seen.append(cmd[-1])
        return 0, json.dumps({"_type": "playlist", "title": "x", "entries": [
            {"id": "z", "title": "something else entirely", "duration": 200,
             "url": "https://www.youtube.com/watch?v=z"}]})

    monkeypatch.setattr(fetch_mod, "ytdlp", lambda: "/usr/bin/true")
    monkeypatch.setattr(fetch_mod, "_run", fake_run)
    parsed = T.parse("Some Artist - Some Track (X Remix)")
    found = S.find_original(parsed, pause=0.0)
    assert len(seen) == len(T.search_queries(parsed))
    assert found and found[0].score < 0.5


def test_a_failing_search_is_reported_as_one_sentence(monkeypatch) -> None:
    from fourfloor import fetch as fetch_mod

    monkeypatch.setattr(fetch_mod, "ytdlp", lambda: "/usr/bin/true")
    monkeypatch.setattr(fetch_mod, "_run",
                        lambda cmd, timeout, on_line=None:
                        (1, "ERROR: Unable to download API page: network is unreachable"))
    with pytest.raises(S.SearchError) as exc:
        S._entries("anything", 5, 30.0)
    assert "check the network" in str(exc.value)
