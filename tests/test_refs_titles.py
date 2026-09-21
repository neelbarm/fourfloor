"""What a remix title claims, pinned against the titles DJs actually post.

The parser is a pile of guesses about a format nobody agreed on, so the only
honest test is the corpus: forty-odd titles in the shapes they arrive in --
bracketed credits, stacked credits, download bait, label prefixes, ``feat.``
tails, SoundCloud's silent convention that the account is the credit -- and one
assertion each about the three facts the pipeline needs.

The five titles at the top are not invented. They are the filenames sitting in
``~/Desktop/Mixpilot``, the ones that started this module, and four of the five
have no brackets in them at all: ``The_Sweet_Escape_BOSEP_Remix`` has to be cut
somewhere and the cut is a guess. Those cases are pinned at the level the
parser can actually be held to -- the artist, that *somebody* was credited,
that the split was flagged :attr:`~fourfloor.refs.titles.Parsed.ambiguous` --
plus the one thing that matters downstream: that the queries it produces still
contain the real record. Where the guess is visibly wrong the test says so in a
comment rather than pretending the output is right.

Nothing here touches the network, reads a file or decodes audio; it is string
in, string out.
"""

from __future__ import annotations

import re

import pytest

from fourfloor.refs import titles

# ---------------------------------------------------------------------------
# the five off the desktop
# ---------------------------------------------------------------------------

#: The real filenames, and what the parser makes of each. Two of them it reads
#: correctly; three it guesses at, and the guesses are marked below.
DESKTOP = [
    # Correct: the dash gives the artist, and the credit guess takes one name
    # token -- but the credit here is two names ("JLOOD Kosuk"), so the first
    # of them is left stuck on the end of the track. Hence the ``E85``
    # fallback query, and hence test_the_hard_filename_still_finds_the_record.
    ("Don_Toliver_-_E85_JLOOD_Kosuk_Extended_Remix_FREE_DOWNLOAD_KLICKAUD.mp3",
     "Don Toliver", "E85 JLOOD", "Kosuk", "remix", ["E85"]),
    # Correct, and the interesting part: the remixer is the SoundCloud handle
    # "its murph", found by walking left through lowercase tokens. The second
    # credit ("PLAYSTATE Edit") is somebody's edit of the remix and is dropped.
    ("Never_Be_Like_You_its_murph_Remix_PLAYSTATE_Edit_Free_DL_KLICKAUD.mp3",
     "", "Never Be Like You", "its murph", "remix", ["Never Be Like"]),
    # Wrong, and unavoidably so: this one is ``Track - Artist``, the reverse of
    # the convention, so "Stay Fly" is read as the artist. The reversed query
    # is what rescues it -- test_a_guess_is_searched_for_both_ways round.
    ("Stay_Fly_-_Three_6_Mafia_Bittersweet_Remix_KLICKAUD.mp3",
     "Stay Fly", "Three 6 Mafia", "Bittersweet", "remix", ["Three 6"]),
    # Correct: one name token before the keyword, and no artist to be had.
    ("The_Sweet_Escape_BOSEP_Remix_FREE_DL_KLICKAUD.mp3",
     "", "The Sweet Escape", "BOSEP", "remix", ["The Sweet"]),
    # Not a remix at all, and nothing left once KLICKAUD goes but the name.
    ("babybabyyy_KLICKAUD.mp3", "", "babybabyyy", "", "", []),
]


@pytest.mark.parametrize("name, artist, track, remixer, kind, fallbacks", DESKTOP)
def test_a_file_off_the_desktop_reads_the_way_we_read_it(
        name, artist, track, remixer, kind, fallbacks) -> None:
    p = titles.parse(name)
    assert (p.artist, p.track, p.remixer, p.kind) == (artist, track, remixer, kind)
    assert p.track_fallbacks == fallbacks
    assert ".mp3" not in p.track and "KLICKAUD" not in p.label()
    assert "_" not in p.label()


@pytest.mark.parametrize("name", [row[0] for row in DESKTOP])
def test_a_file_off_the_desktop_admits_it_is_guessing(name) -> None:
    """Every one of these lacks the brackets that would make it certain."""
    assert titles.parse(name).ambiguous is True


def test_the_hard_filename_still_finds_the_record() -> None:
    """The point of the fallback queries, on the title that needs them.

    ``E85_JLOOD_Kosuk_Extended_Remix`` is a two-name credit and the guesser
    takes one name, so the track comes out as "E85 JLOOD" -- wrong, and known
    to be wrong, which is why the split is flagged ambiguous. What has to
    survive that is the search: one of the queries is the artist and the real
    track, with nothing else in it.
    """
    p = titles.parse(DESKTOP[0][0])
    assert p.artist == "Don Toliver"
    assert p.remixer, "somebody was credited"
    assert p.ambiguous is True
    assert "Don Toliver E85 official audio" in titles.search_queries(p)


def test_a_guess_is_searched_for_both_ways_round() -> None:
    """``Stay Fly - Three 6 Mafia`` is the pair backwards, and we cannot tell."""
    p = titles.parse(DESKTOP[2][0])
    queries = titles.search_queries(p)
    assert "Stay Fly Three 6 Mafia official audio" == queries[0]
    assert "Three 6 Mafia Stay Fly official audio" in queries


def test_a_soundcloud_handle_survives_being_lowercase() -> None:
    """``its murph`` is two tokens and both are lowercase; it is still a name."""
    p = titles.parse(DESKTOP[1][0])
    assert p.remixer == "its murph"
    assert p.track == "Never Be Like You"
    assert p.kind == "remix" and p.is_remix


# ---------------------------------------------------------------------------
# brackets, which are the easy case
# ---------------------------------------------------------------------------

BRACKETED = [
    # title, artist, track, remixer, kind
    ("Tame Impala - Let It Happen (Soulwax Remix)",
     "Tame Impala", "Let It Happen", "Soulwax", "remix"),
    ("Midnight City [Skrillex Edit]",
     "", "Midnight City", "Skrillex", "edit"),
    ("Kanye West - Flashing Lights - Ben Bohmer Bootleg",
     "Kanye West", "Flashing Lights", "Ben Bohmer", "bootleg"),
    ("Robyn - Dancing On My Own {Kaytranada Flip}",
     "Robyn", "Dancing On My Own", "Kaytranada", "flip"),
    ("Don Toliver - E85 (JLOOD & Kosuk Extended Remix)",
     "Don Toliver", "E85", "JLOOD & Kosuk", "remix"),
    ("Disclosure - You & Me (Flume VIP)",
     "Disclosure", "You & Me", "Flume", "vip"),
    ("Fred again.. - Rumble (Skrillex & Flowdan VIP)",
     "Fred again..", "Rumble", "Skrillex & Flowdan", "vip"),
    ("Daft Punk - One More Time (Kaytranada Rework)",
     "Daft Punk", "One More Time", "Kaytranada", "rework"),
    ("ODESZA - Say My Name [Rufus Du Sol Remix]",
     "ODESZA", "Say My Name", "Rufus Du Sol", "remix"),
]


@pytest.mark.parametrize("title, artist, track, remixer, kind", BRACKETED)
def test_a_bracketed_credit_says_who_made_it(title, artist, track, remixer,
                                             kind) -> None:
    p = titles.parse(title)
    assert (p.artist, p.track, p.remixer, p.kind) == (artist, track, remixer, kind)
    assert p.is_remix
    assert p.credited_to_uploader is False, "the title named somebody"


def test_a_qualifier_is_not_a_name() -> None:
    """``Extended`` and ``Jersey Club`` describe the edit, they do not sign it."""
    assert titles.credit_name("JLOOD & Kosuk Extended Remix") == "JLOOD & Kosuk"
    assert titles.credit_name("Club Cheval Remix") == "Club Cheval"
    assert titles.credit_name("Extended Mix") == ""
    assert titles.credit_name("Remix") == ""
    p = titles.parse("Cassö x RAYE - Prada (Jersey Club Remix)")
    assert p.remixer == "" and p.kind == "remix"


def test_stacked_credits_keep_both_and_the_first_one_wins() -> None:
    """A remix of a remix: the nearer credit is the one we go looking for."""
    p = titles.parse("Fred again.. - Delilah (Skream Remix) (Joy Orbison Edit)")
    assert p.credits == ["Skream", "Joy Orbison"]
    assert p.remixer == "Skream" and p.kind == "remix"


# ---------------------------------------------------------------------------
# the advertising
# ---------------------------------------------------------------------------

JUNK = [
    # title, artist, track, remixer
    ("Drake - Passionfruit (Nora En Pure Remix) FREE DOWNLOAD",
     "Drake", "Passionfruit", "Nora En Pure"),
    ("Jamie xx - Gosh (Peggy Gou Edit) *FREE DL*",
     "Jamie xx", "Gosh", "Peggy Gou"),
    ("Flume - Never Be Like You (Disclosure Remix) KLICKAUD",
     "Flume", "Never Be Like You", "Disclosure"),
    ("Duke Dumont - Ocean Drive (Vintage Culture Remix) BUY = FREE DOWNLOAD",
     "Duke Dumont", "Ocean Drive", "Vintage Culture"),
    ("Lana Del Rey - Summertime Sadness (Cedric Gervais Remix) [Official Audio]",
     "Lana Del Rey", "Summertime Sadness", "Cedric Gervais"),
    ("Gorillaz - Feel Good Inc (Official Music Video)",
     "Gorillaz", "Feel Good Inc", ""),
    ("Justice - D.A.N.C.E. (Fred Falke Remix) HQ",
     "Justice", "D.A.N.C.E.", "Fred Falke"),
    ("Calvin Harris - Slide (Mura Masa Flip) #housemusic #remix",
     "Calvin Harris", "Slide", "Mura Masa"),
    ("SG Lewis - Impact (Jax Jones Remix) www.freedl.com",
     "SG Lewis", "Impact", "Jax Jones"),
    ("Adele - Hello (Marshmello Remix) https://fanlink.to/x",
     "Adele", "Hello", "Marshmello"),
    ("Yeah Yeah Yeahs - Heads Will Roll (A-Trak Remix) OUT NOW",
     "Yeah Yeah Yeahs", "Heads Will Roll", "A-Trak"),
    ("Nelly Furtado - Say It Right (Kaytranada Remix) | KLICKAUD",
     "Nelly Furtado", "Say It Right", "Kaytranada"),
    ("PREMIERE: Roisin Murphy - Incapable (DJ Koze Remix)",
     "Roisin Murphy", "Incapable", "DJ Koze"),
    ("FREE DL - Rihanna - Kiss It Better (Kaytranada Remix)",
     "Rihanna", "Kiss It Better", "Kaytranada"),
    ("Defected Records | Fisher - Losing It (Chris Lake Edit)",
     "Fisher", "Losing It", "Chris Lake"),
]

#: Nothing on this list is part of anybody's song.
NOT_IN_A_SONG = ("free", "download", "klickaud", "www", "http", "#", "buy",
                 "premiere", "official", "hq", "out now", "|", "*")


@pytest.mark.parametrize("title, artist, track, remixer", JUNK)
def test_the_advertising_comes_off(title, artist, track, remixer) -> None:
    p = titles.parse(title)
    assert (p.artist, p.track, p.remixer) == (artist, track, remixer)
    left = f"{p.artist} {p.track} {p.remixer}".lower()
    for junk in NOT_IN_A_SONG:
        assert junk not in left, f"{junk!r} survived in {left!r}"


@pytest.mark.parametrize("title, artist, track, remixer", JUNK)
def test_no_query_goes_out_carrying_download_bait(title, artist, track,
                                                  remixer) -> None:
    for q in titles.search_queries(titles.parse(title)):
        for junk in NOT_IN_A_SONG:
            if junk == "official":
                continue                                # "official audio" is ours
            assert junk not in q.lower(), f"{junk!r} survived in {q!r}"


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

FEATURES = [
    # title, artist, track, feat
    ("Calvin Harris - Feels feat. Pharrell Williams (Mistajam Remix)",
     "Calvin Harris", "Feels", "Pharrell Williams"),
    ("Major Lazer - Cold Water ft. Justin Bieber (Kaskade Remix)",
     "Major Lazer", "Cold Water", "Justin Bieber"),
    ("DJ Snake featuring Justin Bieber - Let Me Love You (Marshmello Remix)",
     "DJ Snake", "Let Me Love You", "Justin Bieber"),
    ("David Guetta - Titanium (feat. Sia) [Nicky Romero Remix]",
     "David Guetta", "Titanium", "Sia"),
]


@pytest.mark.parametrize("title, artist, track, feat", FEATURES)
def test_a_feature_is_kept_but_set_aside(title, artist, track, feat) -> None:
    """The guest is worth recording and worth keeping out of the search."""
    p = titles.parse(title)
    assert (p.artist, p.track, p.feat) == (artist, track, feat)
    assert "feat" not in p.track.lower() and "ft." not in p.track.lower()
    guest = feat.split()[-1].lower()
    for q in titles.search_queries(p):
        assert guest not in q.lower(), q


# ---------------------------------------------------------------------------
# soundcloud: the account is the credit
# ---------------------------------------------------------------------------

UPLOADED = [
    # title, uploader, kind -- a remix that names nobody was made by whoever
    # posted it, which on SoundCloud is nearly always true
    ("Never Be Like You (Remix)", "its murph", "remix"),
    ("Aaliyah - One In A Million (Remix)", "kaytranada", "remix"),
    ("Chase & Status - Alive (Bootleg)", "kosuk", "bootleg"),
    ("Cassö x RAYE - Prada (Jersey Club Remix)", "salute", "remix"),
]


@pytest.mark.parametrize("title, uploader, kind", UPLOADED)
def test_a_remix_credited_to_nobody_is_the_uploaders(title, uploader,
                                                     kind) -> None:
    anonymous = titles.parse(title)
    assert anonymous.remixer == "" and anonymous.kind == kind
    assert anonymous.credited_to_uploader is True

    p = titles.parse(title, uploader)
    assert p.remixer == uploader
    assert p.uploader == uploader
    assert p.is_remix


VERSION_TAGS = [
    # title, uploader, kind -- an artist's own version tag, not a credit
    ("Swedish House Mafia - One (Extended Mix)", "Big Label Records", "mix"),
    ("MK - 17 (Extended Mix)", "Area10", "mix"),
    ("Deadmau5 - Strobe (Original Mix)", "House Channel", ""),
    ("Camelphat - Cola (Original Mix)", "House Channel", ""),
    ("Fleetwood Mac - Dreams (Remastered)", "Warner Records", ""),
]


@pytest.mark.parametrize("title, uploader, kind", VERSION_TAGS)
def test_a_version_tag_does_not_invent_a_remixer(title, uploader, kind) -> None:
    """``(Extended Mix)`` is the artist's own. Crediting the channel for one
    would turn every label upload into a remix by the label."""
    p = titles.parse(title, uploader)
    assert p.credited_to_uploader is False
    assert p.remixer == ""
    assert p.kind == kind
    assert uploader not in p.label()


# ---------------------------------------------------------------------------
# everything else people type
# ---------------------------------------------------------------------------

PLAIN = [
    # title, artist, track, kind
    ("Midnight City", "", "Midnight City", ""),
    ("Four Tet - Baby", "Four Tet", "Baby", ""),
    ("Bicep - Glue [Official Audio]", "Bicep", "Glue", ""),
    ("Burial – Archangel", "Burial", "Archangel", ""),               # en dash
    ("Röyksopp — Eple (Fred Falke Remix)", "Röyksopp", "Eple", "remix"),  # em dash
    ("Sébastien Tellier - La Ritournelle (Todd Terje Edit)",
     "Sébastien Tellier", "La Ritournelle", "edit"),
    ("Céline Dion - My Heart Will Go On (Tiësto Remix)",
     "Céline Dion", "My Heart Will Go On", "remix"),
    ("A - B - C - D Remix", "A", "B C", "remix"),        # the third part is a credit
    ("Artist -", "", "Artist", ""),
    ("", "", "", ""),
    ("-", "", "", ""),
    ("!!!???", "", "!!!???", ""),
]


@pytest.mark.parametrize("title, artist, track, kind", PLAIN)
def test_a_title_with_nothing_special_in_it_still_parses(title, artist, track,
                                                         kind) -> None:
    p = titles.parse(title)
    assert (p.artist, p.track, p.kind) == (artist, track, kind)
    assert p.raw == title.strip()
    assert p.to_dict()["track"] == track


def test_a_dash_and_nothing_else_is_not_a_crash() -> None:
    for title in ("", "   ", "-", " - ", "---", "()", "[]", "|", "."):
        p = titles.parse(title)
        assert isinstance(p.track, str)
        assert p.slug()
        assert isinstance(titles.search_queries(p), list)


def test_normalising_is_the_first_thing_that_happens() -> None:
    assert titles.normalize("Artist_-_Track_Name.mp3") == "Artist - Track Name"
    assert titles.normalize("  Artist – Track — Thing.WAV ") == "Artist - Track - Thing"
    assert titles.normalize("") == ""
    assert titles.normalize("a  b   c") == "a b c"


# ---------------------------------------------------------------------------
# the name on disk
# ---------------------------------------------------------------------------

ALL_TITLES = ([row[0] for row in DESKTOP] + [row[0] for row in BRACKETED]
              + [row[0] for row in JUNK] + [row[0] for row in FEATURES]
              + [row[0] for row in PLAIN])


@pytest.mark.parametrize("title", ALL_TITLES)
def test_a_slug_is_a_filename_we_would_type(title) -> None:
    slug = titles.parse(title).slug()
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", slug), slug
    assert len(slug) <= 56
    assert slug == titles.parse(title).slug(), "the same title, a different name"


def test_a_slug_survives_a_title_that_is_only_junk() -> None:
    assert titles.parse("").slug() == "remix"
    assert titles.parse("!!!???").slug() == "remix"
    assert titles.parse("KLICKAUD.mp3").slug() == "remix"


def test_a_slug_drops_the_accents_rather_than_the_word() -> None:
    assert titles.parse("Sébastien Tellier - La Ritournelle (Todd Terje Edit)"
                        ).slug() == "sebastien-tellier-la-ritournelle-todd-terje"


def test_a_label_reads_like_a_person_wrote_it() -> None:
    assert (titles.parse("Tame Impala - Let It Happen (Soulwax Remix)").label()
            == "Tame Impala - Let It Happen (Soulwax Remix)")
    assert titles.parse("Bicep - Glue [Official Audio]").label() == "Bicep - Glue"
    assert titles.parse("Midnight City").label() == "Midnight City"


# ---------------------------------------------------------------------------
# what we go and search for
# ---------------------------------------------------------------------------

def test_the_official_upload_is_what_we_ask_for_first() -> None:
    q = titles.search_queries(titles.parse(
        "Tame Impala - Let It Happen (Soulwax Remix)"))
    assert q[0] == "Tame Impala Let It Happen official audio"
    assert q[1] == "Tame Impala - Let It Happen"
    assert not any("Soulwax" in x for x in q), "we want the original, not the remix"


def test_a_title_with_no_artist_asks_for_the_song_itself() -> None:
    q = titles.search_queries(titles.parse("The_Sweet_Escape_BOSEP_Remix_KLICKAUD.mp3"))
    assert q[0] == "The Sweet Escape official audio"
    assert "The Sweet Escape original song" in q


def test_the_list_never_runs_past_the_limit() -> None:
    p = titles.parse(DESKTOP[0][0])                     # the one with fallbacks
    assert len(titles.search_queries(p)) == 5
    assert len(titles.search_queries(p, limit=2)) == 2
    assert titles.search_queries(p, limit=1) == ["Don Toliver E85 JLOOD official audio"]
    assert titles.search_queries(p, limit=0) == []


def test_the_same_query_is_not_asked_for_twice() -> None:
    for title in ALL_TITLES:
        q = titles.search_queries(titles.parse(title))
        assert len(q) == len({x.lower() for x in q}), title


def test_a_title_we_could_not_read_asks_for_nothing() -> None:
    assert titles.search_queries(titles.parse("")) == []
    assert titles.search_queries(titles.parse("-")) == []
