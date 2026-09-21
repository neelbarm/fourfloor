"""What a remix title is actually telling you.

The titles DJs trade are not metadata, they are advertising: ``Don Toliver -
E85 (JLOOD & Kosuk Extended Remix) [FREE DOWNLOAD] | KLICKAUD``. Somewhere in
there are the three facts this pipeline needs -- the artist, the track, and who
remixed it -- and around them is a layer of label prefixes, download bait, site
names and version tags that has to come off before anything can be searched for.

The rules, in the order they are applied:

1. **Normalise.** Underscores become spaces when the string is a filename;
   fancy dashes become ``-``; the file extension goes.
2. **Strip the advertising.** ``FREE DOWNLOAD``, ``FREE DL``, ``KLICKAUD``,
   ``OUT NOW``, ``PREMIERE:``, ``[Official Audio]``, hashtags, and the rest of
   the list in :data:`NOISE`, wherever they appear.
3. **Read the brackets.** ``(… Remix)``, ``[… Edit]``, ``{… Bootleg}`` are
   credits; ``(feat. X)`` is a feature; anything left is a version tag. A credit
   with nothing but version words in it (``(Extended Mix)``) names no remixer.
4. **Split on the dash.** ``Artist - Track``. A third dashed part carrying a
   keyword is a credit too: ``Artist - Track - X Bootleg``.
5. **Guess, when there are no brackets.** ``The Sweet Escape BOSEP Remix`` has
   to be cut somewhere, and the cut is a guess: one name token, extended
   leftwards through lowercase tokens so handles like ``its murph`` survive. The
   guess is marked :attr:`Parsed.ambiguous`, and the queries it produces include
   shorter and shorter prefixes of the track so the search is not betting on it.
6. **SoundCloud's convention.** A remix whose credit names nobody, uploaded by
   somebody, was made by the uploader.

Nothing here is certain and nothing here needs to be: the parser's job is to
produce *searchable* text, and the verification step is what decides whether the
record it found is the same song.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

#: Words that make a bracketed group a remix credit rather than a version tag.
KEYWORDS = (
    "remix", "rmx", "edit", "bootleg", "flip", "rework", "refix", "remake",
    "mashup", "mash-up", "bounce", "dub", "vip", "remaster", "cover", "mix",
)

#: The kinds that name a person. ``mix``, ``remaster`` and ``cover`` do not:
#: an ``(Extended Mix)`` is the artist's own, so the uploader is not its author.
AUTHORED = frozenset({"remix", "rmx", "edit", "bootleg", "flip", "rework",
                      "refix", "remake", "mashup", "mash-up", "bounce", "vip"})

#: Words inside a credit that describe the *kind* of edit rather than its
#: author. ``(JLOOD & Kosuk Extended Remix)`` is by JLOOD & Kosuk.
QUALIFIERS = {
    "extended", "radio", "club", "festival", "private", "official", "original",
    "vip", "dub", "instrumental", "acapella", "acappella", "house", "afro",
    "tech", "techno", "trance", "dnb", "drum", "bass", "jersey", "drill",
    "hard", "hardstyle", "bounce", "uk", "garage", "amapiano", "baile",
    "slowed", "sped", "up", "reverb", "nightcore", "version", "master",
    "mastered", "cut", "full", "long", "short", "intro", "outro", "clean",
    "dirty", "free", "dl", "download", "bootleg", "edit", "remix", "mix",
    "rmx", "flip", "rework", "refix", "remake", "mashup", "remaster", "cover",
    "2023", "2024", "2025", "2026",
}

#: Junk, wherever it appears. Ordered longest-first where it matters, because
#: ``free download`` has to go before ``download`` would have a chance to --
#: and a whole URL goes before any of it, or ``www.freedl.com`` is left as
#: ``www. .com`` by the ``free dl`` rule biting a hole in the middle of it.
NOISE = (
    r"\bwww\.\S+", r"https?://\S+",
    r"buy\s*=\s*free\s*(download|dl)", r"free\s*(download|dl)\b",
    r"\bfree\b(?=\s*[\]\)\|]|$)", r"\bklickaud\b", r"\bklick\s*aud\b",
    r"download\s+in\s+(the\s+)?(description|bio|link)", r"\bout\s+now\b",
    r"\bofficial\s+(music\s+)?video\b", r"\bofficial\s+audio\b",
    r"\bofficial\s+visuali[sz]er\b", r"\bofficial\s+lyric\s+video\b",
    r"\blyrics?\s+video\b", r"\bvisuali[sz]er\b", r"\baudio\s+only\b",
    r"\bfull\s+(song|version|album)\b", r"\bhq\b", r"\bhd\b", r"\b4k\b",
    r"\bncs\s+release\b", r"\bno\s+copyright\b", r"\bcopyright\s+free\b",
    r"\bre-?upload\b", r"\bpitched\b", r"\bfree\s+to\s+use\b",
    r"\bsupport\s+the\s+artist\b", r"\bplease\s+support\b",
    r"\bsubscribe\b", r"\bnew\s+song\b", r"\btiktok\s+(song|version)\b",
    r"#\w+", r"\bprod\.?\s*by\s+[\w .'&-]+",
)

#: Prefixes that are a label, a premiere or a shout, not the artist.
PREFIXES = (
    "premiere", "premier", "exclusive", "free download", "free dl", "out now",
    "new", "video premiere", "first play", "klickaud", "fd", "free",
)

_EXT = re.compile(r"\.(mp3|m4a|wav|flac|aiff?|ogg|opus|webm|mp4)$", re.I)
_SPACE = re.compile(r"\s+")
_DASH = re.compile(r"\s+[-–—―~]+\s+")
_GROUP = re.compile(r"[\(\[\{]([^\(\)\[\]\{\}]*)[\)\]\}]")
_FEAT = re.compile(r"\b(feat\.?|ft\.?|featuring|with)\s+(.+)$", re.I)
_KEYWORD = re.compile(r"\b(" + "|".join(KEYWORDS) + r")\b", re.I)
_WORD = re.compile(r"[\w'&]+", re.UNICODE)
_SLUG_DROP = re.compile(r"[^a-z0-9]+")


def strip_accents(text: str) -> str:
    """``Beyoncé`` and ``Beyonce`` are the same search."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(ch))


_strip_accents = strip_accents          # the name the first draft used


def normalize(title: str) -> str:
    """A title reduced to plain text: no extension, no underscores, one space."""
    text = str(title or "").strip()
    text = _EXT.sub("", text)
    text = text.replace("｜", "|").replace("：", ":")
    text = text.replace("＂", '"').replace("“", '"').replace("”", '"')
    if "_" in text and text.count("_") >= text.count(" "):
        text = text.replace("_", " ")
    text = text.replace("–", "-").replace("—", "-").replace("―", "-")
    return _SPACE.sub(" ", text).strip()


def _denoise(text: str) -> str:
    out = text
    for pattern in NOISE:
        out = re.sub(pattern, " ", out, flags=re.I)
    # brackets and pipes left empty by the removals
    out = re.sub(r"[\(\[\{]\s*[\)\]\}]", " ", out)
    out = re.sub(r"\|\s*(?=\||$)", " ", out)
    out = out.replace("*", " ").replace("«", " ").replace("»", " ")
    return _SPACE.sub(" ", out).strip(" -|·•,")


def _strip_prefix(text: str) -> str:
    """Drop a leading ``PREMIERE:`` / ``Label |`` / ``FREE DL -`` style prefix."""
    out = text
    for _ in range(3):                                  # at most a few stacked
        m = re.match(r"^\s*([^:|]{1,40}?)\s*([:|])\s*(.+)$", out, flags=re.S)
        if not m:
            break
        head, sep, rest = m.group(1).strip(), m.group(2), m.group(3).strip()
        low = head.lower().strip(" .-")
        # A short name in front of a pipe, with an ``Artist - Track`` behind it,
        # is the label or the channel: ``Defected Records | Fisher - Losing It``.
        # The head must not itself be a dashed pair, or a title whose *tail* is
        # the pipe ("Artist - Track | Free DL") would lose its artist.
        named_source = (len(head.split()) <= 4
                        and _DASH.search(head) is None
                        and _DASH.search(rest) is not None
                        and (head.isupper() or sep == "|"))
        is_prefix = (low in PREFIXES
                     or any(low.startswith(p) for p in PREFIXES)
                     or named_source)
        if not is_prefix or not rest:
            break
        out = rest
    return out.strip()


def _tail_junk(text: str) -> str:
    """Trailing ``| site``-style leftovers once the known noise has gone."""
    out = text
    out = re.sub(r"\s*\|\s*[^|]{0,30}$", "", out) if out.count("|") else out
    return _SPACE.sub(" ", out).strip(" -|·•,")


def credit_name(group: str) -> str:
    """The author named by a credit, or ``""`` if it only names a version.

    ``JLOOD & Kosuk Extended Remix`` is by JLOOD & Kosuk; ``Extended Mix`` is by
    nobody. Qualifiers are peeled off the *end* only, so a remixer actually
    called "Club Cheval" keeps his name.
    """
    words = group.strip().split()
    while words and words[-1].lower().strip(".,'\"") in QUALIFIERS:
        words.pop()
    name = " ".join(words).strip(" -,&")
    return "" if name.lower() in QUALIFIERS else name


def _is_credit(group: str) -> bool:
    return bool(_KEYWORD.search(group))


def _lowercase_run(tokens: list[str], start: int) -> int:
    """Extend a name run leftwards through lowercase tokens (``its murph``)."""
    i = start
    while i > 0 and tokens[i - 1][:1].islower() and tokens[i - 1].lower() not in QUALIFIERS:
        i -= 1
    return i


@dataclass
class Parsed:
    """What a title claims, and how sure we are of it."""

    raw: str = ""
    uploader: str = ""
    artist: str = ""
    track: str = ""
    remixer: str = ""
    kind: str = ""
    feat: str = ""
    credits: list[str] = field(default_factory=list)
    ambiguous: bool = False
    track_fallbacks: list[str] = field(default_factory=list)

    @property
    def is_remix(self) -> bool:
        return bool(self.kind) and self.kind not in ("cover",)

    @property
    def credited_to_uploader(self) -> bool:
        """Whether the uploader should be read as the remixer.

        Only for the kinds that name an author. ``(Extended Mix)`` and
        ``(Remastered)`` are version tags an artist puts on his own record, and
        crediting the channel for one would invent a remixer out of the artist.
        """
        return not self.remixer and self.kind in AUTHORED

    def label(self) -> str:
        """One line for a person: ``Artist - Track (Remixer Remix)``."""
        head = f"{self.artist} - {self.track}" if self.artist else self.track
        if self.remixer:
            return f"{head} ({self.remixer} {self.kind or 'remix'.title()})".replace(
                "remix)", "Remix)")
        return head or self.raw

    def slug(self) -> str:
        """A filesystem name for this pair: ``artist-track-remixer``."""
        parts = [self.artist, self.track, self.remixer]
        text = _SLUG_DROP.sub("-", _strip_accents(" ".join(p for p in parts if p)).lower())
        text = re.sub(r"-+", "-", text).strip("-")[:56].strip("-")
        return text or "remix"

    def to_dict(self) -> dict:
        return asdict(self)


def parse(title: str, uploader: str = "") -> Parsed:
    """Read a remix title into artist, track and remixer.

    ``uploader`` is the channel or SoundCloud account. It is used only when the
    title says a remix was made and does not say by whom, which on SoundCloud is
    the normal case -- the account *is* the credit.
    """
    raw = str(title or "")
    text = _tail_junk(_denoise(_strip_prefix(normalize(raw))))
    p = Parsed(raw=raw.strip(), uploader=str(uploader or "").strip())

    # --- bracketed groups: credits, features, version tags -----------------
    groups = [g.strip() for g in _GROUP.findall(text)]
    body = _SPACE.sub(" ", _GROUP.sub(" ", text)).strip(" -|·•,")
    for g in groups:
        if not g:
            continue
        feat = _FEAT.match(g)
        if feat:
            p.feat = p.feat or feat.group(2).strip()
            continue
        if _is_credit(g):
            p.kind = p.kind or _bare_kind(g)
            name = credit_name(g)
            if name:
                p.credits.append(name)
        elif len(g.split()) <= 4 and not re.search(r"\d{4}", g):
            pass                                        # a version tag; drop it

    # --- the dash --------------------------------------------------------
    parts = [x.strip(" -|·•,") for x in _DASH.split(body) if x.strip(" -|·•,")]
    tail_credit = ""
    while len(parts) > 2 and _is_credit(parts[-1]):
        tail_credit = parts.pop()
        p.kind = p.kind or _bare_kind(tail_credit)
        name = credit_name(tail_credit)
        if name:
            p.credits.append(name)
    if len(parts) >= 2:
        p.artist, p.track = parts[0], " ".join(parts[1:])
    elif parts:
        p.track = parts[0]
        p.ambiguous = True
    else:
        p.track = body

    # a feature left in the plain text belongs to the artist, not the title
    feat = _FEAT.search(p.track)
    if feat:
        p.feat = p.feat or feat.group(2).strip()
        p.track = p.track[:feat.start()].strip(" -,")
    feat = _FEAT.search(p.artist)
    if feat:
        p.feat = p.feat or feat.group(2).strip()
        p.artist = p.artist[:feat.start()].strip(" -,")

    # --- no brackets, a keyword in the tail: guess where the credit starts --
    if not p.credits and _KEYWORD.search(p.track):
        guessed, track, fallbacks = _split_bare_credit(p.track)
        p.kind = p.kind or _bare_kind(text)
        if guessed:
            p.remixer, p.track, p.ambiguous = guessed, track, True
            p.track_fallbacks = fallbacks
        elif track:
            p.track = track
    if p.credits and not p.remixer:
        p.remixer = p.credits[0]

    # --- SoundCloud: the account is the credit ---------------------------
    if not p.kind:
        p.kind = _bare_kind(text)
    if p.credited_to_uploader and p.uploader:
        p.remixer = p.uploader

    p.artist = p.artist.strip(" -,&")
    p.track = p.track.strip(" -,&")
    if not p.track and p.artist:                        # ``Artist -`` and nothing else
        p.track, p.artist = p.artist, ""
        p.ambiguous = True
    return p


def _bare_kind(text: str) -> str:
    m = _KEYWORD.search(text)
    kind = m.group(1).lower() if m else ""
    return "" if kind == "mix" and re.search(r"\boriginal\s+mix\b", text, re.I) else kind


def _split_bare_credit(text: str) -> tuple[str, str, list[str]]:
    """``The Sweet Escape BOSEP Remix`` -> ``("BOSEP", "The Sweet Escape", …)``.

    Returns the guessed remixer, the track, and shorter prefixes of the track to
    search for as well -- because one name token is only the most common case,
    not the certain one, and ``E85 JLOOD Kosuk Extended Remix`` really is two.
    """
    tokens = text.split()
    idx = [i for i, t in enumerate(tokens)
           if _KEYWORD.fullmatch(t.strip(".,!?'\"")) is not None]
    if not idx:
        return "", text, []
    # the *first* keyword: a title can carry two credits stacked, and the one
    # nearest the track is the remix, the rest are somebody's edit of it
    cut = idx[0]
    end = cut
    while end > 0 and tokens[end - 1].lower().strip(".,'\"") in QUALIFIERS:
        end -= 1
    if end <= 1:                                        # nothing left for a track
        return "", " ".join(tokens[:end]) or text, []
    start = _lowercase_run(tokens, end - 1)
    if start <= 0:
        start = end - 1
    # names joined by & or x keep their partner: "JLOOD & Kosuk Remix"
    while start > 1 and tokens[start - 1].lower() in ("&", "x", "and", "+"):
        start = _lowercase_run(tokens, start - 2)
    remixer = " ".join(tokens[start:end])
    track_tokens = tokens[:start]
    # one shorter prefix, for the case where the guess ate a word of the title:
    # "E85 JLOOD Kosuk Extended Remix" is E85, not "E85 JLOOD"
    fallbacks = ([" ".join(track_tokens[:-1])] if len(track_tokens) > 1 else [])
    return remixer, " ".join(track_tokens), fallbacks


# ---------------------------------------------------------------------------
# what to search for
# ---------------------------------------------------------------------------

def search_queries(p: Parsed, limit: int = 5) -> list[str]:
    """Queries to hand yt-dlp's search, best first.

    ``<artist> <track> official audio`` first, because a record's official
    upload is what we want and that phrase is on nearly all of them; then the
    bare ``<artist> - <track>``; then the lyric-video phrasing, which is often
    the only full upload of an older record. When the split was a guess the
    reversed order and the shorter track prefixes follow, cheaply, in case the
    title was ``Track - Artist`` or the credit ate a word.
    """
    artist, track = p.artist.strip(), p.track.strip()
    out: list[str] = []

    def add(q: str) -> None:
        q = _SPACE.sub(" ", q).strip()
        if q and q.lower() not in {x.lower() for x in out}:
            out.append(q)

    if artist and track:
        add(f"{artist} {track} official audio")
        add(f"{artist} - {track}")
        add(f"{track} {artist} lyrics")
    elif track:
        add(f"{track} official audio")
        add(f"{track} original song")
        add(f"{track} lyrics")
    for short in p.track_fallbacks:                     # the guess may have eaten a word
        add(f"{artist} {short} official audio" if artist else f"{short} official audio")
    if p.ambiguous and artist and track:
        add(f"{track} {artist} official audio")         # …or the title is reversed
    return out[:limit]
