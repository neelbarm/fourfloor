"""Are these two files the same song?

This is the part that has to be right. Everything upstream is a guess -- a
title parsed by regex, a search result ranked by word overlap -- and if the
guess is wrong the pipeline files somebody else's record as the original of a
remix and then *learns* from the difference between two unrelated songs.

A remixer's edit shares almost nothing measurable with the record it came from.
The drums are different, the arrangement is different, the tempo has moved by
10-20%, the key may have moved, and the whole bottom of the spectrum has been
replaced. What survives is the voice: the same singer singing the same melody
to the same words. So the comparison is made on separated vocals, and on the
part of them that is invariant to everything a remixer does except pitch --
chroma, the twelve pitch classes.

How the comparison is run:

* **Excerpts, not whole files.** Demucs on a whole track is minutes of CPU for
  nothing: 60-90 seconds of the remix's most vocal-rich stretch against two or
  three minutes of the original is enough to be sure, and it is what keeps a
  hundred pasted links from taking all night. The excerpts are chosen by
  vocal-band energy, and the stems are deleted the moment the features are out
  of them.
* **Tempo is normalised, not assumed.** Both files' BPM is detected, and the
  remix's time base is scaled by the ratio -- and by twice and half that ratio,
  because beat trackers are octave-ambiguous and remixers genuinely do read a
  70 BPM record as 140.
* **Pitch is searched.** The chroma is rotated through -3..+3 semitones. A
  remixer who pitched the vocal up two semitones to fit his key has not made a
  different song, and the rotation that wins is reported as the shift.
* **Subsequence DTW.** The remix excerpt is a *piece* of the original, sitting
  somewhere in it, possibly stretched a little unevenly. Subsequence DTW finds
  the best-matching stretch of the reference with a free start and a free end,
  which is exactly that question. The score is the mean cosine similarity along
  the winning path, normalised by path length so that a path which warps
  aggressively cannot win by visiting fewer cells.

Two thresholds, calibrated on known truths (see :data:`ACCEPT` and
:data:`REVIEW`): above the first the pair is filed, between them it waits for a
person, below the second it is rejected and the next candidate is tried.

A remix with no vocal in it falls back to the same machinery run on the whole
mix's chroma -- the chords are still the chords -- and is marked low
confidence, because an instrumental match is a weaker claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..analysis import features as F
from ..audio import decode

#: Feature frame rate for the comparison. Ten frames a second resolves a
#: syllable and keeps a three-minute reference to under two thousand columns.
FEATURE_FPS = 10.0

#: How much of each file is looked at. The remix excerpt is the query, the
#: original excerpt is the haystack, so the haystack is longer.
REMIX_SECONDS = 75.0
ORIGINAL_SECONDS = 180.0

#: Pitch shifts tried, in semitones.
SEMITONES = tuple(range(-3, 4))

#: Tempo relations tried on top of the detected BPM ratio: straight, half and
#: double, because a beat tracker cannot tell 70 from 140 and neither can a
#: remixer who decided to read the record the other way.
RELATIONS = (1.0, 2.0, 0.5)

#: Fine tempo refinement around the winning relation.
REFINE = (0.94, 0.97, 1.0, 1.03, 1.06)

#: How long a piece of the remix is aligned at a time, and what share of the
#: pieces the score is the mean of. See :func:`score_against`.
SEGMENT_SECONDS = 20.0
SEGMENT_KEEP = 0.6

#: Below this duty cycle there is not enough voice in a stem to compare, and
#: the comparison falls back to the whole mix's chroma.
MIN_VOCAL_DUTY = 0.06

#: Log compression applied to chroma before the features are centred. Standard
#: for chroma similarity: it stops one loud bass note from being the whole
#: column.
GAMMA = 10.0

#: What a half- or double-time reading has to beat the straight one by. It is
#: not a free choice: stretching the query to twice its length gives the
#: warping twice as much room to find something, and on the unrelated pairs
#: this was calibrated against, the best wrong answer was always a half-time
#: one.
RELATION_COST = 0.05

# --- the thresholds ---------------------------------------------------------
#: Calibrated on five originals against six house remixes -- thirty
#: comparisons, five of them true pairs and twenty-five known-false ones, run
#: on real records. The two populations came out like this:
#:
#: =========  ==================  =================
#: ..         score               margin
#: =========  ==================  =================
#: 4 true     0.728 - 0.777       0.322 - 0.403
#: 25 false   0.346 - 0.459       0.019 - 0.106
#: =========  ==================  =================
#:
#: …plus one true pair that lands inside the false population and is rejected:
#: a bootleg whose vocal is cut into 0.29-second pieces. Nothing in the gap is
#: a coincidence, and nothing in these thresholds is fitted tighter than the gap
#: allows -- ``ACCEPT`` sits a tenth below the lowest true score and a tenth
#: above the highest false one.
#:
#: Two numbers, not one. ``ACCEPT`` is the score -- mean cosine similarity of
#: vocal chroma along the winning path -- and ``MARGIN`` is how far that winner
#: stands above the median of every tempo-and-pitch trial made on the same two
#: files. The margin is what catches the near miss: *every* alignment of two
#: unrelated songs is equally mediocre, so the winner barely clears its own
#: field, while a true pair's winner stands a third of a point clear of it.
ACCEPT = 0.52
REVIEW = 0.46
MARGIN = 0.15
REVIEW_MARGIN = 0.12
#: An instrumental (chroma-only) match is a weaker claim -- whole-mix chroma
#: correlates on genre alone -- so it has to clear more to be believed. These
#: four are scaled from the vocal ones rather than calibrated: there was no
#: instrumental pair in the reference folder to calibrate them against.
ACCEPT_CHROMA = 0.58
REVIEW_CHROMA = 0.52
MARGIN_CHROMA = 0.18
REVIEW_MARGIN_CHROMA = 0.14


class VerifyError(RuntimeError):
    """The comparison could not be made at all."""


# ---------------------------------------------------------------------------
# excerpts
# ---------------------------------------------------------------------------

def vocal_band(mono: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame energy in the band a voice lives in, over the band it does not.

    A ratio rather than a level, because the loudest part of a house remix is
    the drop and the drop is mostly kick. 300 Hz - 4 kHz against 30-200 Hz says
    "something is singing here" on both a dense mix and a sparse one.
    """
    mid = F.band_energy(mono, sr, 300.0, 4000.0)
    low = F.band_energy(mono, sr, 30.0, 200.0)
    n = min(len(mid), len(low))
    if n == 0:
        return np.zeros(0)
    return mid[:n] / (low[:n] + float(np.percentile(low[:n], 60)) + 1e-9)


def pick_window(mono: np.ndarray, sr: int, seconds: float) -> float:
    """Start time of the ``seconds``-long window with the most voice in it.

    The first eighth of a track is skipped where there is room, because a DJ
    edit opens with drums and an intro is the one stretch guaranteed to have no
    singing in it.
    """
    duration = len(mono) / float(sr)
    if duration <= seconds + 1.0:
        return 0.0
    score = vocal_band(mono, sr)
    if not len(score):
        return max(0.0, (duration - seconds) / 2.0)
    fps = F.frame_rate(sr)
    win = max(1, int(seconds * fps))
    kernel = np.ones(win) / win
    smooth = np.convolve(score, kernel, mode="valid")
    skip = int(min(duration * 0.125, max(0.0, duration - seconds - 1.0)) * fps)
    smooth = smooth[skip:]
    if not len(smooth):
        return 0.0
    start = (skip + int(np.argmax(smooth))) / fps
    return float(min(max(0.0, start), max(0.0, duration - seconds)))


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def _resample_cols(x: np.ndarray, n_out: int) -> np.ndarray:
    """Linear-resample a (d, n) feature sequence along time."""
    n_in = x.shape[1]
    if n_in == 0 or n_out <= 1:
        return np.zeros((x.shape[0], max(n_out, 1)))
    src = np.linspace(0.0, n_in - 1.0, n_out)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    w = (src - lo)[None, :]
    return x[:, lo] * (1.0 - w) + x[:, hi] * w


def chroma_sequence(mono: np.ndarray, sr: int, fps: float = FEATURE_FPS) -> np.ndarray:
    """Centred, L2-normalised chroma at ``fps`` frames a second.

    Log compression first, then the mean over the whole excerpt is subtracted
    from every column. Centring is what makes the score mean something: raw
    chroma columns are non-negative, so two unrelated songs correlate at 0.7
    simply by both being music. Centred, unrelated material sits near zero and a
    real match sits between 0.4 and 0.9.
    """
    if len(mono) < F.CHROMA_FFT:
        return np.zeros((12, 0))
    c = F.chromagram(mono, sr)
    if c.shape[1] < 2:
        return np.zeros((12, 0))
    c = np.log1p(GAMMA * c)
    n_out = max(2, int(round(c.shape[1] / F.frame_rate(sr) * fps)))
    c = _resample_cols(c, n_out)
    c = c - c.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(c, axis=0, keepdims=True)
    return c / np.maximum(norm, 1e-9)


def novelty(seq: np.ndarray) -> np.ndarray:
    """Chroma flux: how much the harmony is changing, frame by frame."""
    if seq.shape[1] < 2:
        return np.zeros(seq.shape[1])
    flux = np.maximum(0.0, np.diff(seq, axis=1)).sum(axis=0)
    flux = np.concatenate([[0.0], flux])
    if flux.max() > 0:
        flux = flux / flux.max()
    return flux - flux.mean()


def phrases(mono: np.ndarray, sr: int, floor: float = 0.12
            ) -> tuple[float, list[float]]:
    """Duty cycle of a vocal stem, and the length of every phrase in it.

    A phrase is a run of frames where the voice is sounding, with gaps shorter
    than 150 ms bridged -- a singer breathing between words has not stopped
    singing. This is the measurement that separates a remixer who played the
    vocal as sung from one who cut it into pieces.
    """
    rms = F.rms_envelope(mono, hop=512, win=2048)
    if not len(rms):
        return 0.0, []
    peak = max(float(rms.max()), 1e-9)
    on = rms > floor * peak
    fps = sr / 512.0
    bridge = int(round(0.15 * fps))
    out: list[float] = []
    i, n = 0, len(on)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        gap = 0
        while j < n:
            if on[j]:
                gap = 0
            else:
                gap += 1
                if gap > bridge:
                    break
            j += 1
        end = j - gap
        out.append(max(0.0, (end - i) / fps))
        i = j
    return float(np.mean(on)), [p for p in out if p >= 0.12]


@dataclass
class Fingerprint:
    """Everything the comparison needs from one file."""

    path: str = ""
    kind: str = "remix"
    source: str = "vocals"            # "vocals" or "mix"
    bpm: float = 0.0
    downbeat: float = 0.0
    start: float = 0.0
    seconds: float = 0.0
    duty: float = 0.0
    phrase_median: float = 0.0
    phrase_count: int = 0
    lattice: dict = field(default_factory=dict)
    chroma: np.ndarray = field(default_factory=lambda: np.zeros((12, 0)), repr=False)

    def to_meta(self) -> dict:
        d = asdict(self)
        d.pop("chroma", None)
        return d


def _cache_key(path: Path, kind: str, source: str, seconds: float) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{kind}|{source}|{seconds}|3"
    return hashlib.sha1(raw.encode("utf8")).hexdigest()[:20]


def fingerprint(path: str | Path, kind: str = "remix", demucs: bool = True,
                cache_dir: str | Path | None = None, seconds: float | None = None,
                progress=None) -> Fingerprint:
    """Measure one file: tempo, an excerpt, its vocals, and their chroma.

    ``kind`` is ``"remix"`` (a short excerpt, chosen for voice) or
    ``"original"`` (a long one, so the remix's excerpt has somewhere to be
    found). Demucs stems are held in memory and never written to the reference
    folder; the features that come out of them are a few kilobytes and are
    cached, so accepting a pair later does not run the model again.
    """
    from ..analysis.tempo import analyze_beats

    step = progress or (lambda *_a, **_k: None)
    path = Path(path)
    want = float(seconds if seconds is not None
                 else (REMIX_SECONDS if kind == "remix" else ORIGINAL_SECONDS))
    source = "vocals" if demucs else "mix"
    cache = Path(cache_dir) / f"{_cache_key(path, kind, source, want)}.npz" if cache_dir else None
    if cache is not None and cache.is_file():
        try:
            data = np.load(cache, allow_pickle=False)
            meta = json.loads(str(data["meta"]))
            step("cached", path.name)
            return Fingerprint(chroma=data["chroma"], **meta)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            cache.unlink(missing_ok=True)

    step("decode", path.name)
    clip = decode(path)
    mono_all = clip.mono
    start = pick_window(mono_all, clip.sr, want)
    a = int(start * clip.sr)
    b = min(len(clip.samples), a + int(want * clip.sr))
    excerpt = clip.samples[a:b]
    if len(excerpt) < clip.sr:
        raise VerifyError(f"{path.name} is too short to compare")
    mono = excerpt.mean(axis=1)

    step("tempo", f"{path.name}")
    grid = analyze_beats(mono, clip.sr)
    bpm, downbeat = float(grid.bpm), float(grid.first_downbeat)

    voice = None
    if demucs:
        step("demucs", f"vocals from {path.name}")
        from ..stems import separate_demucs
        stems = separate_demucs(excerpt, clip.sr, want_bass=False)
        voice = stems.vocals
        if voice is None:
            voice = stems.harmonic
    fp = Fingerprint(path=str(path), kind=kind, bpm=bpm, downbeat=downbeat,
                     start=float(start), seconds=(b - a) / clip.sr, source=source)
    if voice is not None:
        v_mono = voice.mean(axis=1) if voice.ndim == 2 else voice
        duty, ph = phrases(v_mono, clip.sr)
        fp.duty, fp.phrase_count = duty, len(ph)
        fp.phrase_median = float(np.median(ph)) if ph else 0.0
        if duty >= MIN_VOCAL_DUTY:
            from ..analysis.alignment import vocal_fit
            # the excerpt's own downbeat, not zero: the straight and triplet
            # lattices are only comparable when both are in phase with the bar,
            # and a triplet grid is the denser of the two, so a phase-less
            # reading flatters it by simply having more places to land
            fp.lattice = {k: round(float(v), 4)
                          for k, v in vocal_fit(voice, clip.sr, bpm,
                                                first_downbeat=downbeat).items()}
            fp.source = "vocals"
            fp.chroma = chroma_sequence(v_mono, clip.sr)
        else:
            fp.source = "mix"                       # nothing sang; use the mix
    if fp.source == "mix":
        fp.chroma = chroma_sequence(mono, clip.sr)
    del clip, excerpt, voice                        # the audio goes now, not later

    if cache is not None and fp.chroma.size:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, chroma=fp.chroma.astype(np.float32),
                            meta=json.dumps(fp.to_meta()))
    return fp


# ---------------------------------------------------------------------------
# subsequence DTW
# ---------------------------------------------------------------------------

def subsequence_dtw(cost: np.ndarray) -> tuple[float, int, int]:
    """Best mean cost of matching every row of ``cost`` inside its columns.

    Free start and free end along the reference axis, with the three-step
    pattern (1,1), (1,2), (2,1) -- diagonal, and one step of warping either way.
    That pattern has no dependency inside a row, so the whole thing is a loop of
    numpy operations rather than a loop of Python arithmetic, which is the
    difference between 15 ms and half a second per comparison.

    The number of cells on the path is accumulated alongside the cost, so the
    mean is a true mean: a path that warps hard visits fewer cells and does not
    get to win by accumulating less.

    Returns ``(mean_cost, end_column, cells)``.
    """
    m, n = cost.shape
    if m < 3 or n < 3:
        return 1.0, 0, 0
    big = np.float64(1e9)
    D = np.full((m, n), big)
    L = np.zeros((m, n))
    D[0] = cost[0]
    L[0] = 1.0
    # row 1: only row 0 can be a predecessor, via (1,1) and (1,2)
    prev = np.full(n, big)
    prevl = np.zeros(n)
    prev[1:] = D[0, :-1]
    prevl[1:] = L[0, :-1]
    alt = np.full(n, big)
    altl = np.zeros(n)
    alt[2:] = D[0, :-2]
    altl[2:] = L[0, :-2]
    take = alt < prev
    D[1] = np.where(take, alt, prev) + cost[1]
    L[1] = np.where(take, altl, prevl) + 1.0
    for i in range(2, m):
        d1 = np.full(n, big); l1 = np.zeros(n)
        d1[1:] = D[i - 1, :-1]; l1[1:] = L[i - 1, :-1]
        d2 = np.full(n, big); l2 = np.zeros(n)
        d2[2:] = D[i - 1, :-2]; l2[2:] = L[i - 1, :-2]
        d3 = np.full(n, big); l3 = np.zeros(n)
        d3[1:] = D[i - 2, :-1]; l3[1:] = L[i - 2, :-1]
        stack = np.stack([d1, d2, d3])
        pick = np.argmin(stack, axis=0)
        D[i] = np.take_along_axis(stack, pick[None], 0)[0] + cost[i]
        L[i] = np.take_along_axis(np.stack([l1, l2, l3]), pick[None], 0)[0] + 1.0
    end = D[m - 1] / np.maximum(L[m - 1], 1.0)
    j = int(np.argmin(np.where(L[m - 1] > 0, end, big)))
    return float(end[j]), j, int(L[m - 1, j])


def _cost(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Cosine distance in 0..1 between every pair of columns."""
    return (1.0 - query.T @ ref) * 0.5


def score_against(query: np.ndarray, ref: np.ndarray) -> tuple[float, int, int]:
    """How well a remix excerpt is explained by an original, in segments.

    Subsequence DTW assumes the query runs through the reference in order, and
    a remixer chopping a vocal breaks exactly that assumption: he takes the
    second line, repeats it, then the first, then a word from the bridge. One
    long alignment of a chopped vocal reads as a near miss -- which is what
    *Stay Fly* did, scoring 0.33 against the record it is unmistakably made of.

    So the query is cut into pieces of about :data:`SEGMENT_SECONDS`, each
    aligned into the reference on its own, and the score is the mean of the best
    :data:`SEGMENT_KEEP` of them. Order between pieces stops mattering, a piece
    that is pure drums or a new topline stops dragging the rest down, and the
    cost is unchanged: ``k`` alignments of ``m/k`` frames is the same arithmetic
    as one of ``m``.

    Returns ``(score, end column of the best piece, cells walked)``.
    """
    m = query.shape[1]
    seg = max(8, int(round(SEGMENT_SECONDS * FEATURE_FPS)))
    k = max(1, int(round(m / seg)))
    pieces = np.array_split(np.arange(m), k) if k > 1 else [np.arange(m)]
    scores: list[float] = []
    ends: list[tuple[float, int]] = []
    cells = 0
    for idx in pieces:
        if len(idx) < 8:
            continue
        mean_cost, end, walked = subsequence_dtw(_cost(query[:, idx], ref))
        score = 1.0 - 2.0 * mean_cost
        scores.append(score)
        ends.append((score, end))
        cells += walked
    if not scores:
        return -1.0, 0, 0
    keep = max(1, int(np.ceil(len(scores) * SEGMENT_KEEP)))
    best = sorted(scores, reverse=True)[:keep]
    return float(np.mean(best)), int(max(ends)[1]), cells


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------

@dataclass
class Match:
    """The verdict on one candidate original."""

    score: float = 0.0
    verdict: str = "reject"           # match | needs_review | reject
    method: str = "vocal"             # vocal | chroma
    confidence: str = "high"
    tempo_ratio: float = 1.0          # original seconds per remix second
    beat_relation: float = 1.0        # straight, half or double
    stretch_refine: float = 1.0       # the few percent the alignment preferred
    semitones: int = 0
    bpm_original: float = 0.0
    bpm_remix: float = 0.0
    novelty_corr: float = 0.0
    cells: int = 0
    runner_up: float = 0.0
    null: float = 0.0                 # the median of every trial: the field
    margin: float = 0.0               # how far the winner stands above it
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("score", "tempo_ratio", "stretch_refine", "novelty_corr",
                  "runner_up", "null", "margin"):
            d[k] = round(float(d[k]), 4)
        return d


def _align(query: np.ndarray, ref: np.ndarray, scale: float,
           semitones=SEMITONES) -> tuple[float, int, int, int, list[float]]:
    """Best ``(score, semitones, end column, cells, all scores)`` at one scale.

    The semitone number is signed the way a musician would say it: ``+2`` means
    the remix sits two semitones *above* the original. Internally that is the
    query being rolled *down* two to line up, hence the negation.

    Every trial's score comes back as well, because the *field* matters as much
    as the winner: see :func:`compare`.
    """
    n_out = int(round(query.shape[1] * scale))
    if n_out < 8 or n_out > ref.shape[1] * 1.5:
        return -1.0, 0, 0, 0, []
    q = _resample_cols(query, n_out)
    q = q / np.maximum(np.linalg.norm(q, axis=0, keepdims=True), 1e-9)
    best = (-1.0, 0, 0, 0)
    field_: list[float] = []
    for k in semitones:
        rolled = np.roll(q, k, axis=0)
        score, end, cells = score_against(rolled, ref)
        field_.append(float(score))
        if score > best[0]:
            best = (float(score), -int(k), int(end), int(cells))
    return (*best, field_)


def compare(original: Fingerprint, remix: Fingerprint) -> Match:
    """Score a remix excerpt against an original, searching tempo and pitch."""
    if original.chroma.shape[1] < 8 or remix.chroma.shape[1] < 8:
        raise VerifyError("one of these files gave no usable features")
    method = "vocal" if (original.source == "vocals" and remix.source == "vocals") \
        else "chroma"
    base = (remix.bpm / original.bpm) if original.bpm > 0 else 1.0

    scored: list[tuple[float, int, float, int, int]] = []
    field_: list[float] = []
    for relation in RELATIONS:
        score, k, end, cells, trials = _align(remix.chroma, original.chroma,
                                              base * relation)
        field_.extend(trials)
        if score > -1.0:
            # A half- or double-time reading is a real thing remixers do, and
            # it is also where a *wrong* pair scores best: stretching the query
            # to twice its length gives the warping twice as much room. So it
            # has to win by a margin rather than by a hair.
            adjusted = score - (0.0 if abs(relation - 1.0) < 1e-9 else RELATION_COST)
            scored.append((adjusted, k, relation, end, cells))
    if not scored:
        raise VerifyError("no tempo relation between these two files was usable")
    scored.sort(key=lambda t: -t[0])
    top, runner = scored[0], (scored[1][0] if len(scored) > 1 else 0.0)

    # refine the winning relation a few percent either way: a remixer's stretch
    # is rarely exactly the ratio of two detected tempi. The ratio *reported* is
    # still the one the two detected tempi imply -- that is the number style
    # learning wants -- and the refinement only decides the score.
    best, fine_used = top, 1.0
    for fine in REFINE:
        if abs(fine - 1.0) < 1e-9:
            continue
        score, k, end, cells, _ = _align(remix.chroma, original.chroma,
                                         base * top[2] * fine,
                                         semitones=(-top[1],))
        if score > best[0]:
            best, fine_used = (score, k, top[2], end, cells), fine

    score, semis, relation, _end, cells = best
    null = float(np.median(field_)) if field_ else 0.0
    match = Match(
        score=float(max(0.0, score)), method=method,
        tempo_ratio=float(base * relation), beat_relation=float(relation),
        stretch_refine=float(fine_used),
        semitones=int(semis), bpm_original=float(original.bpm),
        bpm_remix=float(remix.bpm), cells=int(cells),
        runner_up=float(max(0.0, runner)), null=null,
        margin=float(max(0.0, score) - null),
        novelty_corr=_novelty_corr(original.chroma, remix.chroma,
                                   base * relation * fine_used),
    )
    accept = ACCEPT if method == "vocal" else ACCEPT_CHROMA
    review = REVIEW if method == "vocal" else REVIEW_CHROMA
    margin = MARGIN if method == "vocal" else MARGIN_CHROMA
    soft = REVIEW_MARGIN if method == "vocal" else REVIEW_MARGIN_CHROMA
    if match.score >= accept and match.margin >= margin:
        match.verdict = "match"
    elif match.score >= review and match.margin >= soft:
        match.verdict = "needs_review"
    else:
        match.verdict = "reject"
    match.confidence = "high" if method == "vocal" else "low"
    if method == "chroma":
        match.note = ("no separated vocal to compare, so this is a harmonic "
                      "match on the whole mix -- weaker evidence")
    return match


def _novelty_corr(ref: np.ndarray, query: np.ndarray, scale: float) -> float:
    """Peak normalised cross-correlation of the two chroma novelty curves.

    A second opinion, and a cheap one: where the harmony changes is a rhythm of
    its own, and two takes of the same song change chords in the same places.
    It is reported rather than thresholded -- on a remix that replaced the
    chords it is meaningless, and the DTW score already knows that.
    """
    a = novelty(ref)
    q = novelty(query)
    n_out = int(round(len(q) * scale))
    if len(a) < 8 or n_out < 8:
        return 0.0
    b = _resample_cols(q[None, :], n_out)[0]
    if float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
        return 0.0
    corr = np.correlate(a / (np.std(a) * len(a)), b / np.std(b), mode="valid")
    return float(np.max(corr)) if len(corr) else 0.0


def verify(original: str | Path, remix: str | Path, demucs: bool = True,
           cache_dir: str | Path | None = None, progress=None
           ) -> tuple[Match, Fingerprint, Fingerprint]:
    """Fingerprint both files and compare them."""
    o = fingerprint(original, kind="original", demucs=demucs,
                    cache_dir=cache_dir, progress=progress)
    r = fingerprint(remix, kind="remix", demucs=demucs,
                    cache_dir=cache_dir, progress=progress)
    return compare(o, r), o, r
