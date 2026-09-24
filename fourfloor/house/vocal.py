"""Whether to play a voice straight through, or cut it into pieces.

A rapper's flow is not a drum machine. Triplet flows -- three syllables to the
beat instead of two or four -- have been the default in hip-hop for a decade,
and a triplet syllable lands a third of a beat away from anything on a
sixteenth lattice: 156 ms at 128 BPM. The warp cannot fix it. The warp puts the
song's *beats* on the grid, and the voice is phrased against those beats
exactly as it was rapped.

Laid down as one continuous take over a straight house kit, that reads as two
things happening at once. The producer's answer is not to stretch the voice --
that is what makes a remix sound like a broken tape -- it is to stop playing it
continuously. Cut it into pieces that each begin on a syllable and each land on
a beat, repeat the good ones, and leave gaps. Every slice re-synchronises at
its own start, so nothing has time to drift.

The measurement decides, and it has to be a measurement that can say "I do
not know". The first version compared how much of the voice sat on a
sixteenth lattice with how much sat on a six-per-beat one, each within 30 ms.
The denser lattice wins that by luck -- a voice with no rhythm at all reads 51%
against 77% at 128 BPM -- so every real voice came out a triplet flow and was
chopped: CAN'T SAY, whose flowing vocal is the render a listener loved, and
sung pop along with it. Measured on the separated, warped stems, CAN'T SAY
(62/83) and *Body* (63/83) were indistinguishable on it.

What decides now is a comparison luck cannot win: of the syllables that land
on a triplet-only position or a sixteenth-only one -- two zones of the same
width -- the share on the triplet ones, phrase after phrase, with the voice's
own lag behind the beat taken out first. Luck gives one half at any tempo. A
voice is chopped only when that share is well clear of a half *and* it holds in
most four-bar phrases; everything else is played as it was sung, which is the
default the loved render had.

Calibration, on the demucs vocal stems (``--shifts 0``) warped to 128, read by
:func:`~fourfloor.analysis.alignment.vocal_fit` -- triplet share, share of
four-bar phrases where the triplet positions win, and what the old lattice fit
said (straight / six-per-beat):

======================  =====  =====  =========
CAN'T SAY               0.47   0.31   0.62/0.83
Body                    0.55   0.64   0.63/0.83
You Belong With Me      0.51   0.44   0.59/0.78
Never Be Like You       0.49   0.42   0.43/0.77
The Sweet Escape        0.39   0.30   0.51/0.85
E85                     0.44   0.32   0.64/0.80
Cold Shoulder           0.47   0.40   0.56/0.77
rhythmless voice        0.43-0.60  0.29-0.78  (36 synthetic runs, 100-150 BPM)
======================  =====  =====  =========

*Body* leans to the triplet grid more than any of the others -- but no further
than a voice with syllables placed at random does by luck, so a threshold that
chopped it would chop about one rhythmless voice in five. The thresholds sit
above what luck reaches: a voice is chopped only when its syllables are clearly
on the triplet grid, phrase after phrase, and everything else is played as it
was sung, the default the loved CAN'T SAY render had. ``--vocal chop`` chops a
voice on request (*Body* included) and ``--vocal flow`` never does.
"""

from __future__ import annotations

#: Of the syllables on a triplet-only or a sixteenth-only position, the share
#: on triplet ones before a voice counts as a triplet flow. Chance is 0.5 at
#: any tempo, and a rhythmless voice reached 0.60 by luck in 36 runs.
TRIPLET_SHARE = 0.62

#: ...and the share of four-bar phrases that have to agree (luck reached 0.78,
#: but never together with a share this high).
TRIPLET_WINDOWS = 0.65

#: Below this there is not enough voice in the stem to be worth deciding about.
MIN_DUTY = 0.05


def choose_vocal(vocals, sr: int, bpm: float) -> tuple[str, dict, str]:
    """``("flow" | "chop", measurement, why)`` for a separated vocal stem.

    Thresholds learned from reference pairs (``straight_lock`` in the style
    file) are no longer read here: they were fitted on the raw sixteenth-lattice
    fit of each *original* at its own tempo, a number dominated by how much of
    the timeline the lattice covers at that tempo, so it was not on the scale of
    anything the engine measures and three chance-level pairs could flip the
    decision.
    """
    from ..analysis.alignment import vocal_fit

    m = vocal_fit(vocals, sr, bpm)
    if m["duty"] < MIN_DUTY:
        return "flow", m, "there is barely a vocal in this to arrange"
    share, windows = m["triplet_share"], m["triplet_windows"]
    if share >= TRIPLET_SHARE and windows >= TRIPLET_WINDOWS:
        return "chop", m, (
            f"the vocal is a triplet flow: {share:.0%} of its off-grid syllables "
            f"land on triplet positions, in {windows:.0%} of its phrases -- a "
            "triplet flow will not lock to four-on-the-floor however it is "
            "warped, so it is chopped into slices that each start on a syllable "
            "and land on a beat")
    return "flow", m, (
        f"the vocal is not a triplet flow ({share:.0%} of its off-grid syllables "
        "sit on triplet positions, where chance alone gives 50%), so it is "
        "played as it was sung")
