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

The measurement decides. A voice that already sits on the straight grid is
played straight through, which is what a listener liked about the CAN'T SAY
render; a voice that sits on a triplet grid is chopped.
"""

from __future__ import annotations

#: How much of a voice has to sit on the straight sixteenth lattice before it
#: can simply be played. Measured on two Don Toliver records warped to 128:
#: CAN'T SAY, which a listener called amazing played straight, and *Body*,
#: which he called "terrible with overlaps".
STRAIGHT_LOCK = 0.72

#: ...and how much better the triplet lattice has to fit before the voice is
#: treated as a triplet flow rather than as a loose straight one.
TRIPLET_EDGE = 0.06

#: Below this there is not enough voice in the stem to be worth deciding about.
MIN_DUTY = 0.05


def choose_vocal(vocals, sr: int, bpm: float) -> tuple[str, dict, str]:
    """``("flow" | "chop", measurement, why)`` for a separated vocal stem."""
    from ..analysis.alignment import vocal_fit

    m = vocal_fit(vocals, sr, bpm)
    if m["duty"] < MIN_DUTY:
        return "flow", m, "there is barely a vocal in this to arrange"
    if m["straight"] >= STRAIGHT_LOCK:
        return "flow", m, (
            f"{m['straight']:.0%} of the vocal already lands on the grid, so it "
            "is played as it was sung")
    if m["advantage"] >= TRIPLET_EDGE or m["straight"] < 0.60:
        return "chop", m, (
            f"the vocal fits a triplet grid better than a straight one "
            f"({m['triplet']:.0%} against {m['straight']:.0%}) -- a triplet flow "
            "will not lock to four-on-the-floor however it is warped, so it is "
            "chopped into slices that each start on a syllable and land on a beat")
    return "flow", m, (
        f"{m['straight']:.0%} of the vocal lands on the grid and no other grid "
        "fits it better, so it is played as it was sung")
