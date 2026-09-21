"""An automatic listener for fourfloor renders.

Nobody working on this project can hear the output, and tuning to
band-level numbers produced a remix that was rated garbage. So the critic
does not score a render against a spec; it scores it against records that
are known to be good, plus four detectors aimed at the specific things
that went wrong.

Six sub-scores, each 0-100:

``similarity``  how close the render's drops sit to real house drops in a
                learned embedding space (LAION-CLAP when installed).
``groove``      is there one sharp pulse, and do the onsets land on it.
``clarity``     onset density and 2-8 Hz modulation depth -- the signature
                of two source spans playing over each other.
``clicks``      sample-level discontinuities, weighted at section cues.
``vocal``       syllabic-rate energy in the vocal band and its level.
``loudness``    RMS, crest and clipping.

``--ear gemini`` adds a second opinion from a model that can actually hear
the audio. It never replaces the local numbers; it sits beside them.

The real code is in :mod:`fourfloor.critic.score`. This module only
forwards to it, lazily, so that registering the ``critic`` subcommand does
not pull scipy into every ``fourfloor --help``.
"""

from __future__ import annotations

__all__ = ["critique", "measure", "Critique", "SubScore", "Measured",
           "WEIGHTS", "BANDS", "GROOVE_GATE", "session_for", "verdict_for"]


def __getattr__(name: str):
    if name in __all__:
        from . import score

        return getattr(score, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
