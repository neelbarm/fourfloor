"""The same song with the same options is the same render, every time.

The dress rehearsal found two renders of one command audibly different:
demucs shifted its input by a random offset before separating, and every
decision downstream (the vocal and bass calls, the chop's slice starts) reads
those stems. A render a listener signs off on has to be the file the gig plays.
"""

from __future__ import annotations

import numpy as np
import pytest

from fourfloor.remix import RemixOptions, remix
from fourfloor.stems import demucs_available


def test_two_renders_of_the_fixture_are_byte_identical(fixture_path, tmp_path) -> None:
    opts = RemixOptions(target_bpm=124.0, length="1:30", stems="hpss", kit="none")
    a = remix(fixture_path, tmp_path / "a" / "fixture.house.mp3", opts)
    b = remix(fixture_path, tmp_path / "b" / "fixture.house.mp3", opts)
    for key in ("mp3", "wav", "plan"):
        assert a.paths[key].read_bytes() == b.paths[key].read_bytes(), key
    assert a.session == b.session


@pytest.mark.skipif(not demucs_available(), reason="demucs is not installed")
def test_demucs_separates_the_same_audio_the_same_way_twice(fixture_path) -> None:
    """``--shifts 0`` takes the random offset out; the model is otherwise a
    pure function of its input on the CPU."""
    from fourfloor.audio import decode
    from fourfloor.stems import separate_demucs

    clip = decode(fixture_path)
    x = clip.samples[: 8 * clip.sr]
    one = separate_demucs(x, clip.sr, jobs=1)
    two = separate_demucs(x, clip.sr, jobs=1)
    for name in ("vocals", "other", "percussive", "bass"):
        assert np.array_equal(getattr(one, name), getattr(two, name)), name
    assert float(np.abs(one.vocals).max()) > 0.0
