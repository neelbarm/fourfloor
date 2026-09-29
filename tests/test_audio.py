"""Audio I/O regressions."""

from __future__ import annotations

import numpy as np

from fourfloor.audio import MP3_CEILING_DB, decode, write_mp3, write_mp3_under

SR = 44100


def loud_master(seconds: float = 8.0) -> np.ndarray:
    """A brick-walled master at a -1.0 dBFS *sample* peak: hard-clipped
    square-ish waves, the shape that comes back over full scale from LAME."""
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for hz in (55.0, 110.0, 440.0, 3520.0):
        x += np.sign(np.sin(2 * np.pi * hz * t)) * 0.4
    x = np.clip(x, -0.9, 0.9)
    x = x / np.max(np.abs(x)) * 10 ** (-1.0 / 20)
    return np.stack([x, x[::-1]], axis=1).astype(np.float32)


def test_a_master_at_minus_one_decodes_under_the_ceiling(tmp_path) -> None:
    """Five of six set tracks decoded above 0 dBFS while their sessions said
    -1.0: the master trims the sample peak and the encoder overshoots it."""
    x = loud_master()
    raw = decode(write_mp3(tmp_path / "raw.mp3", x, SR)).samples
    raw_db = 20 * np.log10(np.max(np.abs(raw)))
    assert raw_db > MP3_CEILING_DB + 0.5, f"the stimulus only reached {raw_db:.2f} dBFS"

    path, used, peak = write_mp3_under(tmp_path / "out.mp3", x, SR)
    got = decode(path).samples
    got_db = 20 * np.log10(np.max(np.abs(got)))
    assert got_db <= MP3_CEILING_DB + 1e-3, f"decoded peak {got_db:.2f} dBFS"
    assert abs(got_db - peak) < 1e-3, "the reported peak is the file's"
    assert got_db > MP3_CEILING_DB - 1.0, "turned down no further than needed"
    # the buffer handed back is what was encoded
    assert np.max(np.abs(used)) < np.max(np.abs(x))
    assert not list(tmp_path.glob(".*encoding*")), "no temporary left behind"


def test_a_quiet_file_is_left_alone(tmp_path) -> None:
    x = loud_master() * 0.3
    _path, used, peak = write_mp3_under(tmp_path / "q.mp3", x, SR)
    np.testing.assert_array_equal(used, x)
    assert peak < MP3_CEILING_DB
