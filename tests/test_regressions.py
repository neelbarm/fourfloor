"""Regressions for bugs found in the pre-launch review.

Each test here pins down a specific failure that shipped once: a flag that was
validated only after a full render, an arrangement that quietly ignored the
length it was given, a stem buffer the renderer scribbled on, and a preview
page whose audio element pointed at nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fourfloor.arrange import FORMS, form_min_bars, parse_length, plan, scale_form
from fourfloor.house.engine import _loop_to
from fourfloor.preview import _audio_src
from fourfloor.remix import RemixOptions, remix
from fourfloor.stems import separate
from fourfloor.style import Style


# ---------------------------------------------------------------------------
# the renderer must not write into the stems it was handed
# ---------------------------------------------------------------------------

def test_loop_to_returns_a_buffer_the_caller_can_fade_in_place() -> None:
    """`render_source` fades every slot's edges in place.

    `_loop_to` used to hand back a view whenever the requested span already had
    the right length, so the fade wrote straight back into the shared harmonic
    stem. The drop slots have no filter sweep and no chop to force a copy, so
    both drops read the same span and the second one got a doubly-faded edge.
    """
    sr = 8000
    src = np.ones((sr * 8, 2), dtype=np.float32)
    before = src.copy()

    seg = _loop_to(src, 0, sr * 2, sr * 2, sr)
    seg[:64] *= np.linspace(0.0, 1.0, 64)[:, None]

    assert np.array_equal(src, before), "the source stem was modified in place"


def test_loop_to_copies_even_when_it_has_to_loop() -> None:
    sr = 8000
    src = np.ones((sr * 2, 2), dtype=np.float32)
    before = src.copy()
    seg = _loop_to(src, 0, sr * 5, sr, sr)   # wants more than the source holds
    seg *= 0.0
    assert np.array_equal(src, before)


# ---------------------------------------------------------------------------
# --length must be honoured, or the caller must be told it could not be
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form", list(FORMS))
def test_form_minimum_is_reachable_and_exact(form: str) -> None:
    """`scale_form` has to land on its target, not merely near it.

    `room` was floored at ``8 * n_scalable`` while ``fixed`` was added on top,
    so any target below ``fixed + 8 * n_scalable`` was unreachable and the
    trim loop gave up, returning a longer form without a word.
    """
    lo = form_min_bars(FORMS[form])
    for target in (lo, lo + 8, lo + 64, lo + 200):
        out = scale_form(FORMS[form], target)
        assert sum(b for _, b in out) == target, f"{form} could not reach {target} bars"


@pytest.mark.parametrize("form", list(FORMS))
def test_plan_never_undershoots_the_form_minimum(fixture_analysis, form: str) -> None:
    a = fixture_analysis
    p = plan(a, 124.0, 2.0, form_name=form, length=30.0)
    assert p.total_bars == form_min_bars(FORMS[form])
    assert p.total_bars % 8 == 0


def test_a_reachable_length_lands_within_one_phrase(fixture_analysis) -> None:
    a = fixture_analysis
    for seconds in (180.0, 240.0, 300.0, 420.0):
        p = plan(a, 124.0, 2.0, form_name="club", length=seconds)
        assert abs(p.duration - seconds) <= 8 * p.bar_dur


def test_parse_length_rejects_nonsense() -> None:
    assert parse_length("4m30s") == 270.0
    for bad in ("garbage", "", "  ", "abc:def", "-5", "0"):
        with pytest.raises(ValueError):
            parse_length(bad)


# ---------------------------------------------------------------------------
# bad flags fail before the work, not after it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bpm", [0.0, -5.0, 59.0, 250.0])
def test_out_of_range_bpm_is_rejected_before_decoding(tmp_path, fixture_path, bpm) -> None:
    """--bpm 0 was silently ignored, -5 crashed inside numpy, and 250 rendered
    the whole track before the session schema rejected it -- leaving a partial
    mp3 on disk."""
    out = tmp_path / "x.mp3"
    with pytest.raises(ValueError, match="bpm"):
        remix(fixture_path, out, RemixOptions(target_bpm=bpm, length="2:00"))
    assert not list(tmp_path.iterdir()), "a rejected remix left files behind"


@pytest.mark.parametrize("swing", [-0.1, 0.9])
def test_out_of_range_swing_is_rejected(tmp_path, fixture_path, swing) -> None:
    with pytest.raises(ValueError, match="swing"):
        remix(fixture_path, tmp_path / "x.mp3", RemixOptions(swing=swing, length="2:00"))


@pytest.mark.parametrize("name", ["noext", "out.ogg", "out.flac"])
def test_output_extension_is_checked_up_front(tmp_path, fixture_path, name) -> None:
    """`-o` with no extension used to run the whole pipeline and then die on a
    raw ffmpeg "Invalid argument"."""
    with pytest.raises(ValueError, match="mp3 or .wav"):
        remix(fixture_path, tmp_path / name, RemixOptions(length="2:00"))
    assert not list(tmp_path.iterdir())


def test_style_load_reports_the_file_it_could_not_read(tmp_path) -> None:
    bad = tmp_path / "style.json"
    bad.write_text("not json at all")
    with pytest.raises(ValueError, match="style profile"):
        Style.load(bad)
    bad.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="style profile"):
        Style.load(bad)


def test_style_round_trips(tmp_path) -> None:
    p = Style(bpm=126.0, swing=0.12, length=300.0, n_refs=3).save(tmp_path / "s.json")
    back = Style.load(p)
    assert back.bpm == 126.0 and back.swing == 0.12 and back.length == 300.0


# ---------------------------------------------------------------------------
# separation happens on the warped buffer, not the original file
# ---------------------------------------------------------------------------

def test_separate_consumes_the_warped_buffer() -> None:
    """demucs used to be pointed at the untouched source file while the
    arrangement addressed stems in warped seconds, so every demucs remix was
    laid out against a bed at the wrong tempo."""
    sr = 8000
    warped = np.zeros((sr * 2, 2), dtype=np.float32)
    warped[::400] = 0.5
    st = separate(warped, sr, "hpss")
    assert len(st.harmonic) == len(warped)
    assert len(st.percussive) == len(warped)
    assert st.source_name == "hpss"


def test_demucs_says_how_to_install_itself_when_it_is_missing() -> None:
    from fourfloor.stems import demucs_available

    if demucs_available():
        pytest.skip("demucs is installed, so there is no message to check")
    with pytest.raises(RuntimeError, match=r"fourfloor\[stems\]"):
        separate(np.zeros((16, 2), dtype=np.float32), 8000, "demucs")


# ---------------------------------------------------------------------------
# the preview page has to point at the audio it was given
# ---------------------------------------------------------------------------

def test_preview_audio_src_is_relative_to_the_page(tmp_path) -> None:
    """`preview -o` can put the page anywhere; the src used to be a bare
    filename, which resolves to nothing from another directory."""
    audio = tmp_path / "mix" / "song.house.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"")

    assert _audio_src(audio, tmp_path / "mix" / "preview.html") == "song.house.mp3"
    assert _audio_src(audio, tmp_path / "preview.html") == "mix/song.house.mp3"
    assert _audio_src(audio, tmp_path / "a" / "b" / "preview.html").endswith(
        "mix/song.house.mp3")


def test_preview_audio_src_is_percent_encoded(tmp_path) -> None:
    """A `#` in a song title truncates the URL at the fragment and the player
    silently loads nothing."""
    audio = tmp_path / "my sóng's \"remix\" #1.house.mp3"
    audio.write_bytes(b"")
    src = _audio_src(audio, tmp_path / "preview.html")
    assert "#" not in src and " " not in src and '"' not in src
    assert "%23" in src and "%20" in src


# ---------------------------------------------------------------------------
# nothing non-finite may reach the outputs
# ---------------------------------------------------------------------------

def test_session_and_plan_json_are_strict_json(remix_of_a_quiet_clip) -> None:
    """json.dumps writes bare NaN/Infinity, which is valid JavaScript and
    invalid JSON -- a DJ host parsing it strictly would reject the file."""
    res = remix_of_a_quiet_clip
    for key in ("session", "plan"):
        raw = res.paths[key].read_text()
        assert "NaN" not in raw and "Infinity" not in raw
        json.loads(raw)          # a strict parser must accept it


def test_quiet_and_silent_sources_still_master_cleanly(remix_of_a_quiet_clip) -> None:
    res = remix_of_a_quiet_clip
    audio = res.audio
    assert np.isfinite(audio).all(), "non-finite samples reached the render"
    assert float(np.max(np.abs(audio))) < 1.0, "the master clipped"
    assert res.metrics["peak_db"] <= -0.9
