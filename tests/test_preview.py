"""preview.html: the session file's free text is data, never markup."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from fourfloor import preview

EVIL = "</script><img src=x onerror=alert(1)>"


@pytest.fixture
def page(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(preview, "waveform", lambda path, points=0: [0.1, 0.5, 0.2])
    session = {
        "bpm": 124.0, "key": "F minor", "camelot": "4A", "duration": 60.0,
        "bars": 31, "semitone_shift": 0, "source": {"file": "song.mp3"},
        "loudness": {"peak_db": -1.0, "rms_db": -9.0},
        "sections": [{"kind": "drop", "start": 0.0, "end": 60.0, "bars": 31,
                      "source_label": EVIL, "source_start": 0.0,
                      "note": "a note that says __WAVE_JSON__ and " + EVIL}],
        "cues": [{"name": EVIL, "time": 0.0, "kind": "drop", "bar": 0}],
        "feedback": [{"time": 1.0, "note": EVIL + " & __SESSION__"}],
    }
    audio = tmp_path / "remix.mp3"
    audio.write_bytes(b"")
    out = preview.write_preview(audio, session)
    return out.read_text(encoding="utf8"), session


def test_a_note_cannot_close_the_script_block(page) -> None:
    html, _ = page
    script = html.split("<script>", 1)[1]
    assert script.count("</script>") == 1                 # only the real one
    assert "<img" not in html


def test_the_page_reads_back_exactly_the_session_it_was_given(page) -> None:
    html, session = page
    line = re.search(r"const SESSION = (.*);\n", html).group(1)
    assert json.loads(line) == session
    wave = re.search(r"const WAVE = (.*);\n", html).group(1)
    assert json.loads(wave) == [0.1, 0.5, 0.2]


def test_a_placeholder_inside_the_data_is_not_filled_in(page) -> None:
    html, _ = page
    assert "__WAVE_JSON__" in html and "__SESSION__" in html   # left as text
    assert html.count("[0.1, 0.5, 0.2]") == 1


def test_session_text_goes_through_the_page_escaper() -> None:
    js = preview._HTML
    for field in ("s.note", "s.source_label", "c.name"):
        assert re.search(r"\$\{" + re.escape(field) + r"\}", js) is None, field
        assert "esc(" + field + ")" in js
