"""The command line refuses cheaply, protects the source, and says what it did.

Each test here is a failure somebody hit or could hit the night before a gig:
a remix written over the only copy of a record, a pinned kit replaced by a
rebuild under the same name, a typo found after an hour of Demucs, a set that
lost tracks and still exited 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fourfloor import cli, ui
from fourfloor import kit as K


def _stored(home: Path, name: str, source: str = "rec.mp3") -> None:
    from fourfloor.audio import write_wav

    folder = K.kits_home(home) / name
    folder.mkdir(parents=True, exist_ok=True)
    write_wav(folder / "loop.wav", np.zeros((4410, 2), dtype=np.float32))
    (folder / "meta.json").write_text(json.dumps({"name": name, "source": source}))


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "home"
    monkeypatch.setenv("FOURFLOOR_HOME", str(h))
    return h


@pytest.fixture
def no_audio_work(monkeypatch):
    """Fail the test if anything decodes, analyses or downloads."""
    def boom(*_a, **_k):
        raise AssertionError("audio work ran before the cheap checks")

    monkeypatch.setattr("fourfloor.remix.decode", boom)
    monkeypatch.setattr("fourfloor.remix.analyze", boom)
    monkeypatch.setattr("fourfloor.kit.analyze", boom)
    monkeypatch.setattr(cli, "_from_link", boom)
    monkeypatch.setattr("fourfloor.style.learn", boom)


# ---------------------------------------------------------------------------
# the source is never an output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("src_name, out_name, extra", [
    ("song.wav", "song.mp3", []),              # the WAV copy lands on the source
    ("song.mp3", "song.wav", ["--no-wav"]),    # a .wav output writes its .mp3 too
    ("song.mp3", "song.mp3", []),
    ("Song.mp3", "song.mp3", ["--no-wav"]),    # the Mac's disk ignores case
])
def test_remix_refuses_to_write_over_its_source(tmp_path, home, no_audio_work, capsys,
                                                src_name, out_name, extra) -> None:
    src = tmp_path / src_name
    src.write_bytes(b"the only copy")
    code = cli.main(["remix", str(src), "-o", str(tmp_path / out_name), *extra])
    assert code == 1
    assert "would replace the source" in capsys.readouterr().err
    assert src.read_bytes() == b"the only copy"


def test_an_output_beside_the_source_that_misses_it_is_allowed(tmp_path) -> None:
    from fourfloor.remix import RemixOptions, validate_options

    src = tmp_path / "song.wav"
    src.write_bytes(b"x")
    validate_options(RemixOptions(wav=False), tmp_path / "song.mp3", source=src)
    validate_options(RemixOptions(), tmp_path / "song.house.mp3", source=src)


# ---------------------------------------------------------------------------
# cheap checks before expensive work
# ---------------------------------------------------------------------------

def test_a_mistyped_kit_is_refused_before_the_analysis(fixture_path, tmp_path, home,
                                                       no_audio_work, capsys) -> None:
    _stored(home, "murph")
    code = cli.main(["remix", str(fixture_path), "-o", str(tmp_path / "x.mp3"),
                     "--kit", "murhp"])
    assert code == 1
    assert "no kit called 'murhp'" in capsys.readouterr().err


def test_a_link_is_not_downloaded_for_a_request_that_will_be_refused(
        tmp_path, home, no_audio_work, capsys) -> None:
    code = cli.main(["remix", "--url", "https://youtu.be/xyz", "--bpm", "250",
                     "-o", str(tmp_path / "x.mp3")])
    assert code == 1
    assert "out of range" in capsys.readouterr().err


def test_an_unwritable_output_is_refused_before_the_render(fixture_path, tmp_path, home,
                                                           no_audio_work, capsys) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        code = cli.main(["remix", str(fixture_path), "-o", str(locked / "x.mp3")])
        assert code == 1
        assert "not writable" in capsys.readouterr().err
        code = cli.main(["learn", str(tmp_path), "-o", str(locked / "style.json")])
        assert code == 1
        assert "not writable" in capsys.readouterr().err
    finally:
        locked.chmod(0o700)


def test_kit_build_checks_its_name_before_demucs(tmp_path, home, no_audio_work) -> None:
    with pytest.raises(ValueError, match="not a kit name"):
        K.build(tmp_path / "rec.mp3", name="../x")


# ---------------------------------------------------------------------------
# kits
# ---------------------------------------------------------------------------

def test_kit_build_will_not_replace_a_kit_without_force(tmp_path, home, no_audio_work,
                                                        capsys) -> None:
    """The pin stores a name: rebuilding 'murph' from another record would have
    changed the drums under the loved recipe with `kit list` unchanged."""
    _stored(home, "murph", source="the loved one.mp3")
    K.pin("murph")
    code = cli.main(["kit", "build", str(tmp_path / "other.mp3"), "--name", "murph"])
    assert code == 1
    assert "already a kit called 'murph'" in capsys.readouterr().err
    assert K.load("murph").source == "the loved one.mp3"
    # --force gets past the guard (and on to the analysis, stubbed to fail here)
    with pytest.raises(AssertionError, match="audio work"):
        K.build(tmp_path / "other.mp3", name="murph", force=True)


def test_kit_default_says_what_a_remix_will_use(home, capsys) -> None:
    _stored(home, "murph")
    _stored(home, "newer")
    assert cli.main(["kit", "default", "Murph"]) == 0
    out = capsys.readouterr().out
    assert "murph" in out and "Murph" not in out
    assert K.default_name() == "murph"


def test_a_refused_kit_pin_is_a_usage_error(home, capsys) -> None:
    assert cli.main(["kit", "default", "nope"]) == 2


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_calls(monkeypatch):
    seen: dict = {}

    def fake_run(folder, out, **kw):
        seen.update(kw)
        seen["folder"], seen["out"] = folder, out
        return {"summary": seen.pop("_summary", {"ok": 1, "skipped": 0, "failed": 0}),
                "tracks": [], "exports": {}, "notes": []}

    monkeypatch.setattr("fourfloor.batch.run", fake_run)
    return seen


def test_batch_takes_every_musical_option_remix_has(tmp_path, batch_calls) -> None:
    style = tmp_path / "style.json"
    style.write_text("{}")
    code = cli.main(["batch", str(tmp_path), "-o", str(tmp_path / "out"), "--bpm", "128",
                     "--vocal", "flow", "--drums-db", "-1.75", "--no-kick-reinforce",
                     "--style", str(style), "-q"])
    assert code == 0
    assert batch_calls["vocal"] == "flow"
    assert batch_calls["drums_db"] == -1.75
    assert batch_calls["kick_reinforce"] is False
    assert batch_calls["style"] == str(style)


def test_batch_defaults_match_remix(tmp_path, batch_calls) -> None:
    """The CAN'T SAY recipe through batch is the same as through remix."""
    cli.main(["batch", str(tmp_path), "-o", str(tmp_path / "out"), "--bpm", "128", "-q"])
    args = cli.parse_remix_args(["remix", "x.mp3"])
    assert batch_calls["vocal"] == args.vocal == "auto"
    assert batch_calls["drums_db"] == args.drums_db == 0.0
    assert batch_calls["kick_reinforce"] is True and not args.no_kick_reinforce
    assert batch_calls["stems"] == args.stems == cli.default_stems()
    assert batch_calls["bass"] == args.bass == "auto"


@pytest.mark.parametrize("summary, code", [
    ({"ok": 9, "skipped": 0, "failed": 0}, 0),
    ({"ok": 8, "skipped": 1, "failed": 1}, 3),       # a set with a hole in it
    ({"ok": 0, "skipped": 0, "failed": 3}, 1),
])
def test_batch_exit_code_says_whether_the_set_is_whole(tmp_path, batch_calls,
                                                       summary, code) -> None:
    batch_calls["_summary"] = summary
    assert cli.main(["batch", str(tmp_path), "-o", str(tmp_path / "o"),
                     "--bpm", "128", "-q"]) == code


# ---------------------------------------------------------------------------
# separation default
# ---------------------------------------------------------------------------

def test_the_default_separation_is_demucs_when_it_is_installed(monkeypatch) -> None:
    monkeypatch.delenv("FOURFLOOR_STEMS", raising=False)
    monkeypatch.setattr("fourfloor.stems.demucs_available", lambda: True)
    assert cli.default_stems() == "demucs"
    assert cli.parse_remix_args(["remix", "x.mp3"]).stems == "demucs"
    monkeypatch.setattr("fourfloor.stems.demucs_available", lambda: False)
    assert cli.default_stems() == "hpss"
    assert cli.parse_remix_args(["remix", "x.mp3"]).stems == "hpss"
    # still selectable either way
    assert cli.parse_remix_args(["remix", "x.mp3", "--stems", "hpss"]).stems == "hpss"
    monkeypatch.setenv("FOURFLOOR_STEMS", "hpss")
    monkeypatch.setattr("fourfloor.stems.demucs_available", lambda: True)
    assert cli.default_stems() == "hpss"


def test_help_says_which_separation_is_the_default(monkeypatch, capsys) -> None:
    monkeypatch.delenv("FOURFLOOR_STEMS", raising=False)
    monkeypatch.setattr("fourfloor.stems.demucs_available", lambda: True)
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["remix", "--help"])
    assert "default here: demucs" in " ".join(capsys.readouterr().out.split())


# ---------------------------------------------------------------------------
# small correctness
# ---------------------------------------------------------------------------

def test_preview_reads_the_session_of_the_file_it_was_given(tmp_path, capsys) -> None:
    """song.tool.mp3's session is song.tool.session.json, not song.session.json."""
    base = {"key": "F minor", "camelot": "4A", "duration": 60.0, "bars": 32,
            "semitone_shift": 0, "source": {"file": "song.mp3"}, "cues": [],
            "sections": [], "loudness": {"peak_db": -1.0, "rms_db": -9.0}}
    from fourfloor.audio import write_wav

    for name in ("song.wav", "song.tool.wav"):
        write_wav(tmp_path / name, np.zeros((44100, 2), dtype=np.float32))
    (tmp_path / "song.session.json").write_text(json.dumps({**base, "bpm": 100.0}))
    (tmp_path / "song.tool.session.json").write_text(json.dumps({**base, "bpm": 128.0}))
    assert cli.main(["preview", str(tmp_path / "song.wav")]) == 0
    assert cli.main(["preview", str(tmp_path / "song.tool.wav")]) == 0
    tool = (tmp_path / "song.tool.preview.html").read_text()
    club = (tmp_path / "song.preview.html").read_text()
    assert '"bpm": 128.0' in tool and '"bpm": 100.0' in club


def test_refs_learn_json_is_nothing_but_json(tmp_path, monkeypatch, capsys) -> None:
    from fourfloor.refs import pipeline

    monkeypatch.chdir(tmp_path)                      # no styles/ folder here
    monkeypatch.setattr(pipeline, "learn",
                        lambda opts, on=None, repo_out=None: {"style": {"bpm": 128.0}})
    assert cli.main(["refs", "learn", "--refs", str(tmp_path), "--json"]) == 0
    cap = capsys.readouterr()
    assert json.loads(cap.out) == {"bpm": 128.0}
    assert "no folder for" in cap.err


def test_serve_refuses_a_port_that_cannot_exist(capsys) -> None:
    assert cli.main(["serve", "--port", "99999"]) == 2
    assert "out of range" in capsys.readouterr().err


@pytest.mark.parametrize("env, want", [
    ({"FORCE_COLOR": "0"}, False),
    ({"FORCE_COLOR": "false"}, False),
    ({"FORCE_COLOR": "1"}, True),
    ({"NO_COLOR": "1", "FORCE_COLOR": "1"}, False),
    ({"NO_COLOR": "", "FORCE_COLOR": "1"}, True),     # empty NO_COLOR is unset
])
def test_colour_follows_the_conventions(monkeypatch, env, want) -> None:
    for k in ("FORCE_COLOR", "NO_COLOR"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    assert ui.supports_color() is want


def test_an_os_error_is_a_message_not_a_traceback(monkeypatch, capsys) -> None:
    def full(*_a, **_k):
        raise OSError(28, "No space left on device", "/Volumes/USB/x.mp3")

    monkeypatch.setattr(cli, "cmd_export", full)
    assert cli.main(["export", "x.mp3"]) == 1
    assert "No space left on device" in capsys.readouterr().err
