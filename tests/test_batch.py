"""Batch logic without the engine: key flow, resume, and failure isolation.

The renderer is replaced by a stub here, so these run in milliseconds and test
the decisions rather than the audio. ``test_batch_end_to_end.py`` drives the
real pipeline once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fourfloor import batch
from fourfloor.batch import BatchError

from test_export import make_session


# ---------------------------------------------------------------------------
# key flow
# ---------------------------------------------------------------------------

def shifts(codes, max_shift: int = 2) -> list[int]:
    return [s["shift"] for s in batch.plan_key_flow(codes, max_shift)]


def targets(codes, max_shift: int = 2) -> list[str]:
    return [s["camelot"] for s in batch.plan_key_flow(codes, max_shift)]


def test_shift_camelot_walks_the_wheel_and_keeps_the_mode() -> None:
    # +7 semitones is a perfect fifth, which is +1 on the Camelot wheel
    assert batch.shift_camelot("8A", 7) == "9A"
    assert batch.shift_camelot("8B", 7) == "9B"
    assert batch.shift_camelot("8A", 0) == "8A"
    assert batch.shift_camelot("12A", 7) == "1A"      # wraps round the wheel


def test_the_first_track_sets_the_key_and_is_never_shifted() -> None:
    assert shifts(["3A", "9B", "6A"])[0] == 0
    assert targets(["3A"])[0] == "3A"


def test_a_track_that_already_mixes_is_left_alone() -> None:
    """Same code, one step either way, or the relative major: all compatible."""
    assert shifts(["8A", "8A"]) == [0, 0]
    assert shifts(["8A", "9A"]) == [0, 0]
    assert shifts(["8A", "7A"]) == [0, 0]
    assert shifts(["8A", "8B"]) == [0, 0]


def test_an_incompatible_track_takes_the_smallest_shift_that_works() -> None:
    """2A against 8A is across the wheel; +2 semitones lands on 4A... which is

    still not adjacent, so the planner must find the one that is.
    """
    plan = batch.plan_key_flow(["8A", "2A"])
    assert plan[1]["compatible"] is True
    assert abs(plan[1]["shift"]) <= 2
    from fourfloor.analysis.key import camelot_neighbours
    assert plan[1]["camelot"] in camelot_neighbours("8A")


def test_shifts_prefer_the_gentler_move() -> None:
    """1A is two wheel steps from 8A... 3A is one step from 4A, so from 4A a
    single semitone should be enough and two should never be chosen over one."""
    for a, b in (("8A", "1A"), ("5B", "11B"), ("12A", "6A")):
        plan = batch.plan_key_flow([a, b])
        if plan[1]["compatible"]:
            smaller = [s for s in range(-1, 2)
                       if batch.shift_camelot(b, s) in _wheel(a)]
            if smaller:
                assert abs(plan[1]["shift"]) <= 1


def _wheel(code: str) -> set[str]:
    from fourfloor.analysis.key import camelot_neighbours
    return set(camelot_neighbours(code))


def test_a_key_nothing_can_reach_is_flagged_not_forced() -> None:
    """A two-semitone shift that still clashes is the worst of both worlds."""
    plan = batch.plan_key_flow(["8A", "2A"], max_shift=0)
    assert plan[1]["shift"] == 0
    assert plan[1]["compatible"] is False


def test_every_consecutive_pair_mixes_across_a_whole_set() -> None:
    codes = ["8A", "2A", "11B", "5A", "9B", "1A", "6A", "12B"]
    plan = batch.plan_key_flow(codes)
    for prev, nxt in zip(plan, plan[1:]):
        if nxt["compatible"]:
            assert nxt["camelot"] in _wheel(prev["camelot"])
    assert all(abs(s["shift"]) <= 2 for s in plan)


def test_the_chain_follows_the_shifted_key_not_the_original() -> None:
    """Track three has to mix with what track two *became*."""
    plan = batch.plan_key_flow(["8A", "2A", "2A"])
    assert plan[2]["camelot"] in _wheel(plan[1]["camelot"])


def test_an_unreadable_key_does_not_derail_the_set() -> None:
    plan = batch.plan_key_flow(["8A", "", "9A"])
    assert plan[1]["compatible"] is False and plan[1]["shift"] == 0
    assert plan[2]["shift"] == 0                     # still planned against 8A


def test_plan_key_flow_returns_a_readable_key_name() -> None:
    assert batch.plan_key_flow(["8A"])[0]["key"] == "Am"


def test_a_seeded_chain_plans_the_first_track_too() -> None:
    """What a resumed batch needs: mix out of the track already on disk."""
    plan = batch.plan_key_flow(["2A"], previous="8A")
    assert plan[0]["shift"] != 0
    assert plan[0]["camelot"] in _wheel("8A")
    assert batch.plan_key_flow(["9A"], previous="8A")[0]["shift"] == 0


# ---------------------------------------------------------------------------
# a stub renderer
# ---------------------------------------------------------------------------

class FakeResult:
    def __init__(self, paths, sess, metrics):
        self.paths, self.session, self.metrics = paths, sess, metrics


def stub_remix(*, fail_on=(), calls=None, key_seen=None, opts_seen=None):
    """A ``remix`` replacement that writes the two files a batch looks for."""

    def _remix(path, out, opts=None, style=None, progress=None):
        path, out = Path(path), Path(out)
        if calls is not None:
            calls.append(path.name)
        if key_seen is not None:
            key_seen[path.name] = getattr(opts, "key", None)
        if opts_seen is not None:
            opts_seen[path.name] = opts
        if any(token in path.name for token in fail_on):
            raise RuntimeError(f"engine refused {path.name}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 512)
        sess = make_session(bpm=getattr(opts, "target_bpm", 124.0) or 124.0,
                            file=out.name)
        from fourfloor.export import session_path_for
        session_path_for(out).write_text(json.dumps(sess), encoding="utf8")
        return FakeResult({"mp3": out}, sess, {"alignment_score": 0.93})

    return _remix


@pytest.fixture
def originals(tmp_path) -> Path:
    """Three source files, named the way a real folder is."""
    folder = tmp_path / "originals"
    folder.mkdir()
    for name in ("01 first.mp3", "02 Nuit — Blanche.mp3", "03 third.mp3"):
        (folder / name).write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 4096)
    return folder


@pytest.fixture
def engine(monkeypatch):
    """Install a stub renderer and hand back the knobs to configure it."""
    state = {"calls": [], "keys": {}, "opts": {}}

    def install(fail_on=()):
        monkeypatch.setattr("fourfloor.remix.remix",
                            stub_remix(fail_on=fail_on, calls=state["calls"],
                                       key_seen=state["keys"],
                                       opts_seen=state["opts"]))
        return state

    return install


# ---------------------------------------------------------------------------
# refusals before any work
# ---------------------------------------------------------------------------

def test_a_missing_source_folder_is_a_plain_error(tmp_path) -> None:
    with pytest.raises(NotADirectoryError):
        batch.run(tmp_path / "nope", tmp_path / "out", bpm=124.0)


def test_an_empty_source_folder_says_so(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(BatchError, match="no audio files"):
        batch.run(tmp_path / "empty", tmp_path / "out", bpm=124.0)


def test_rendering_into_the_source_folder_is_refused(originals) -> None:
    """Otherwise the next run would remix its own output."""
    with pytest.raises(BatchError, match="other than the source folder"):
        batch.run(originals, originals, bpm=124.0)


def test_an_unknown_key_strategy_is_refused(originals, tmp_path) -> None:
    with pytest.raises(BatchError, match="lock.*auto"):
        batch.run(originals, tmp_path / "out", bpm=124.0, key_strategy="vibes")


# ---------------------------------------------------------------------------
# a clean run
# ---------------------------------------------------------------------------

def test_a_clean_run_remixes_everything_and_writes_the_set(originals, tmp_path,
                                                           engine) -> None:
    engine()
    out = tmp_path / "gig"
    manifest = batch.run(originals, out, bpm=126.0, set_name="Friday")
    assert manifest["summary"] == {"total": 3, "ok": 3, "skipped": 0, "failed": 0,
                                   "set_duration": manifest["summary"]["set_duration"],
                                   "elapsed": manifest["summary"]["elapsed"]}
    assert [t["status"] for t in manifest["tracks"]] == ["ok"] * 3
    assert (out / "set.json").is_file()
    assert (out / "rekordbox.xml").is_file()
    assert (out / "cues.csv").is_file()
    assert manifest["set"] == "Friday"


def test_the_set_is_listed_in_source_order(originals, tmp_path, engine) -> None:
    engine()
    manifest = batch.run(originals, tmp_path / "gig", bpm=126.0)
    assert [Path(t["source"]).name for t in manifest["tracks"]] == [
        "01 first.mp3", "02 Nuit — Blanche.mp3", "03 third.mp3"]


def test_every_track_is_rendered_at_the_one_tempo(originals, tmp_path, engine) -> None:
    engine()
    manifest = batch.run(originals, tmp_path / "gig", bpm=126.0)
    assert {t["bpm"] for t in manifest["tracks"]} == {126.0}


def test_one_kit_and_one_bass_policy_reach_every_track(originals, tmp_path,
                                                       engine) -> None:
    """A set wants one drum kit and one low end, not a decision per file."""
    state = engine()
    manifest = batch.run(originals, tmp_path / "gig", bpm=126.0,
                         kit="murph", bass="sub", stems="hpss", form="tool")
    assert len(state["opts"]) == 3
    for opts in state["opts"].values():
        assert opts.kit == "murph"
        assert opts.bass == "sub"
        assert opts.form == "tool"
    assert manifest["kit"] == "murph" and manifest["bass"] == "sub"


def test_the_bass_policy_defaults_to_auto(originals, tmp_path, engine) -> None:
    state = engine()
    batch.run(originals, tmp_path / "gig", bpm=126.0)
    assert {o.bass for o in state["opts"].values()} == {"auto"}


def test_set_json_carries_the_cues_and_whatever_alignment_the_engine_reports(
        originals, tmp_path, engine) -> None:
    engine()
    manifest = batch.run(originals, tmp_path / "gig", bpm=126.0)
    track = manifest["tracks"][0]
    assert [c["name"] for c in track["cues"]][:3] == ["intro", "build 1", "drop 1"]
    assert track["alignment"] == {"alignment_score": 0.93}
    assert track["output"].endswith("01 first.house.mp3")
    assert track["source_bpm"] == 80.0


def test_set_json_is_json(originals, tmp_path, engine) -> None:
    engine()
    out = tmp_path / "gig"
    batch.run(originals, out, bpm=126.0)
    assert json.loads((out / "set.json").read_text(encoding="utf8"))["schema"] == 1


def test_the_set_name_defaults_to_the_source_folder(originals, tmp_path,
                                                    engine) -> None:
    engine()
    assert batch.run(originals, tmp_path / "gig", bpm=126.0)["set"] == "originals"


def test_key_lock_asks_the_engine_for_no_key_change(originals, tmp_path,
                                                    engine) -> None:
    state = engine()
    batch.run(originals, tmp_path / "gig", bpm=126.0, key_strategy="lock")
    assert set(state["keys"].values()) == {None}
    assert batch.run(originals, tmp_path / "gig2", bpm=126.0)["key_flow"] == []


# ---------------------------------------------------------------------------
# --key auto, with the analyser stubbed too
# ---------------------------------------------------------------------------

def stub_analyze(camelots: dict):
    """Report a fixed key per filename, the way ``analyze`` would."""
    from fourfloor.analysis.key import KeyEstimate, camelot_to_key

    class FakeGrid:
        bpm = 120.0

    class FakeAnalysis:
        def __init__(self, code):
            pc, minor = camelot_to_key(code)
            self.key = KeyEstimate(pc, minor, 0.9)
            self.grid = FakeGrid()

    def _analyze(path, keep_audio=True, clip=None):
        code = camelots.get(Path(path).name)
        if code is None:
            raise RuntimeError(f"cannot analyse {Path(path).name}")
        return FakeAnalysis(code)

    return _analyze


def test_key_auto_asks_the_engine_for_the_planned_key(originals, tmp_path, engine,
                                                      monkeypatch) -> None:
    state = engine()
    monkeypatch.setattr("fourfloor.analysis.analyze", stub_analyze({
        "01 first.mp3": "8A", "02 Nuit — Blanche.mp3": "2A", "03 third.mp3": "9A"}))
    manifest = batch.run(originals, tmp_path / "gig", bpm=126.0, key_strategy="auto")

    flow = {Path(t["source"]).name: s
            for t, s in zip(manifest["tracks"], manifest["key_flow"])}
    assert flow["01 first.mp3"]["shift"] == 0
    assert state["keys"]["01 first.mp3"] is None      # the first track is untouched
    for name, step in flow.items():
        if step["shift"]:
            assert state["keys"][name] == step["key"]


def test_key_auto_records_the_flow_in_set_json(originals, tmp_path, engine,
                                               monkeypatch) -> None:
    engine()
    monkeypatch.setattr("fourfloor.analysis.analyze", stub_analyze({
        "01 first.mp3": "8A", "02 Nuit — Blanche.mp3": "2A", "03 third.mp3": "2A"}))
    out = tmp_path / "gig"
    batch.run(originals, out, bpm=126.0, key_strategy="auto")
    data = json.loads((out / "set.json").read_text(encoding="utf8"))
    assert data["key_strategy"] == "auto"
    assert len(data["key_flow"]) == 3
    for prev, nxt in zip(data["key_flow"], data["key_flow"][1:]):
        if nxt["compatible"]:
            assert nxt["camelot"] in _wheel(prev["camelot"])


def test_resuming_with_key_auto_mixes_out_of_what_is_already_there(
        originals, tmp_path, engine, monkeypatch) -> None:
    keys = {"01 first.mp3": "8A", "02 Nuit — Blanche.mp3": "2A",
            "03 third.mp3": "2A"}
    state = engine(fail_on=("02", "03"))
    monkeypatch.setattr("fourfloor.analysis.analyze", stub_analyze(keys))
    out = tmp_path / "gig"
    batch.run(originals, out, bpm=126.0, key_strategy="auto")   # only track one lands

    state["calls"].clear()
    engine()
    manifest = batch.run(originals, out, bpm=126.0, key_strategy="auto", resume=True)
    assert manifest["summary"]["skipped"] == 1
    # the two new tracks are planned against track one's key, not from scratch
    assert manifest["key_flow"][0]["shift"] != 0
    assert manifest["key_flow"][0]["camelot"] in _wheel(manifest["tracks"][0]["camelot"])


def test_a_track_the_analyser_cannot_read_still_gets_remixed(originals, tmp_path,
                                                             engine,
                                                             monkeypatch) -> None:
    """A key we cannot find is a reason to leave the track alone, not to drop it."""
    state = engine()
    monkeypatch.setattr("fourfloor.analysis.analyze", stub_analyze({
        "01 first.mp3": "8A", "03 third.mp3": "9A"}))
    manifest = batch.run(originals, tmp_path / "gig", bpm=126.0, key_strategy="auto")
    assert manifest["summary"]["ok"] == 3
    assert len(state["calls"]) == 3
    unreadable = [s for s in manifest["key_flow"] if s["source"] == ""]
    assert unreadable and unreadable[0]["shift"] == 0


# ---------------------------------------------------------------------------
# failure isolation
# ---------------------------------------------------------------------------

def test_one_bad_track_does_not_abort_the_batch(originals, tmp_path, engine) -> None:
    engine(fail_on=("02",))
    manifest = batch.run(originals, tmp_path / "gig", bpm=126.0)
    statuses = {Path(t["source"]).name: t["status"] for t in manifest["tracks"]}
    assert statuses["01 first.mp3"] == "ok"
    assert statuses["03 third.mp3"] == "ok"
    assert statuses["02 Nuit — Blanche.mp3"] == "failed"
    assert manifest["summary"]["ok"] == 2 and manifest["summary"]["failed"] == 1


def test_a_failure_records_what_went_wrong(originals, tmp_path, engine) -> None:
    engine(fail_on=("02",))
    manifest = batch.run(originals, tmp_path / "gig", bpm=126.0)
    bad = [t for t in manifest["tracks"] if t["status"] == "failed"][0]
    assert "RuntimeError" in bad["error"] and "engine refused" in bad["error"]
    assert bad["output"] is None


def test_the_survivors_are_still_exported(originals, tmp_path, engine) -> None:
    engine(fail_on=("02",))
    out = tmp_path / "gig"
    batch.run(originals, out, bpm=126.0)
    import xml.etree.ElementTree as ET
    root = ET.fromstring((out / "rekordbox.xml").read_text(encoding="utf8"))
    assert root.find("COLLECTION").get("Entries") == "2"


def test_a_batch_where_everything_fails_still_writes_set_json(originals, tmp_path,
                                                              engine) -> None:
    engine(fail_on=("0",))
    out = tmp_path / "gig"
    manifest = batch.run(originals, out, bpm=126.0)
    assert manifest["summary"]["failed"] == 3
    assert (out / "set.json").is_file()
    assert not (out / "rekordbox.xml").exists()
    assert any("nothing rendered" in n for n in manifest["notes"])


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

def test_without_resume_everything_is_rendered_again(originals, tmp_path,
                                                     engine) -> None:
    state = engine()
    out = tmp_path / "gig"
    batch.run(originals, out, bpm=126.0)
    state["calls"].clear()
    batch.run(originals, out, bpm=126.0)
    assert len(state["calls"]) == 3


def test_resume_skips_what_is_already_on_disk(originals, tmp_path, engine) -> None:
    state = engine()
    out = tmp_path / "gig"
    batch.run(originals, out, bpm=126.0)
    state["calls"].clear()
    manifest = batch.run(originals, out, bpm=126.0, resume=True)
    assert state["calls"] == []
    assert manifest["summary"] == {"total": 3, "ok": 0, "skipped": 3, "failed": 0,
                                   "set_duration": manifest["summary"]["set_duration"],
                                   "elapsed": manifest["summary"]["elapsed"]}


def test_resume_finishes_an_interrupted_run(originals, tmp_path, engine) -> None:
    """The realistic case: the first two rendered, then ctrl-C."""
    state = engine(fail_on=("03",))
    out = tmp_path / "gig"
    batch.run(originals, out, bpm=126.0)
    state["calls"].clear()
    engine()                                          # the third track now works
    manifest = batch.run(originals, out, bpm=126.0, resume=True)
    assert state["calls"] == ["03 third.mp3"]
    assert manifest["summary"]["skipped"] == 2 and manifest["summary"]["ok"] == 1


def test_resume_re_renders_a_track_whose_session_file_is_missing(originals, tmp_path,
                                                                 engine) -> None:
    """An mp3 on its own is a half-written render, not a finished track."""
    state = engine()
    out = tmp_path / "gig"
    batch.run(originals, out, bpm=126.0)
    (out / "01 first.house.session.json").unlink()
    state["calls"].clear()
    batch.run(originals, out, bpm=126.0, resume=True)
    assert state["calls"] == ["01 first.mp3"]


def test_resume_re_renders_a_track_whose_mp3_is_empty(originals, tmp_path,
                                                      engine) -> None:
    state = engine()
    out = tmp_path / "gig"
    batch.run(originals, out, bpm=126.0)
    (out / "03 third.house.mp3").write_bytes(b"")
    state["calls"].clear()
    batch.run(originals, out, bpm=126.0, resume=True)
    assert state["calls"] == ["03 third.mp3"]


def test_resume_re_renders_a_track_whose_session_file_is_corrupt(originals, tmp_path,
                                                                 engine) -> None:
    state = engine()
    out = tmp_path / "gig"
    batch.run(originals, out, bpm=126.0)
    (out / "02 Nuit — Blanche.house.session.json").write_text("{ not json",
                                                              encoding="utf8")
    state["calls"].clear()
    batch.run(originals, out, bpm=126.0, resume=True)
    assert state["calls"] == ["02 Nuit — Blanche.mp3"]


def test_a_skipped_track_still_appears_in_the_export(originals, tmp_path,
                                                     engine) -> None:
    engine()
    out = tmp_path / "gig"
    batch.run(originals, out, bpm=126.0)
    batch.run(originals, out, bpm=126.0, resume=True)
    import xml.etree.ElementTree as ET
    root = ET.fromstring((out / "rekordbox.xml").read_text(encoding="utf8"))
    assert root.find("COLLECTION").get("Entries") == "3"


def test_already_done_needs_both_files(originals, tmp_path) -> None:
    out = tmp_path / "gig"
    src = originals / "01 first.mp3"
    assert batch.already_done(src, out) is False
    dest = batch.output_for(src, out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"\x00")
    assert batch.already_done(src, out) is False
    from fourfloor.export import session_path_for
    session_path_for(dest).write_text("{}", encoding="utf8")
    assert batch.already_done(src, out) is True


def test_output_naming_keeps_the_original_name(tmp_path) -> None:
    got = batch.output_for(Path("/x/Nuit — Blanche #2.m4a"), tmp_path)
    assert got.name == "Nuit — Blanche #2.house.mp3"
    assert got.parent == tmp_path


# ---------------------------------------------------------------------------
# the terminal report
# ---------------------------------------------------------------------------

def test_the_report_names_every_track_and_counts_the_outcome(originals, tmp_path,
                                                             engine) -> None:
    from fourfloor import ui

    engine(fail_on=("02",))
    manifest = batch.run(originals, tmp_path / "gig", bpm=126.0)
    text = batch.Reporter(ui.C(False), quiet=True).report(manifest)
    assert "01 first.house.mp3" in text
    assert "failed" in text and "engine refused" in text
    assert "2 remixed" in text and "1 failed" in text
    assert "rekordbox.xml" in text and "set.json" in text


def test_the_reporter_prints_nothing_when_quiet(originals, tmp_path, engine,
                                                capsys) -> None:
    from fourfloor import ui

    engine()
    batch.run(originals, tmp_path / "gig", bpm=126.0,
              on_event=batch.Reporter(ui.C(False), quiet=True))
    assert capsys.readouterr().out == ""


def test_the_reporter_draws_a_line_per_track(originals, tmp_path, engine,
                                             capsys) -> None:
    from fourfloor import ui

    engine()
    batch.run(originals, tmp_path / "gig", bpm=126.0, set_name="Friday",
              on_event=batch.Reporter(ui.C(False), quiet=False))
    out = capsys.readouterr().out
    assert "batch: Friday" in out
    assert "1/3" in out and "3/3" in out
    assert "126.00 BPM" in out
