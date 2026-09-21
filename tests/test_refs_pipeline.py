"""The ingest run: filing, resuming, failing one link at a time, and learning.

Nothing here touches the network, Demucs or a kit build. The remix is the
bundled fixture copied in as if it had been downloaded, the search is a
function that returns a candidate, and the verification is a function that
returns a verdict -- which leaves exactly what these tests are about: what ends
up on disk, what the ledger says, and what happens when one link of many goes
wrong.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from fourfloor.refs import learned, pipeline, verify
from fourfloor.refs.ledger import Entry, Ledger
from fourfloor.refs.search import Candidate

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lofi-7.mp3"


# ---------------------------------------------------------------------------
# a run with everything expensive replaced
# ---------------------------------------------------------------------------

def a_match(score: float = 0.62, verdict: str = "match") -> verify.Match:
    return verify.Match(score=score, verdict=verdict, method="vocal",
                        tempo_ratio=0.89, semitones=0, bpm_original=146.0,
                        bpm_remix=130.0, margin=0.3, null=0.3)


def a_fingerprint(duty: float, phrase: float, straight: float = 0.8,
                  bpm: float = 130.0) -> verify.Fingerprint:
    return verify.Fingerprint(
        bpm=bpm, duty=duty, phrase_median=phrase, phrase_count=12,
        source="vocals",
        lattice={"straight": straight, "triplet": 0.4, "advantage": 0.4 - straight,
                 "duty": duty, "scatter_ms": 20.0, "onsets_per_bar": 8.0})


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    """Search, fetch, compare and kit build, all stood in for."""
    state = {"searches": [], "fetches": [], "compares": [], "kits": [],
             "candidates": [Candidate(url="https://www.youtube.com/watch?v=orig",
                                      title="Somebody - The Original",
                                      uploader="Somebody", duration=210.0,
                                      score=0.93, query="somebody the original")],
             "match": a_match()}

    def fake_search(parsed, **kwargs):
        state["searches"].append(parsed.track)
        return list(state["candidates"])

    def fake_fetch(url, dest, name=None, kind=None, **kwargs):
        from fourfloor.fetch import Fetched, free_path
        state["fetches"].append(url)
        stem = ".".join(x for x in (name or "track", kind) if x)
        target = free_path(Path(dest), stem, ".mp3")
        shutil.copyfile(FIXTURE, target)
        return Fetched(path=target, title="Somebody - The Original",
                       uploader="Somebody", duration=210.0, url=url,
                       site="youtube", bytes=target.stat().st_size)

    def fake_compare(original, remix, opts, note):
        state["compares"].append((Path(original).name, Path(remix).name))
        return (state["match"], a_fingerprint(0.7, 1.9), a_fingerprint(0.5, 0.8))

    def fake_kit(remix, slug, opts, note):
        state["kits"].append(slug)
        return slug

    monkeypatch.setattr(pipeline.search, "find_original", fake_search)
    monkeypatch.setattr(pipeline.fetch_mod, "fetch", fake_fetch)
    monkeypatch.setattr(pipeline, "_compare", fake_compare)
    monkeypatch.setattr(pipeline, "_build_kit", fake_kit)
    return state


@pytest.fixture
def opts(tmp_path) -> pipeline.Options:
    return pipeline.Options(home=tmp_path / "house-refs", pause=0.0)


def local(tmp_path, name: str) -> str:
    """The fixture, under a remix-shaped filename, as if it were on the desk."""
    path = tmp_path / f"{name}.mp3"
    shutil.copyfile(FIXTURE, path)
    return str(path)


# ---------------------------------------------------------------------------
# filing
# ---------------------------------------------------------------------------

def test_a_local_remix_becomes_a_filed_pair(tmp_path, opts, fake_run) -> None:
    source = local(tmp_path, "Somebody - The Original (Neel Remix) FREE DOWNLOAD")
    (entry,) = pipeline.add([source], opts)

    assert entry.status == "done"
    assert entry.slug == "somebody-the-original-neel"
    assert (opts.pairs / "somebody-the-original-neel.remix.mp3").is_file()
    assert (opts.pairs / "somebody-the-original-neel.original.mp3").is_file()
    assert entry.kit == entry.slug
    assert entry.score == pytest.approx(0.62)
    assert entry.tempo_ratio == pytest.approx(0.89)


def test_the_sidecar_says_where_both_halves_came_from(tmp_path, opts, fake_run) -> None:
    source = local(tmp_path, "Somebody - The Original (Neel Remix)")
    (entry,) = pipeline.add([source], opts)
    side = json.loads((opts.pairs / f"{entry.slug}.json").read_text(encoding="utf8"))

    assert side["original"]["url"] == "https://www.youtube.com/watch?v=orig"
    assert side["remix"]["remixer"] == "Neel"
    assert side["match"]["score"] == pytest.approx(0.62)
    assert side["vocal"]["original_lattice"] == "straight"
    assert side["vocal"]["treatment"] in ("continuous", "chopped")
    assert "excerpts" in side


def test_the_audio_stays_out_of_the_repository(tmp_path, opts, fake_run) -> None:
    """Everything written lives under the reference folder, nowhere else."""
    source = local(tmp_path, "Somebody - The Original (Neel Remix)")
    (entry,) = pipeline.add([source], opts)
    for path in (entry.remix_path, entry.original_path):
        assert Path(path).resolve().is_relative_to(opts.home.resolve())


# ---------------------------------------------------------------------------
# resuming
# ---------------------------------------------------------------------------

def test_running_it_again_skips_what_is_already_done(tmp_path, opts, fake_run) -> None:
    source = local(tmp_path, "Somebody - The Original (Neel Remix)")
    pipeline.add([source], opts)
    assert len(fake_run["searches"]) == 1

    (again,) = pipeline.add([source], opts)
    assert len(fake_run["searches"]) == 1          # nothing was done twice
    assert again.status == "done"


def test_force_does_it_again(tmp_path, opts, fake_run) -> None:
    source = local(tmp_path, "Somebody - The Original (Neel Remix)")
    pipeline.add([source], opts)
    opts.force = True
    pipeline.add([source], opts)
    assert len(fake_run["searches"]) == 2


def test_a_crash_halfway_leaves_a_ledger_to_carry_on_from(tmp_path, opts,
                                                          fake_run) -> None:
    """The ledger is written after the remix lands, before the search result."""
    one = local(tmp_path, "Somebody - The Original (Neel Remix)")
    two = local(tmp_path, "Another - Track (Someone Edit)")

    def explode(original, remix, opts_, note):
        raise KeyboardInterrupt

    pipeline.add([one], opts)
    import fourfloor.refs.pipeline as P
    P._compare, saved = explode, P._compare
    try:
        with pytest.raises(KeyboardInterrupt):
            pipeline.add([two], opts)
    finally:
        P._compare = saved

    led = Ledger(opts.ledger_path)
    assert led.by_url(one).status == "done"
    assert led.by_url(two).status == "pending"
    assert Path(led.by_url(two).remix_path).is_file()


# ---------------------------------------------------------------------------
# when it goes wrong
# ---------------------------------------------------------------------------

def test_one_bad_link_does_not_stop_the_others(tmp_path, opts, fake_run,
                                               monkeypatch) -> None:
    good_a = local(tmp_path, "A - One (X Remix)")
    good_b = local(tmp_path, "B - Two (Y Remix)")
    dead = "https://www.youtube.com/watch?v=nope"

    def probe(url, timeout=None):
        raise pipeline.fetch_mod.FetchError("that track is not available any more.")

    monkeypatch.setattr(pipeline.fetch_mod, "probe", probe)
    entries = pipeline.add([good_a, dead, good_b], opts)

    assert [e.status for e in entries] == ["done", "failed", "done"]
    assert "not available" in entries[1].error
    led = Ledger(opts.ledger_path)
    assert led.by_url(dead).status == "failed"


def test_a_remix_with_no_original_is_kept_on_its_own(tmp_path, opts, fake_run) -> None:
    fake_run["candidates"] = []
    source = local(tmp_path, "Somebody - The Original (Neel Remix)")
    (entry,) = pipeline.add([source], opts)

    assert entry.status == "standalone"
    assert Path(entry.remix_path).parent == opts.remixes
    assert not (opts.pairs / f"{entry.slug}.remix.mp3").exists()
    assert entry.kit == entry.slug                 # it still yields a kit


def test_a_score_between_the_thresholds_waits_for_a_person(tmp_path, opts,
                                                           fake_run) -> None:
    fake_run["match"] = a_match(score=0.34, verdict="needs_review")
    source = local(tmp_path, "Somebody - The Original (Neel Remix)")
    (entry,) = pipeline.add([source], opts)

    assert entry.status == "needs_review"
    assert entry.candidates[0]["url"] == "https://www.youtube.com/watch?v=orig"
    assert not (opts.pairs / f"{entry.slug}.original.mp3").exists()
    assert (opts.pairs / f"{entry.slug}.remix.mp3").is_file()


def test_accepting_a_reviewed_pair_files_it(tmp_path, opts, fake_run) -> None:
    fake_run["match"] = a_match(score=0.34, verdict="needs_review")
    source = local(tmp_path, "Somebody - The Original (Neel Remix)")
    (entry,) = pipeline.add([source], opts)

    fake_run["match"] = a_match(score=0.34, verdict="needs_review")
    kept = pipeline.accept(entry.slug, opts)
    assert kept.status == "done"
    assert (opts.pairs / f"{entry.slug}.original.mp3").is_file()
    assert kept.verdict == "accepted by hand"


def test_rejecting_a_reviewed_pair_keeps_the_remix_alone(tmp_path, opts,
                                                         fake_run) -> None:
    fake_run["match"] = a_match(score=0.34, verdict="needs_review")
    source = local(tmp_path, "Somebody - The Original (Neel Remix)")
    (entry,) = pipeline.add([source], opts)

    dropped = pipeline.reject(entry.slug, opts)
    assert dropped.status == "standalone"
    assert not dropped.original_url
    assert Path(dropped.remix_path).parent == opts.remixes


def test_accepting_something_that_was_never_ingested_says_so(opts) -> None:
    with pytest.raises(pipeline.PipelineError):
        pipeline.accept("no-such-slug", opts)


def test_a_kit_that_cannot_be_built_is_one_warning_not_a_failure(
        tmp_path, opts, fake_run, monkeypatch) -> None:
    import fourfloor.kit as kit_mod

    monkeypatch.undo()                              # restore the real _build_kit
    monkeypatch.setattr(pipeline.search, "find_original",
                        lambda parsed, **k: list(fake_run["candidates"]))
    monkeypatch.setattr(pipeline.fetch_mod, "fetch",
                        lambda *a, **k: (_ for _ in ()).throw(
                            pipeline.fetch_mod.FetchError("no")))
    monkeypatch.setattr(kit_mod, "build", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("no 8-bar stretch held a steady four-on-the-floor")))
    warnings: list[str] = []
    source = local(tmp_path, "Somebody - The Original (Neel Remix)")
    (entry,) = pipeline.add([source], opts,
                            on=lambda kind, text: warnings.append(f"{kind}:{text}"))

    assert entry.status == "standalone"
    assert entry.kit == ""
    assert any("no kit from this one" in w for w in warnings)


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------

def test_two_links_that_parse_to_the_same_name_do_not_collide(tmp_path, opts,
                                                              fake_run) -> None:
    (tmp_path / "a").mkdir(exist_ok=True)
    (tmp_path / "b").mkdir(exist_ok=True)
    one = local(tmp_path / "a", "Somebody - The Original (Neel Remix)")
    two = local(tmp_path / "b", "Somebody - The Original (Neel Remix) FREE DL")
    entries = pipeline.add([one, two], opts)
    assert entries[0].slug != entries[1].slug
    assert entries[1].slug.endswith("-2")


def test_a_dry_run_says_when_it_would_not_fetch_anything(tmp_path, opts, fake_run,
                                                         capsys) -> None:
    """A candidate too weak to download must not read as "would fetch"."""
    fake_run["candidates"][0].score = 0.05
    opts.dry_run = True
    source = local(tmp_path, "Somebody - The Original (Neel Remix)")
    lines: list[str] = []
    pipeline.add([source], opts, on=lambda kind, text: lines.append(text))
    assert any("nothing worth fetching" in t and "standalone" in t for t in lines)


def test_a_dry_run_writes_nothing_at_all(tmp_path, opts, fake_run) -> None:
    opts.dry_run = True
    source = local(tmp_path, "Somebody - The Original (Neel Remix)")
    (entry,) = pipeline.add([source], opts)

    assert entry.status == "dry-run"
    assert entry.original_title == "Somebody - The Original"
    assert not opts.ledger_path.exists()
    assert not opts.pairs.exists()
    assert not fake_run["fetches"]


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------

def test_the_ledger_survives_being_half_written(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text('{"schema": 1, "links": [{"url": "a", "slug": "a"', encoding="utf8")
    led = Ledger(path)
    assert led.entries == []
    assert (tmp_path / "ledger.json.broken").is_file()   # nothing was thrown away
    led.put(Entry(url="b", slug="b", status="done"))
    assert Ledger(path).by_url("b").status == "done"


def test_the_ledger_stamps_and_replaces_by_link(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    led = Ledger(path)
    led.put(Entry(url="a", slug="a", status="pending"))
    added = led.by_url("a").added
    led.put(Entry(url="a", slug="a", status="done"))
    assert len(led.entries) == 1
    assert led.by_url("a").status == "done"
    assert led.by_url("a").added == added          # the first time it was seen


def test_a_free_slug_never_takes_a_name_that_is_used(tmp_path) -> None:
    led = Ledger(tmp_path / "ledger.json")
    led.put(Entry(url="a", slug="body"))
    assert led.free_slug("body") == "body-2"
    assert led.free_slug("body", taken={"body-2"}) == "body-3"


def test_links_come_out_of_a_text_file_one_per_line(tmp_path) -> None:
    path = tmp_path / "links.txt"
    path.write_text("# my set\nhttps://a.example/1\n\n  https://b.example/2  \n"
                    "https://c.example/3 # with a note\n", encoding="utf8")
    assert pipeline.read_links(path) == ["https://a.example/1", "https://b.example/2",
                                         "https://c.example/3"]


def test_a_file_on_this_machine_is_not_a_link(tmp_path) -> None:
    path = local(tmp_path, "something")
    assert pipeline.is_local(path)
    assert not pipeline.is_local("https://www.youtube.com/watch?v=x")
    assert not pipeline.is_local("/no/such/file.mp3")


# ---------------------------------------------------------------------------
# what the pairs teach
# ---------------------------------------------------------------------------

def test_a_voice_cut_into_short_pieces_reads_as_chopped() -> None:
    got = learned.treatment_of(a_fingerprint(0.75, 2.4), a_fingerprint(0.3, 0.7),
                               tempo_ratio=0.9)
    assert got["treatment"] == "chopped"
    assert got["vocal_kept"] == pytest.approx(0.4)


def test_a_voice_played_as_sung_reads_as_continuous() -> None:
    got = learned.treatment_of(a_fingerprint(0.7, 2.6), a_fingerprint(0.66, 2.4),
                               tempo_ratio=0.95)
    assert got["treatment"] == "continuous"
    assert got["phrase_ratio"] > learned.CHOP_RATIO


def test_a_triplet_flow_is_recognised_as_one() -> None:
    assert learned.lattice_of({"straight": 0.5, "triplet": 0.75}) == "triplet"
    assert learned.lattice_of({"straight": 0.8, "triplet": 0.5}) == "straight"
    assert learned.lattice_of({}) == "unknown"


def test_the_table_counts_the_cells_and_sets_the_threshold() -> None:
    rows = [
        {"original_lattice": "straight", "treatment": "continuous", "straight": 0.78,
         "tempo_ratio": 0.9, "vocal_kept": 0.9, "bpm_remix": 128.0},
        {"original_lattice": "straight", "treatment": "continuous", "straight": 0.74,
         "tempo_ratio": 0.92, "vocal_kept": 0.8, "bpm_remix": 130.0},
        {"original_lattice": "triplet", "treatment": "chopped", "straight": 0.52,
         "tempo_ratio": 0.88, "vocal_kept": 0.4, "bpm_remix": 130.0},
    ]
    table = learned.table(rows)
    assert table["n_pairs"] == 3
    assert table["cells"]["straight/continuous"]["n"] == 2
    assert table["cells"]["triplet/chopped"]["n"] == 1
    assert 0.52 <= table["straight_lock"] < 0.74   # between the two groups
    assert table["separates"] is True


def test_a_table_whose_groups_overlap_teaches_nothing() -> None:
    """Two remixers chopped tighter voices than a third played straight: on
    this evidence the fit does not predict the treatment, and the honest
    answer is no threshold rather than one drawn through the overlap."""
    rows = [
        {"original_lattice": "straight", "treatment": "continuous", "straight": 0.50},
        {"original_lattice": "straight", "treatment": "chopped", "straight": 0.57},
        {"original_lattice": "straight", "treatment": "chopped", "straight": 0.47},
    ]
    table = learned.table(rows)
    assert table["straight_lock"] is None
    assert table["separates"] is False
    assert "does not separate" in table["note"]


# ---------------------------------------------------------------------------
# the two call sites that read it
# ---------------------------------------------------------------------------

@pytest.fixture
def learned_style(tmp_path, monkeypatch):
    """A style profile in an isolated home, as ``refs learn`` would write it."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("FOURFLOOR_HOME", str(home))
    learned.forget()
    yield home
    learned.forget()


def write_style(home: Path, **fields) -> None:
    data = {"bpm": 130.0, "n_refs": 6, "vocal": {"n_pairs": 4, "straight_lock": 0.62,
                                                 "triplet_edge": 0.06}}
    data.update(fields)
    (home / "style.json").write_text(json.dumps(data), encoding="utf8")
    learned.forget()


def test_with_nothing_learned_the_defaults_are_untouched(learned_style) -> None:
    from fourfloor.analysis import suggest_house_tempo

    assert learned.load() is None
    assert learned.vocal_thresholds() is None
    assert suggest_house_tempo(120.0) == 124.0     # the tie-break, as it always was
    assert 120.0 <= suggest_house_tempo(97.0) <= 128.0


def test_a_learned_tempo_becomes_the_default_target(learned_style) -> None:
    from fourfloor.analysis import suggest_house_tempo

    before = suggest_house_tempo(97.0)
    write_style(learned_style, bpm=130.0, n_refs=6)
    after = suggest_house_tempo(97.0)
    assert learned.learned_bpm() == 130.0
    assert after == 130.0
    assert after != before


def test_one_reference_is_not_enough_to_move_the_default(learned_style) -> None:
    from fourfloor.analysis import suggest_house_tempo

    write_style(learned_style, bpm=130.0, n_refs=1)
    assert learned.learned_bpm() is None
    assert suggest_house_tempo(97.0) <= 128.0


def test_a_learned_threshold_changes_what_vocal_auto_does(learned_style) -> None:
    """A voice at 0.55 straight is chopped by the built-in threshold and played
    by one learned from pairs whose remixers played looser voices than that."""
    import numpy as np

    from fourfloor.house import vocal as vocal_mod

    measured = {"straight": 0.55, "triplet": 0.40, "advantage": -0.15,
                "duty": 0.6, "scatter_ms": 30.0, "onsets_per_bar": 9.0}
    vocal_mod_fit = lambda *_a, **_k: measured                      # noqa: E731
    import fourfloor.analysis.alignment as align_mod
    original = align_mod.vocal_fit
    align_mod.vocal_fit = vocal_mod_fit
    try:
        silence = np.zeros(1024, dtype=np.float32)
        plain, _m, _why = vocal_mod.choose_vocal(silence, 44100, 128.0)
        write_style(learned_style, vocal={"n_pairs": 4, "straight_lock": 0.62,
                                          "triplet_edge": 0.06})
        taught, _m2, _why2 = vocal_mod.choose_vocal(silence, 44100, 128.0)
    finally:
        align_mod.vocal_fit = original
    assert plain == "chop"
    assert taught == "flow"


def test_a_table_from_too_few_pairs_is_not_trusted(learned_style) -> None:
    write_style(learned_style, vocal={"n_pairs": 1, "straight_lock": 0.62})
    assert learned.vocal_thresholds() is None


# ---------------------------------------------------------------------------
# learning
# ---------------------------------------------------------------------------

def test_learning_writes_a_private_profile_and_an_anonymous_one(
        tmp_path, opts, learned_style) -> None:
    opts.pairs.mkdir(parents=True)
    opts.remixes.mkdir(parents=True)
    shutil.copyfile(FIXTURE, opts.pairs / "one.remix.mp3")
    shutil.copyfile(FIXTURE, opts.pairs / "one.original.mp3")
    (opts.pairs / "one.json").write_text(json.dumps({
        "slug": "one", "vocal": {"original_lattice": "straight",
                                 "treatment": "continuous", "straight": 0.8,
                                 "tempo_ratio": 0.9, "vocal_kept": 0.9,
                                 "bpm_remix": 128.0}}), encoding="utf8")
    repo = tmp_path / "styles" / "pairs.json"
    repo.parent.mkdir()

    result = pipeline.learn(opts, repo_out=repo)

    assert result["n_files"] == 1          # the remix half; the original is not house
    assert result["n_pairs"] == 1
    private = json.loads((learned_style / "style.json").read_text(encoding="utf8"))
    assert private["vocal"]["cells"]["straight/continuous"]["n"] == 1
    assert private["per_track"]                       # the private one keeps them

    public = json.loads(repo.read_text(encoding="utf8"))
    assert "per_track" not in public                  # the shared one never does
    assert json.dumps(public).lower().count(".mp3") == 0
    assert public["n_pairs"] == 1


def test_learning_with_no_references_says_so(opts) -> None:
    with pytest.raises(pipeline.PipelineError):
        pipeline.learn(opts)
