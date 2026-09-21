"""The Gemini ear, with the network replaced by a scripted fake.

No test here reaches the internet or needs a key. ``urlopen`` is swapped
for a recorder that answers from a script and remembers what it was
asked, so the tests can assert on the request the critic *would* have
made -- which model it picked, that the key went in the header and not
the URL, that the reference audio was labelled -- as well as on how it
handles the replies, including the malformed ones a model actually
produces.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fourfloor.critic import gemini as G

KEY = "AIza-test-key-not-real"


# ---------------------------------------------------------------------------
# the fake transport
# ---------------------------------------------------------------------------

class _Response:
    def __init__(self, body: bytes, headers: dict | None = None):
        self._body = body
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Recorder:
    """Answers urlopen from a script and keeps every request."""

    def __init__(self, script: dict, default=None):
        self.script = script
        self.default = default
        self.calls: list[tuple[str, str, dict, bytes | None]] = []

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.calls.append((req.get_method(), url, dict(req.headers), req.data))
        for fragment, answer in self.script.items():
            if fragment in url:
                if isinstance(answer, Exception):
                    raise answer
                return answer
        if self.default is not None:
            if isinstance(self.default, Exception):
                raise self.default
            return self.default
        raise AssertionError(f"unscripted request to {url}")

    def body_for(self, fragment: str) -> dict:
        for _method, url, _headers, data in self.calls:
            if fragment in url and data:
                return json.loads(data)
        raise AssertionError(f"no request body for {fragment}")


MODELS = _Response(json.dumps({"models": [
    {"name": "models/gemini-1.5-pro", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
]}).encode())

GOOD = {
    "overall_score": 62,
    "verdict": "The kick is solid but the vocal is fighting the hats.",
    "issues": [
        {"time_sec": 46.5, "category": "off_beat", "severity": 3, "note": "drop is late"},
        {"time_sec": 90.0, "category": "good", "severity": 1, "note": "nice filter sweep"},
    ],
    "sections": {"drop 1": "lands hard", "breakdown": "too long"},
    "vs_reference": {"reference_does_better": ["tighter low end"],
                     "biggest_gap": "the drums are not swinging"},
}


def _generate(payload: dict | str) -> _Response:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return _Response(json.dumps(
        {"candidates": [{"content": {"parts": [{"text": text}]}}]}).encode())


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Keep the retry ladder honest but instant."""
    monkeypatch.setattr(G, "RETRY_BASE", 0.0)


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    f = tmp_path / "render.mp3"
    f.write_bytes(b"ID3" + b"\x00" * 2048)      # small enough to go inline
    return f


# ---------------------------------------------------------------------------
# model discovery
# ---------------------------------------------------------------------------

def test_it_picks_the_newest_audio_capable_model(monkeypatch):
    rec = Recorder({"/models": MODELS})
    monkeypatch.setattr(urllib.request, "urlopen", rec)
    assert G.Ear(KEY).pick_model() == "gemini-2.5-pro"


def test_it_falls_back_when_the_preferred_family_is_absent(monkeypatch):
    only_flash = _Response(json.dumps({"models": [
        {"name": "models/gemini-2.0-flash", "supportedGenerationMethods": ["generateContent"]},
    ]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", Recorder({"/models": only_flash}))
    assert G.Ear(KEY).pick_model() == "gemini-2.0-flash"


def test_a_preview_variant_beats_the_plain_name(monkeypatch):
    previews = _Response(json.dumps({"models": [
        {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-2.5-pro-preview-11-20",
         "supportedGenerationMethods": ["generateContent"]},
    ]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", Recorder({"/models": previews}))
    # an exact match wins, so a rename cannot silently move us to a preview
    assert G.Ear(KEY).pick_model() == "gemini-2.5-pro"


def test_an_explicit_model_skips_discovery(monkeypatch):
    rec = Recorder({})
    monkeypatch.setattr(urllib.request, "urlopen", rec)
    assert G.Ear(KEY, "gemini-2.5-flash").pick_model() == "gemini-2.5-flash"
    assert rec.calls == []


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------

def test_a_full_review_sends_labelled_audio_parts(monkeypatch, audio, tmp_path):
    ref, source = tmp_path / "ref.mp3", tmp_path / "src.mp3"
    ref.write_bytes(b"ID3" + b"\x01" * 2048)
    source.write_bytes(b"ID3" + b"\x02" * 2048)
    rec = Recorder({"/models?": MODELS, ":generateContent": _generate(GOOD)})
    monkeypatch.setattr(urllib.request, "urlopen", rec)

    out = G.Ear(KEY).review(audio, ref=ref, source=source,
                            session={"cues": [{"name": "drop 1", "time": 46.5}]})

    assert out["overall_score"] == 62.0
    assert out["model"] == "gemini-2.5-pro"
    assert out["vs_reference"]["biggest_gap"]

    parts = rec.body_for(":generateContent")["contents"][0]["parts"]
    audio_parts = [p for p in parts if "inline_data" in p or "file_data" in p]
    assert len(audio_parts) == 3, "render, reference and source should all be sent"
    text = " ".join(p["text"] for p in parts if "text" in p)
    assert "REFERENCE" in text and "SOURCE" in text and "THE RENDER" in text
    assert "drop 1" in text, "session cues should steer the section commentary"


def test_the_key_travels_in_a_header_never_in_the_url(monkeypatch, audio):
    rec = Recorder({"/models?": MODELS, ":generateContent": _generate(GOOD)})
    monkeypatch.setattr(urllib.request, "urlopen", rec)
    G.Ear(KEY).review(audio)
    for _method, url, headers, _data in rec.calls:
        assert KEY not in url
        assert headers.get("X-goog-api-key") == KEY


def test_a_small_file_goes_inline_and_a_large_one_is_uploaded(monkeypatch, tmp_path):
    big = tmp_path / "big.mp3"
    big.write_bytes(b"\x00" * (G.INLINE_LIMIT + 1024))
    start = _Response(b"{}", {"X-Goog-Upload-URL": "https://upload.example/session"})
    finish = _Response(json.dumps(
        {"file": {"uri": "https://files/abc", "name": "files/abc", "state": "ACTIVE"}}).encode())
    rec = Recorder({"upload/v1beta/files": start, "upload.example": finish,
                    "v1beta/files/abc": _Response(json.dumps({"state": "ACTIVE"}).encode())})
    monkeypatch.setattr(urllib.request, "urlopen", rec)

    part = G.Ear(KEY).part_for(big, "render")
    assert part["file_data"]["file_uri"] == "https://files/abc"


def test_an_http_error_is_reported_without_the_key(monkeypatch, audio):
    boom = urllib.error.HTTPError("https://x", 429, "Too Many Requests", {}, None)
    boom.read = lambda: b'{"error": {"message": "quota"}}'
    monkeypatch.setattr(urllib.request, "urlopen", Recorder({"/models": boom}))
    with pytest.raises(G.GeminiError) as exc:
        G.Ear(KEY).review(audio)
    assert "429" in str(exc.value) and KEY not in str(exc.value)


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------

def test_a_fenced_reply_is_unwrapped():
    out = G.repair_json("```json\n" + json.dumps(GOOD) + "\n```")
    assert out["overall_score"] == 62.0


def test_prose_around_the_object_is_discarded():
    out = G.repair_json("Sure! Here you go:\n" + json.dumps(GOOD) + "\nHope that helps.")
    assert out["verdict"].startswith("The kick")


def test_an_invented_category_becomes_artifact():
    out = G.repair_json(json.dumps({"overall_score": 50, "verdict": "x", "issues": [
        {"time_sec": 1, "category": "vibes_are_off", "severity": 2, "note": "n"}]}))
    assert out["issues"][0]["category"] == "artifact"


def test_severity_and_score_are_clamped():
    out = G.repair_json(json.dumps({"overall_score": 250, "verdict": "x", "issues": [
        {"time_sec": -4, "category": "off_beat", "severity": 9, "note": "n"}]}))
    assert out["overall_score"] == 100.0
    assert out["issues"][0]["severity"] == 3
    assert out["issues"][0]["time_sec"] == 0.0


def test_a_score_given_as_a_fraction_is_rescaled():
    assert G.repair_json('{"overall_score": 0.62, "verdict": "x"}')["overall_score"] == 62.0


def test_junk_raises_rather_than_inventing_a_score():
    with pytest.raises(G.GeminiError):
        G.repair_json("I am unable to listen to audio.")


def test_missing_optional_blocks_are_filled_in():
    out = G.repair_json('{"overall_score": 70, "verdict": "fine"}')
    assert out["issues"] == [] and out["sections"] == {}
    assert "vs_reference" not in out


# ---------------------------------------------------------------------------
# key handling
# ---------------------------------------------------------------------------

def test_the_env_var_wins_over_the_key_file(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    monkeypatch.setattr(G, "KEY_FILE", tmp_path / "gemini.key")
    (tmp_path / "gemini.key").write_text("from-file\n")
    assert G.load_key() == "from-env"


def test_the_key_file_is_read_and_stripped(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(G, "KEY_FILE", tmp_path / "gemini.key")
    (tmp_path / "gemini.key").write_text("  from-file \n")
    assert G.load_key() == "from-file"


def test_a_missing_key_explains_both_ways_to_supply_one(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(G, "KEY_FILE", tmp_path / "absent.key")
    with pytest.raises(G.GeminiError) as exc:
        G.load_key()
    assert "GEMINI_API_KEY" in str(exc.value)


# ---------------------------------------------------------------------------
# feedback markers
# ---------------------------------------------------------------------------

SESSION = {"beat_duration_sec": 0.4839, "beats_per_bar": 4, "first_downbeat_sec": 0.0}


def test_markers_are_written_only_for_renders_in_the_remix_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("FOURFLOOR_HOME", str(tmp_path))
    outside = tmp_path / "elsewhere" / "x.mp3"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"")
    assert G.feedback_path(outside) is None

    inside = tmp_path / "remixes" / "abc123" / "x.mp3"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"")
    assert G.feedback_path(inside) == inside.parent / "feedback.json"


def test_markers_carry_the_bar_and_the_author(monkeypatch, tmp_path):
    monkeypatch.setenv("FOURFLOOR_HOME", str(tmp_path))
    render = tmp_path / "remixes" / "abc123" / "x.mp3"
    render.parent.mkdir(parents=True)
    render.write_bytes(b"")

    written = G.append_markers(render, GOOD, SESSION)
    doc = json.loads(written.read_text())
    assert [m["author"] for m in doc["markers"]] == ["gemini", "gemini"]
    assert doc["markers"][0]["bar"] == 24          # 46.5s at 0.4839s/beat, 4/4
    assert doc["markers"][0]["category"] == "off_beat"
    assert set(doc["markers"][0]) == {"time", "bar", "category", "note", "author", "ts"}


def test_a_rerun_replaces_its_own_markers_and_keeps_everyone_elses(monkeypatch, tmp_path):
    monkeypatch.setenv("FOURFLOOR_HOME", str(tmp_path))
    render = tmp_path / "remixes" / "abc123" / "x.mp3"
    render.parent.mkdir(parents=True)
    render.write_bytes(b"")
    target = render.parent / "feedback.json"
    target.write_text(json.dumps({"markers": [
        {"time": 12.0, "bar": 6, "category": "off_beat", "note": "neel heard this",
         "author": "neel", "ts": "2026-09-19T00:00:00Z"}]}))

    G.append_markers(render, GOOD, SESSION)
    G.append_markers(render, GOOD, SESSION)
    markers = json.loads(target.read_text())["markers"]
    assert sum(m["author"] == "neel" for m in markers) == 1
    assert sum(m["author"] == "gemini" for m in markers) == 2, "no duplicate pile-up"
    assert [m["time"] for m in markers] == sorted(m["time"] for m in markers)


# ---------------------------------------------------------------------------
# model ranking and fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("newer,older", [
    ("gemini-3.8-flash", "gemini-3.5-flash"),
    ("gemini-3.5-flash", "gemini-2.5-pro"),
    ("gemini-2.5-pro", "gemini-2.5-flash"),          # tier breaks a version tie
    ("gemini-2.5-flash", "gemini-2.5-flash-lite"),   # "flash-lite" is not "flash"
    ("gemini-3.1-pro", "gemini-3.1-pro-preview"),    # released beats preview
])
def test_model_ranking_prefers_the_newer_name(newer, older):
    assert G.rank_model(newer) > G.rank_model(older)


@pytest.mark.parametrize("name", [
    "gemini-3.1-flash-image", "gemini-3.5-transcribe", "gemini-2.5-flash-preview-tts",
    "gemma-4-31b-it", "lyria-3.5", "text-embedding-004", "gemini-2.5-computer-use-preview",
])
def test_models_that_cannot_review_audio_are_rejected(name):
    assert G.rank_model(name) is None


def test_a_retired_top_model_falls_through_to_the_next(monkeypatch, audio):
    """The failure that actually happened: 2.5-pro was listed but answered 404."""
    listing = _Response(json.dumps({"models": [
        {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"]},
    ]}).encode())
    gone = urllib.error.HTTPError("https://x", 404, "Not Found", {}, None)
    gone.read = lambda: b'{"error":{"message":"no longer available to new users"}}'

    seen: list[str] = []

    def transport(req, timeout=None):
        url = req.full_url
        if "/models?" in url:
            return listing
        seen.append(url.split("/models/")[1].split(":")[0])
        if seen[-1] == "gemini-3.8-flash":
            raise gone
        return _generate(GOOD)

    monkeypatch.setattr(urllib.request, "urlopen", transport)
    out = G.Ear(KEY).review(audio)
    assert seen == ["gemini-3.8-flash", "gemini-3.5-flash"]
    assert out["model"] == "gemini-3.5-flash"


def test_a_saturated_model_is_retried_before_being_abandoned(monkeypatch, audio):
    busy = urllib.error.HTTPError("https://x", 503, "Unavailable", {}, None)
    busy.read = lambda: b'{"error":{"message":"high demand"}}'
    attempts = {"n": 0}

    def transport(req, timeout=None):
        if "/models?" in req.full_url:
            return MODELS
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise busy
        return _generate(GOOD)

    monkeypatch.setattr(urllib.request, "urlopen", transport)
    assert G.Ear(KEY).review(audio)["overall_score"] == 62.0
    assert attempts["n"] == 3, "two retries, then the answer"


def test_a_real_error_is_not_swallowed_by_the_fallback(monkeypatch, audio):
    """A 400 is our bug. Walking to the next model would only hide it."""
    bad = urllib.error.HTTPError("https://x", 400, "Bad Request", {}, None)
    bad.read = lambda: b'{"error":{"message":"malformed part"}}'
    monkeypatch.setattr(urllib.request, "urlopen",
                        Recorder({"/models?": MODELS}, default=bad))
    with pytest.raises(G.GeminiError) as exc:
        G.Ear(KEY).review(audio)
    assert "400" in str(exc.value)
