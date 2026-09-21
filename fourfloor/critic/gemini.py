"""A second opinion from a model that can actually hear the render.

The local critic measures; this asks. It sends the render -- and, when
offered, the reference remix and the original song -- to Gemini as audio
parts, with a prompt that puts the model in the chair of a house producer
auditioning a remix for a gig, and requires strict JSON back.

Nothing here runs unless ``--ear gemini`` is passed. The API key is read
from ``GEMINI_API_KEY`` or ``~/.fourfloor/gemini.key`` and is never
printed, logged or written anywhere; error messages from this module are
scrubbed of it before they surface.

stdlib ``urllib`` only, no SDK: one fewer dependency in a project whose
install already pulls torch.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API_ROOT = "https://generativelanguage.googleapis.com"
KEY_FILE = Path(os.environ.get("FOURFLOOR_HOME", Path.home() / ".fourfloor")) / "gemini.key"
INLINE_LIMIT = 18 * 1024 * 1024     # below this, base64 inline beats an upload
INLINE_BUDGET = 12 * 1024 * 1024    # total bytes allowed inline across all parts
TIMEOUT = 240
RETRY_BASE = 3.0                    # seconds; doubles per attempt

CATEGORIES = ("off_beat", "vocal_buried", "drums_fake", "wrong_notes",
              "rough_transition", "too_loud", "too_quiet", "repetitive",
              "artifact", "good")

#: Model families that cannot review a piece of audio, whatever their
#: version: image and speech generators, embedders, transcribers, the
#: robotics and computer-use towers, and the open-weight Gemma line.
_NOT_A_CRITIC = ("image", "tts", "transcribe", "embedding", "computer-use",
                 "robotics", "customtools", "vision", "gemma", "lyria",
                 "banana", "antigravity", "deep-research")

#: Tier ordering within one version, most specific name first so that
#: "flash-lite" is not read as "flash". Breaks a version tie only.
_TIERS = (("flash-lite", 1), ("pro", 3), ("flash", 2))

#: Aliases Google keeps pointed at a current model. The safety net for
#: the day the naming scheme changes again.
_ALIASES = ("gemini-pro-latest", "gemini-flash-latest")

_VERSION = re.compile(r"^gemini-(\d+)(?:\.(\d+))?-")


def rank_model(name: str) -> tuple | None:
    """Sort key for an audio-capable Gemini, or ``None`` if it is not one.

    Deliberately parsed rather than listed. The first version of this
    module hard-coded ``gemini-2.5-pro`` as the preferred model; by the
    time it first ran against a real key that model answered 404 with
    "no longer available to new users". A name like ``gemini-3.8-flash``
    carries its own ordering, so reading the version out of it keeps
    working across releases that have not happened yet.
    """
    if not name.startswith("gemini-") or any(bad in name for bad in _NOT_A_CRITIC):
        return None
    m = _VERSION.match(name)
    if not m:
        return None
    major, minor = int(m.group(1)), int(m.group(2) or 0)
    tier = next((rank for key, rank in _TIERS if key in name), 0)
    return (major, minor, tier, 0 if "preview" in name else 1)


class GeminiError(RuntimeError):
    """Anything that went wrong talking to the API, with the key removed."""


@dataclass
class Ear:
    key: str
    model: str | None = None

    # -- plumbing ---------------------------------------------------------

    def _scrub(self, text: str) -> str:
        return text.replace(self.key, "<key>") if self.key else text

    def _request(self, method: str, url: str, body: bytes | None = None,
                 headers: dict | None = None, retries: int = 2) -> bytes:
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=body, method=method)
            req.add_header("x-goog-api-key", self.key)
            for k, v in (headers or {}).items():
                req.add_header(k, v)
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf8", "replace")[:400]
                err = GeminiError(
                    self._scrub(f"HTTP {exc.code} from {_path_of(url)}: {detail}"))
                err.status = exc.code
                # 429 and 503 are "ask again", not "this will never work"
                if exc.code in (429, 503) and attempt < retries:
                    time.sleep(RETRY_BASE * 2 ** attempt)
                    continue
                raise err from None
            except urllib.error.URLError as exc:
                raise GeminiError(
                    self._scrub(f"cannot reach the Gemini API: {exc.reason}")) from None
        raise GeminiError("unreachable")

    def _json(self, method: str, url: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        head = {"Content-Type": "application/json"} if payload is not None else {}
        raw = self._request(method, url, body, head)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise GeminiError("the API returned something that was not JSON") from None

    # -- model discovery --------------------------------------------------

    def candidates(self) -> list[str]:
        """Audio-capable models this key is served, newest first.

        A list rather than one name because "the endpoint lists it" and
        "the endpoint will answer for it" are different questions: a
        retired model still appears and answers 404, and a popular one
        answers 503 under load. The caller walks down until one replies.
        """
        if self.model:
            return [self.model]
        data = self._json("GET", f"{API_ROOT}/v1beta/models?pageSize=200")
        served = []
        for m in data.get("models", []):
            name = m.get("name", "").split("/")[-1]
            methods = m.get("supportedGenerationMethods") or m.get("supportedActions") or []
            if "generateContent" in methods:
                served.append(name)
        ranked = [n for _r, n in sorted(((rank_model(n), n) for n in served
                                         if rank_model(n)), reverse=True)]
        ranked += [a for a in _ALIASES if a in served]
        if not ranked:
            raise GeminiError("this key has no model that can review audio")
        return ranked

    def pick_model(self) -> str:
        """The newest audio-capable model this key can actually call."""
        if self.model:
            return self.model
        data = self._json("GET", f"{API_ROOT}/v1beta/models?pageSize=200")
        served = []
        for m in data.get("models", []):
            name = m.get("name", "").split("/")[-1]
            methods = m.get("supportedGenerationMethods") or m.get("supportedActions") or []
            if "generateContent" in methods:
                served.append(name)
        ranked = sorted(((rank_model(n), n) for n in served if rank_model(n)),
                        reverse=True)
        if ranked:
            self.model = ranked[0][1]
            return self.model
        for alias in _ALIASES:
            if alias in served:
                self.model = alias
                return self.model
        raise GeminiError("this key has no model that can review audio")

    # -- files ------------------------------------------------------------

    def part_for(self, path: Path, label: str, inline_ok: bool = True) -> dict:
        """An audio part: inline base64 when small, a Files API handle when not.

        ``inline_ok`` is how the caller enforces the *total* request
        budget. Each of a render, a reference and a source can sit under
        the per-file limit while the three of them together, base64
        expanded by a third, blow past what ``generateContent`` accepts
        in one body.
        """
        mime = mimetypes.guess_type(str(path))[0] or "audio/mpeg"
        if inline_ok and path.stat().st_size <= INLINE_LIMIT:
            data = base64.b64encode(path.read_bytes()).decode()
            return {"inline_data": {"mime_type": mime, "data": data}}
        return {"file_data": {"mime_type": mime, "file_uri": self.upload(path, label, mime)}}

    def upload(self, path: Path, label: str, mime: str) -> str:
        """Resumable upload through the Files API; returns the file URI."""
        size = path.stat().st_size
        start = urllib.request.Request(
            f"{API_ROOT}/upload/v1beta/files", method="POST",
            data=json.dumps({"file": {"display_name": label}}).encode(),
        )
        start.add_header("x-goog-api-key", self.key)
        start.add_header("X-Goog-Upload-Protocol", "resumable")
        start.add_header("X-Goog-Upload-Command", "start")
        start.add_header("X-Goog-Upload-Header-Content-Length", str(size))
        start.add_header("X-Goog-Upload-Header-Content-Type", mime)
        start.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(start, timeout=TIMEOUT) as resp:
                session_url = resp.headers.get("X-Goog-Upload-URL")
        except urllib.error.HTTPError as exc:
            raise GeminiError(self._scrub(f"upload start failed: HTTP {exc.code}")) from None
        if not session_url:
            raise GeminiError("the API did not return an upload URL")
        raw = self._request("POST", session_url, path.read_bytes(), {
            "Content-Length": str(size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        })
        info = json.loads(raw).get("file", {})
        uri, name = info.get("uri"), info.get("name")
        if not uri:
            raise GeminiError("the upload finished without a file URI")
        self._await_active(name)
        return uri

    def _await_active(self, name: str | None, tries: int = 30) -> None:
        """Files are PROCESSING for a few seconds after upload."""
        if not name:
            return
        for _ in range(tries):
            info = self._json("GET", f"{API_ROOT}/v1beta/{name}")
            state = info.get("state", "ACTIVE")
            if state == "ACTIVE":
                return
            if state == "FAILED":
                raise GeminiError("the API could not process the uploaded audio")
            time.sleep(2)
        raise GeminiError("the uploaded audio never became ready")

    # -- the ask ----------------------------------------------------------

    def review(self, render: Path, ref: Path | None = None, source: Path | None = None,
               session: dict | None = None, on_step=None) -> dict:
        sending = [p for p in (render, ref, source) if p]
        total = sum(p.stat().st_size for p in sending)
        inline_ok = total <= INLINE_BUDGET
        if on_step and not inline_ok:
            on_step(f"uploading {len(sending)} files")

        parts: list[dict] = [{"text": prompt_for(session, ref is not None, source is not None)}]
        parts.append({"text": "AUDIO 1 -- THE RENDER under review:"})
        parts.append(self.part_for(render, "render", inline_ok))
        if ref:
            parts.append({"text": "AUDIO 2 -- REFERENCE: a real, professionally "
                                  "produced house remix. This is the target."})
            parts.append(self.part_for(ref, "reference", inline_ok))
        if source:
            parts.append({"text": "AUDIO 3 -- SOURCE: the original, non-house song "
                                  "the render was made from."})
            parts.append(self.part_for(source, "source", inline_ok))
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json",
                                 "maxOutputTokens": 4096},
        }
        models = self.candidates()
        last: GeminiError | None = None
        for model in models[:4]:
            if on_step:
                on_step(f"gemini {model}")
            try:
                data = self._json(
                    "POST", f"{API_ROOT}/v1beta/models/{model}:generateContent", payload)
            except GeminiError as exc:
                # 404 (retired) and 503 (saturated) both mean "try the next
                # one"; anything else is ours to fix, so surface it.
                if getattr(exc, "status", None) in (404, 429, 503):
                    last = exc
                    continue
                raise
            text = _first_text(data)
            if not text:
                last = GeminiError(f"{model} returned no text")
                continue
            out = repair_json(text)
            out["model"] = model
            return out
        raise last or GeminiError("no served model answered")


def _path_of(url: str) -> str:
    return url.split("?")[0].split("generativelanguage.googleapis.com")[-1] or url


def _first_text(data: dict) -> str:
    for cand in data.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "text" in part:
                return part["text"]
    return ""


# ---------------------------------------------------------------------------
# prompt and JSON repair
# ---------------------------------------------------------------------------

def prompt_for(session: dict | None, has_ref: bool, has_source: bool) -> str:
    cues = ""
    if session and session.get("cues"):
        listed = ", ".join(f'"{c.get("name")}" at {float(c.get("time", 0)):.0f}s'
                           for c in session["cues"][:16])
        cues = ("\nThe render's own section cues are: " + listed +
                ". Key your `sections` commentary to those names.")
    extra = ""
    if has_ref:
        extra += ("\nAlso fill `vs_reference`: what the reference does better, "
                  "concretely (drum programming, low end, arrangement, vocal "
                  "treatment, transitions), and the single change that would "
                  "close the biggest gap.")
    if has_source:
        extra += ("\nUse the source only to judge whether the remix respects the "
                  "song: key, vocal phrasing, whether the hook survived.")
    return f"""You are a house producer and working DJ. Someone has handed you a
machine-generated house remix and asked whether you would play it out
tonight. Listen to the whole thing. Be blunt: a polite score is useless to
them. Pay particular attention to whether the elements are actually on the
beat with each other, and whether two parts are playing over each other in
a way that sounds like a mistake rather than a layer.{cues}{extra}

Reply with STRICT JSON only -- no prose, no code fence -- in exactly this shape:

{{
  "overall_score": <integer 0-100, where 90+ is a record you would play,
                    60 is a demo with promise, below 40 is unlistenable>,
  "verdict": "<one sentence, plain English, what is actually wrong or right>",
  "issues": [
    {{"time_sec": <number, where in the render you heard it>,
      "category": "<one of: {', '.join(CATEGORIES)}>",
      "severity": <1 minor, 2 clear, 3 ruins the track>,
      "note": "<one short sentence>"}}
  ],
  "sections": {{"<section name>": "<one sentence about that section>"}},
  "vs_reference": {{"reference_does_better": ["<point>"],
                    "biggest_gap": "<one sentence>"}}
}}

Give between 3 and 12 issues. Use the "good" category for things that
genuinely work. Omit "vs_reference" if you were given no reference."""


def repair_json(text: str) -> dict:
    """Parse the model's reply, forgiving the usual wrappers, then validate.

    Returns a dict with the documented shape. Anything the model got wrong
    -- a fence, a category it invented, a severity of 7, a score as a
    string -- is corrected here rather than raising, because a slightly
    malformed critique is still worth reading.
    """
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise GeminiError("the model's reply was not JSON") from None
        try:
            data = json.loads(raw[start: end + 1])
        except json.JSONDecodeError:
            raise GeminiError("the model's reply was not repairable JSON") from None
    if not isinstance(data, dict):
        raise GeminiError("the model's reply was not a JSON object")

    out: dict = {}
    out["overall_score"] = _as_score(data.get("overall_score"))
    out["verdict"] = str(data.get("verdict") or "").strip() or "(no verdict returned)"
    issues = []
    for item in data.get("issues") or []:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category", "")).strip().lower().replace(" ", "_")
        if cat not in CATEGORIES:
            cat = "artifact"
        try:
            t = max(0.0, float(item.get("time_sec", 0)))
        except (TypeError, ValueError):
            t = 0.0
        try:
            sev = int(round(float(item.get("severity", 2))))
        except (TypeError, ValueError):
            sev = 2
        issues.append({"time_sec": round(t, 2), "category": cat,
                       "severity": min(3, max(1, sev)),
                       "note": str(item.get("note", "")).strip()})
    out["issues"] = issues
    sections = data.get("sections")
    out["sections"] = ({str(k): str(v) for k, v in sections.items()}
                       if isinstance(sections, dict) else {})
    vs = data.get("vs_reference")
    if isinstance(vs, dict):
        better = vs.get("reference_does_better")
        out["vs_reference"] = {
            "reference_does_better": [str(x) for x in better] if isinstance(better, list)
            else ([str(better)] if better else []),
            "biggest_gap": str(vs.get("biggest_gap", "")).strip(),
        }
    return out


def _as_score(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 < v <= 1.0:      # the model answered with a fraction
        v *= 100.0
    return round(min(100.0, max(0.0, v)), 1)


# ---------------------------------------------------------------------------
# key handling and feedback markers
# ---------------------------------------------------------------------------

def load_key() -> str:
    """The API key, from the environment or the key file. Never logged."""
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if key:
        return key
    if KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
        if key:
            return key
    raise GeminiError(
        f"no Gemini key. Set GEMINI_API_KEY or put the key in {KEY_FILE} "
        "(one line, chmod 600). It is never committed or printed."
    )


def feedback_path(render: Path) -> Path | None:
    """``feedback.json`` for a render that lives in a fourfloor remix folder."""
    home = Path(os.environ.get("FOURFLOOR_HOME", Path.home() / ".fourfloor")) / "remixes"
    try:
        render.resolve().relative_to(home.resolve())
    except (ValueError, OSError):
        return None
    return render.resolve().parent / "feedback.json"


def append_markers(render: Path, result: dict, session: dict | None) -> Path | None:
    """Write Gemini's issues into the remix folder's shared feedback file.

    Same marker schema the feedback UI writes, so a human note and a model
    note sit in one list: ``{time, bar, category, note, author, ts}``.
    Existing markers are preserved; previous ``gemini`` markers are
    replaced, so re-running does not pile up duplicates.
    """
    target = feedback_path(render)
    if target is None:
        return None
    beat = float((session or {}).get("beat_duration_sec") or 0.0)
    per_bar = int((session or {}).get("beats_per_bar") or 4)
    first = float((session or {}).get("first_downbeat_sec") or 0.0)
    bar_len = beat * per_bar

    doc: dict = {"markers": []}
    if target.exists():
        try:
            loaded = json.loads(target.read_text())
            if isinstance(loaded, dict):
                doc = loaded
        except (json.JSONDecodeError, OSError):
            pass
    markers = [m for m in doc.get("markers", [])
               if isinstance(m, dict) and m.get("author") != "gemini"]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for issue in result.get("issues", []):
        t = float(issue.get("time_sec", 0.0))
        markers.append({
            "time": round(t, 2),
            "bar": int(max(0.0, t - first) // bar_len) if bar_len > 0 else None,
            "category": issue.get("category", "artifact"),
            "note": issue.get("note", ""),
            "author": "gemini",
            "ts": stamp,
        })
    doc["markers"] = sorted(markers, key=lambda m: (m.get("time") or 0.0))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc, indent=2) + "\n")
    return target


def listen(render: Path, ref: Path | None = None, source: Path | None = None,
           session: dict | None = None, model: str | None = None,
           on_step=None) -> dict:
    """One full Gemini pass. Raises :class:`GeminiError`; callers fall back."""
    return Ear(load_key(), model).review(render, ref, source, session, on_step)
