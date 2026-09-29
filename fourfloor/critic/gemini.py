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
import http.client
import json
import math
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

API_ROOT = "https://generativelanguage.googleapis.com"
KEY_FILE = Path(os.environ.get("FOURFLOOR_HOME", Path.home() / ".fourfloor")) / "gemini.key"
INLINE_LIMIT = 18 * 1024 * 1024     # below this, base64 inline beats an upload
INLINE_BUDGET = 12 * 1024 * 1024    # total bytes allowed inline across all parts
TIMEOUT = 240
RETRY_BASE = 3.0                    # seconds; doubles per attempt

#: Output cap when the listing did not say what a model allows (a forced
#: ``--model``). Every audio-capable Gemini back to 1.5 accepts 8192; the
#: thinking models count their thinking against this cap, so the old 4096
#: could leave a long audio review no room for its JSON.
DEFAULT_OUTPUT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 32768

#: Statuses on which :meth:`Ear.review` moves on to the next served model:
#: retired, rate limited, saturated, or too slow to answer this time.
_TRY_NEXT = (404, 429, 503, "timeout")

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
    #: ``outputTokenLimit`` per model, as the listing reported it.
    limits: dict = field(default_factory=dict)

    # -- plumbing ---------------------------------------------------------

    def _scrub(self, text: str) -> str:
        if not self.key:
            return text
        # the key as sent, and as repr() would quote it inside a message
        for form in {self.key, repr(self.key)[1:-1], repr(self.key.encode())[2:-1]}:
            if form:
                text = text.replace(form, "<key>")
        return text

    def _request(self, method: str, url: str, body: bytes | None = None,
                 headers: dict | None = None, retries: int = 2) -> bytes:
        return self._send(method, url, body, headers, retries)[0]

    def _send(self, method: str, url: str, body: bytes | None = None,
              headers: dict | None = None, retries: int = 2) -> tuple[bytes, object]:
        """One call to the API: ``(body, response headers)``.

        Every way the call can fail comes out as a :class:`GeminiError` with
        the key scrubbed from it. That is not only ``HTTPError`` and
        ``URLError``: urllib wraps neither a failure while waiting for the
        response (a timeout, a dropped connection) nor one while reading
        it (a short read), and ``http.client`` refuses a header value with
        a newline in it by raising a ``ValueError`` whose message quotes
        the whole value -- the key.
        """
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=body, method=method)
            req.add_header("x-goog-api-key", self.key)
            for k, v in (headers or {}).items():
                req.add_header(k, v)
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                    return resp.read(), resp.headers
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf8", "replace")[:400]
                except Exception:           # noqa: BLE001 - the body is optional
                    detail = ""
                err = GeminiError(
                    self._scrub(f"HTTP {exc.code} from {_path_of(url)}: {detail}"))
                err.status = exc.code
                # 429 and 503 are "ask again", not "this will never work"
                if exc.code in (429, 503) and attempt < retries:
                    time.sleep(RETRY_BASE * 2 ** attempt)
                    continue
                raise err from None
            except (OSError, http.client.HTTPException) as exc:
                reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
                if isinstance(reason, TimeoutError):
                    # a request that ran out the clock will do so again;
                    # the caller may try a lighter model instead
                    err = GeminiError(f"the Gemini API did not answer {_path_of(url)} "
                                      f"within {TIMEOUT} s")
                    err.status = "timeout"
                    raise err from None
                if (isinstance(reason, (ConnectionError, http.client.IncompleteRead))
                        and attempt < retries):
                    time.sleep(RETRY_BASE * 2 ** attempt)
                    continue
                raise GeminiError(self._scrub(
                    f"cannot reach the Gemini API: {type(reason).__name__}: {reason}")) from None
            except ValueError:
                # never the message: it quotes the offending header value
                raise GeminiError(
                    "the request to the Gemini API could not be built: a header "
                    "value was refused. Check that the key is one line with "
                    "nothing else in it.") from None
        raise GeminiError("unreachable")

    def _json(self, method: str, url: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        head = {"Content-Type": "application/json"} if payload is not None else {}
        raw = self._request(method, url, body, head)
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise GeminiError("the API returned something that was not JSON") from None
        if not isinstance(data, dict):
            raise GeminiError("the API returned JSON that was not an object")
        return data

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
                try:
                    self.limits[name] = int(m.get("outputTokenLimit") or 0)
                except (TypeError, ValueError):
                    pass
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
        try:
            size = path.stat().st_size
            data = path.read_bytes()
        except OSError as exc:
            raise GeminiError(f"cannot read {path}: {exc.strerror or exc}") from None
        try:
            _raw, head = self._send(
                "POST", f"{API_ROOT}/upload/v1beta/files",
                json.dumps({"file": {"display_name": label}}).encode(), {
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(size),
                    "X-Goog-Upload-Header-Content-Type": mime,
                    "Content-Type": "application/json",
                })
        except GeminiError as exc:
            raise GeminiError(f"upload start failed: {exc}") from None
        session_url = head.get("X-Goog-Upload-URL") if head is not None else None
        if not session_url:
            raise GeminiError("the API did not return an upload URL")
        raw = self._request("POST", session_url, data, {
            "Content-Length": str(size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        })
        try:
            info = json.loads(raw).get("file", {})
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            raise GeminiError("the upload finished with a reply that was not JSON") from None
        if not isinstance(info, dict):
            info = {}
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
        try:
            total = sum(p.stat().st_size for p in sending)
        except OSError as exc:
            raise GeminiError(f"cannot read {exc.filename or 'an audio file'}: "
                              f"{exc.strerror or exc}") from None
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
        try:
            duration = float((session or {}).get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        models = self.candidates()
        last: GeminiError | None = None
        for model in models[:4]:
            if on_step:
                on_step(f"gemini {model}")
            payload = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0.2,
                                     "responseMimeType": "application/json",
                                     "maxOutputTokens": self._output_cap(model)},
            }
            try:
                data = self._json(
                    "POST", f"{API_ROOT}/v1beta/models/{model}:generateContent", payload)
            except GeminiError as exc:
                # 404 (retired), 503 (saturated) and a timeout all mean "try
                # the next one"; anything else is ours to fix, so surface it.
                if getattr(exc, "status", None) in _TRY_NEXT:
                    last = exc
                    continue
                raise
            text, finish = _first_text(data), _finish_reason(data)
            stopped = f" (it stopped: {finish})" if finish and finish != "STOP" else ""
            if not text:
                last = GeminiError(f"{model} returned no text{stopped}")
                continue
            # A reply cut off at the output cap, or otherwise unreadable, is
            # this model's failure, not the ear's: the next one may finish.
            try:
                out = repair_json(text, duration)
            except GeminiError as exc:
                last = GeminiError(f"{model}: {exc}{stopped}")
                continue
            out["model"] = model
            return out
        raise last or GeminiError("no served model answered")

    def _output_cap(self, model: str) -> int:
        """Room for the reply: what the model allows, up to a sane ceiling."""
        limit = self.limits.get(model, 0)
        return min(limit, MAX_OUTPUT_TOKENS) if limit > 0 else DEFAULT_OUTPUT_TOKENS


def _path_of(url: str) -> str:
    return url.split("?")[0].split("generativelanguage.googleapis.com")[-1] or url


def _first_text(data: dict) -> str:
    for cand in data.get("candidates", []) or []:
        if not isinstance(cand, dict):
            continue
        for part in (cand.get("content") or {}).get("parts", []) or []:
            # a thinking model can hand back its thought summary as a part
            if isinstance(part, dict) and "text" in part and not part.get("thought"):
                return str(part["text"])
    return ""


def _finish_reason(data: dict) -> str:
    """Why the first candidate stopped: STOP, MAX_TOKENS, SAFETY, ..."""
    for cand in data.get("candidates", []) or []:
        if isinstance(cand, dict):
            return str(cand.get("finishReason") or "")
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


def repair_json(text: str, duration: float = 0.0) -> dict:
    """Parse the model's reply, forgiving the usual wrappers, then validate.

    Returns a dict with the documented shape. Anything the model got wrong
    -- a fence, a category it invented, a severity of 7, a score as a
    string, a time of ``Infinity`` or past the end -- is corrected here
    rather than raising, because a slightly malformed critique is still
    worth reading. ``duration`` (seconds, 0 when unknown) bounds the times.
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
            t = float(item.get("time_sec", 0))
        except (TypeError, ValueError, OverflowError):
            t = 0.0
        # json reads Infinity, NaN and 1e400; none of them is a moment
        t = max(0.0, t) if math.isfinite(t) else 0.0
        if duration > 0:
            t = min(t, duration)
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
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    # The model answered with a fraction. Strictly below 1: the scale asked
    # for is integers 0-100, and 1 on it is "unlistenable", not "perfect".
    if 0.0 < v < 1.0:
        v *= 100.0
    return round(min(100.0, max(0.0, v)), 1)


# ---------------------------------------------------------------------------
# key handling and feedback markers
# ---------------------------------------------------------------------------

#: What a key may contain: printable ASCII, no spaces. Anything else would
#: be refused as a header value -- by an exception that quotes it.
_KEY_SHAPE = re.compile(r"[\x21-\x7e]+")


def _clean_key(raw: str, where: str) -> str:
    """The key out of ``raw``: its first line that is not blank or a comment.

    A key file picks up a second line easily -- a note about when it was
    rotated, an old key kept below the new one. Only the first line is the
    key; the rest never goes near a request. What is left is checked, and a
    refusal never repeats the value.
    """
    lines = [ln.strip() for ln in raw.lstrip("\ufeff").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        return ""
    key = lines[0]
    if not _KEY_SHAPE.fullmatch(key):
        raise GeminiError(
            f"the Gemini key in {where} contains spaces or characters a key "
            "cannot have. Put the key alone on the first line.")
    return key


def load_key() -> str:
    """The API key, from the environment or the key file. Never logged."""
    key = _clean_key(os.environ.get("GEMINI_API_KEY") or "", "GEMINI_API_KEY")
    if key:
        return key
    if KEY_FILE.exists():
        try:
            text = KEY_FILE.read_text(encoding="utf8")
        except (OSError, UnicodeDecodeError) as exc:
            raise GeminiError(f"cannot read {KEY_FILE} ({type(exc).__name__})") from None
        key = _clean_key(text, str(KEY_FILE))
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

    The file holds Neel's own markers, ratings and votes, and the app
    server writes it too. So this takes the same lock the server does
    (:func:`fourfloor.feedback.locked`), writes through the same atomic
    tmp-and-rename with a ``.bak`` of the old document, and refuses --
    with a :class:`GeminiError`, leaving the file exactly as it was --
    when the existing file cannot be read as a feedback document. Starting
    a blank one over it would erase everything a person put there.
    """
    from .. import feedback as fb

    target = feedback_path(render)
    if target is None:
        return None
    beat = float((session or {}).get("beat_duration_sec") or 0.0)
    per_bar = int((session or {}).get("beats_per_bar") or 4)
    first = float((session or {}).get("first_downbeat_sec") or 0.0)
    bar_len = beat * per_bar
    remix_dir = target.parent

    with fb.locked(remix_dir):
        doc: dict = {"markers": []}
        if target.exists():
            try:
                loaded = json.loads(target.read_text(encoding="utf8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                raise GeminiError(
                    f"{target} could not be read ({type(exc).__name__}), so "
                    "Gemini's markers were not added; the file was left as it "
                    "was") from None
            if not isinstance(loaded, dict) or not isinstance(
                    loaded.get("markers", []), list):
                raise GeminiError(
                    f"{target} is not a feedback document, so Gemini's markers "
                    "were not added; the file was left as it was")
            doc = loaded
        markers = [m for m in doc.get("markers", [])
                   if not (isinstance(m, dict) and m.get("author") == "gemini")]
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for issue in result.get("issues", []):
            try:
                t = float(issue.get("time_sec", 0.0))
            except (TypeError, ValueError, OverflowError):
                t = 0.0
            t = max(0.0, t) if math.isfinite(t) else 0.0
            markers.append({
                "time": round(t, 2),
                "bar": int(max(0.0, t - first) // bar_len) if bar_len > 0 else None,
                "category": issue.get("category", "artifact"),
                "note": issue.get("note", ""),
                "author": "gemini",
                "ts": stamp,
            })
        doc["markers"] = sorted(markers, key=lambda m: _marker_time(m))
        fb.write_raw(remix_dir, doc)
    return target


def _marker_time(marker) -> float:
    """Sort key that survives a marker someone else wrote oddly."""
    try:
        t = float(marker.get("time") or 0.0) if isinstance(marker, dict) else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return t if math.isfinite(t) else 0.0


def listen(render: Path, ref: Path | None = None, source: Path | None = None,
           session: dict | None = None, model: str | None = None,
           on_step=None) -> dict:
    """One full Gemini pass. Raises :class:`GeminiError`; callers fall back."""
    return Ear(load_key(), model).review(render, ref, source, session, on_step)
