"""Optional Claude-planned arrangements.

``--producer`` sends the analysis summary to the Claude API and asks for an
arrangement plan: which source section feeds each slot, which effects to use and
a one-line creative note. The response is validated against a schema and any
failure -- no API key, no network, malformed JSON, a slot that does not line up
-- silently falls back to the deterministic rule-based planner, so the feature
can never break a remix.

Implemented with ``urllib`` from the standard library: no SDK dependency, so
installing fourfloor never pulls in an API client you may not want.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .analysis import Analysis
from .arrange import Plan

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-opus-5"
API_VERSION = "2023-06-01"
#: Server-side fallback routes around a safety refusal without us keeping a
#: model list; harmless if unavailable, since any error falls back to the
#: rule-based planner anyway.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
TIMEOUT = 90

VALID_PATTERNS = {"drop", "drop_var", "intro", "intro_full", "build", "breakdown", "outro"}

SYSTEM = (
    "You are a house music producer planning a remix arrangement. "
    "You will be given an analysis of a source track and a rule-based draft plan. "
    "Return ONLY a JSON object, no prose, with this shape:\n"
    '{"note": "<one line creative direction>", '
    '"slots": [{"index": <int>, "source_label": "<label from the source sections>", '
    '"source_start": <seconds into the source>, "drum_pattern": "<one of: '
    + ", ".join(sorted(VALID_PATTERNS)) + '>", "source_gain": <0.0-1.0>}]}\n'
    "Keep every slot index from the draft. Do not change bar counts."
)


def _request(payload: dict, api_key: str) -> dict:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf8"))


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def apply_producer_plan(analysis: Analysis, plan: Plan,
                        warnings: list[str]) -> tuple[Plan, str]:
    """Ask Claude to revise ``plan``; return ``(plan, note)``.

    On any failure the draft plan is returned unchanged and a warning is
    appended, so ``--producer`` degrades to the rule-based planner rather than
    failing the remix.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        warnings.append("--producer needs ANTHROPIC_API_KEY; used the rule-based planner")
        return plan, ""

    brief = {
        "source": {
            "bpm": round(analysis.grid.bpm, 2),
            "key": analysis.key.name,
            "camelot": analysis.key.camelot,
            "duration": round(analysis.duration, 1),
            "sections": [s.to_dict() for s in analysis.sections],
        },
        "target_bpm": round(plan.target_bpm, 2),
        "draft_slots": [
            {"index": s.index, "kind": s.kind, "bars": s.bars,
             "source_label": s.source_label, "source_start": round(s.source_start, 2),
             "drum_pattern": s.drum_pattern, "source_gain": s.source_gain}
            for s in plan.slots
        ],
    }
    payload = {
        "model": MODEL,
        "max_tokens": 8000,
        # arranging a house track from a structural summary is a light task:
        # low effort keeps the round trip to a few seconds
        "output_config": {"effort": "low"},
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
        "system": SYSTEM,
        "messages": [{"role": "user", "content": json.dumps(brief)}],
    }

    try:
        data = _request(payload, api_key)
        if data.get("stop_reason") == "refusal":
            warnings.append("--producer: the model declined the request; "
                            "used the rule-based planner")
            return plan, ""
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        revised = _extract_json(text)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError,
            json.JSONDecodeError, KeyError, TimeoutError, OSError) as exc:
        warnings.append(f"--producer failed ({type(exc).__name__}); "
                        "used the rule-based planner")
        return plan, ""

    by_index = {s.index: s for s in plan.slots}
    applied = 0
    for item in revised.get("slots", []):
        try:
            slot = by_index[int(item["index"])]
        except (KeyError, TypeError, ValueError):
            continue
        pattern = item.get("drum_pattern")
        if isinstance(pattern, str) and pattern in VALID_PATTERNS:
            slot.drum_pattern = pattern
        gain = item.get("source_gain")
        if isinstance(gain, (int, float)) and 0.0 <= float(gain) <= 1.0:
            slot.source_gain = float(gain)
        start = item.get("source_start")
        if isinstance(start, (int, float)) and 0.0 <= float(start) < plan.duration:
            slot.source_start = float(start)
        label = item.get("source_label")
        if isinstance(label, str) and label:
            slot.source_label = label[:32]
        applied += 1

    if not applied:
        warnings.append("--producer returned nothing usable; used the rule-based planner")
        return plan, ""
    note = str(revised.get("note", ""))[:200]
    return plan, note
