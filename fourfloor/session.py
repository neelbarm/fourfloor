"""The ``*.session.json`` DJ handoff file.

This is the contract between fourfloor and a DJ application such as MixPilot:
exact output tempo, the offset of the first downbeat, key and Camelot code, cue
points for every structural moment, a per-bar energy curve and the semitone
shift that was applied. A host that reads this file can beat-match and cue the
remix without re-analysing the audio.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .arrange import Plan
from .audio import rms_db

SCHEMA_VERSION = 2

REQUIRED_KEYS = (
    "schema", "generator", "file", "bpm", "first_downbeat_sec", "beats_per_bar",
    "key", "camelot", "semitone_shift", "duration", "bars", "cues", "sections",
    "energy_per_bar", "loudness", "source",
)


def _cue_kind(kind: str, seen: dict[str, int]) -> str:
    """Name cues the way a DJ reads them: drop 1, drop 2, breakdown…"""
    seen[kind] = seen.get(kind, 0) + 1
    if kind in ("drop", "build") and seen[kind] > 1:
        return f"{kind} {seen[kind]}"
    if kind == "drop":
        return "drop 1"
    if kind == "build":
        return "build 1"
    return kind


def build(plan: Plan, audio: np.ndarray, sr: int, out_path: str | Path,
          key_name: str, camelot: str, semitones: int, source: dict,
          tempo_plan: dict) -> dict:
    """Assemble the session dictionary for a rendered remix."""
    bar_dur = plan.bar_dur
    n_bars = plan.total_bars
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio

    energy = []
    for b in range(n_bars):
        a, z = int(b * bar_dur * sr), int((b + 1) * bar_dur * sr)
        seg = mono[a:min(z, len(mono))]
        energy.append(float(np.sqrt(np.mean(np.square(seg)))) if len(seg) else 0.0)
    emax = max(energy) if energy else 1.0
    energy = [round(e / max(emax, 1e-9), 4) for e in energy]

    cues, seen = [], {}
    for slot in plan.slots:
        cues.append({
            "name": _cue_kind(slot.kind, seen),
            "bar": slot.start_bar,
            "time": round(slot.start_bar * bar_dur, 4),
            "kind": slot.kind,
        })
    cues.append({"name": "end", "bar": n_bars, "time": round(n_bars * bar_dur, 4),
                 "kind": "end"})

    return {
        "schema": SCHEMA_VERSION,
        "generator": "fourfloor",
        "file": Path(out_path).name,
        # the remix is synthesised on a perfectly regular grid, so the tempo is
        # exact and the first downbeat is sample zero -- a DJ tool can trust both
        "bpm": round(plan.target_bpm, 2),
        "first_downbeat_sec": 0.0,
        "beats_per_bar": 4,
        "beat_duration_sec": round(bar_dur / 4.0, 6),
        "key": key_name,
        "camelot": camelot,
        "semitone_shift": int(semitones),
        "duration": round(n_bars * bar_dur, 3),
        "bars": n_bars,
        "cues": cues,
        "sections": [
            {
                "kind": s.kind, "start_bar": s.start_bar, "bars": s.bars,
                "start": round(s.start_bar * bar_dur, 4),
                "end": round(s.end_bar * bar_dur, 4),
                "source_label": s.source_label,
                "source_start": round(s.source_start, 4),
                "note": s.note,
            }
            for s in plan.slots
        ],
        "energy_per_bar": energy,
        "loudness": {
            "peak_db": round(float(20 * np.log10(max(float(np.max(np.abs(audio))), 1e-6))), 2),
            "rms_db": round(rms_db(audio), 2),
        },
        "tempo_plan": tempo_plan,
        "source": source,
    }


def validate(session: dict) -> list[str]:
    """Check a session dict against the schema. Returns a list of problems."""
    problems = [f"missing key: {k}" for k in REQUIRED_KEYS if k not in session]
    if problems:
        return problems
    if session["schema"] != SCHEMA_VERSION:
        problems.append(f"schema {session['schema']} != {SCHEMA_VERSION}")
    if not (60.0 <= session["bpm"] <= 200.0):
        problems.append(f"bpm out of range: {session['bpm']}")
    if len(session["energy_per_bar"]) != session["bars"]:
        problems.append("energy_per_bar length does not match bars")
    if not session["cues"]:
        problems.append("no cues")
    last = -1.0
    for c in session["cues"]:
        if c["time"] < last:
            problems.append(f"cue {c['name']} is out of order")
        last = c["time"]
    return problems


def write(session: dict, path: str | Path) -> Path:
    """Write a session file, pretty-printed for humans reading a diff."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf8")
    return path
