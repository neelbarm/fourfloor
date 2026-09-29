"""--producer: whatever the model sends back, the remix still validates."""

from __future__ import annotations

import json

from fourfloor import arrange, producer


def test_a_start_off_the_source_bar_grid_is_snapped_onto_it(fixture_analysis,
                                                           monkeypatch) -> None:
    """arrange.validate refuses a slot that starts between the source's bar
    lines (two target bars each when the source is half-time), so an LLM start
    of 13.7 s used to fail the whole remix instead of falling back."""
    p = arrange.plan(fixture_analysis, 128.0, 2.0, length=120.0)
    assert p.source_bar_grain == 2
    grain = p.source_bar_grain * p.bar_dur
    odd = [{"index": s.index, "source_start": round((i + 0.37) * p.bar_dur * 3, 3)}
           for i, s in enumerate(p.slots)]
    reply = {"content": [{"type": "text",
                          "text": json.dumps({"note": "n", "slots": odd})}]}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    monkeypatch.setattr(producer, "_request", lambda payload, key: reply)
    warnings: list[str] = []
    p, note = producer.apply_producer_plan(fixture_analysis, p, warnings)
    assert note == "n" and not warnings
    assert arrange.validate(p) == []
    for s in p.slots:
        assert abs(s.source_start / grain - round(s.source_start / grain)) < 1e-9
