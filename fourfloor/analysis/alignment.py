"""Does the rendered remix actually sit on its own grid?

This module is the objective answer to "some of it is off beat". A remix is
built on a fixed target grid: ``bpm`` beats per minute with bar one starting at
``first_downbeat_sec``. Everything the engine synthesises is placed on that grid
by construction, so the kit is trivially in time; the part that can drift is the
*source* -- the warped song laid over the grid slot by slot. If the warp
anchoring, the loop length or the slot's source offset is wrong, the vocal and
the instrumental sit a few tens of milliseconds -- or a whole beat -- away from
the kick, and the remix sounds drunk.

Three independent measurements, because each catches a different failure:

* **Onset phase error.** Pick onsets out of the layer, measure the distance from
  each to the nearest point of the grid's 16th-note lattice, and report the
  median and 90th percentile in milliseconds. Real music plays on 16ths, so the
  lattice -- not the beat -- is the right ruler: a correctly warped source lands
  on it, a mis-anchored one does not. This catches slow drift and small
  constant offsets.
* **Comb phase.** Cross-correlate the whole onset envelope against a comb of the
  beat grid, sweeping the comb over +/- half a beat. The peak's position is the
  layer's global phase; a peak near half a beat means the source is playing the
  offbeats where the kick expects the downbeats.
* **Bar phase (downbeat parity).** Score the four possible bar phases by the
  low-band and onset energy landing on their beat one. Phase 0 means the
  source's bar one is the grid's bar one; anything else means the arrangement is
  a beat or two out even though every onset is individually on the lattice.

Plus a structural check that has nothing to do with phase: no two arrangement
slots may render source audio at the same time. ``span_overlap`` takes the
per-slot spans the engine reports and returns the worst concurrency it finds.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from ..dsp import filters as FL
from . import features as F

#: Acceptance thresholds. These are the numbers a render has to beat.
ON_GRID_MS = 20.0
MAX_MEDIAN_MS = 12.0
MAX_P90_MS = 25.0
MIN_ON_GRID = 0.85

#: How much louder the offbeat has to be than the beat before a layer counts as
#: playing half a beat out. House offbeat hats routinely beat the kick by a
#: fifth in an onset envelope without anything being wrong.
HALF_BEAT_RATIO = 1.35

FINE_HOP = F.ATTACK_HOP


def _mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def onset_envelope(x: np.ndarray, sr: int, hop: int = FINE_HOP
                   ) -> tuple[np.ndarray, float]:
    """The attack envelope this module measures with, and its frame rate."""
    return F.attack_envelope(_mono(x), sr, hop=hop)


def onset_times(x: np.ndarray, sr: int, hop: int = FINE_HOP,
                floor: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Onset times (seconds) and strengths, with sub-frame peak interpolation.

    ``floor`` is a fraction of the envelope's 95th percentile; peaks below it
    are reverb tails and noise, not events a listener hears as "a hit".
    """
    env, fps = onset_envelope(x, sr, hop=hop)
    if env.size < 4 or env.max() <= 0:
        return np.zeros(0), np.zeros(0)
    times = F.attack_times(len(env), fps)
    ref = float(np.percentile(env[env > 0], 95)) if np.any(env > 0) else 0.0
    height = max(floor * ref, 1e-6)
    peaks, _ = sps.find_peaks(env, height=height, distance=max(1, int(0.05 * fps)))
    peaks = peaks[(peaks > 0) & (peaks < len(env) - 1)]
    if not len(peaks):
        return np.zeros(0), np.zeros(0)
    a, b, c = env[peaks - 1], env[peaks], env[peaks + 1]
    den = a - 2 * b + c
    shift = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / np.where(den == 0, 1e-12, den), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    return times[peaks] + shift / fps, b


def grid_times(bpm: float, first_downbeat: float, duration: float,
               division: int = 4) -> np.ndarray:
    """Every ``division``-th of a beat of the target grid, covering ``duration``.

    ``division=1`` gives beats, ``4`` gives 16th notes. The lattice is extended
    backwards past ``first_downbeat`` so material in the pickup bar is measured
    against the same ruler -- and it is extended in whole *bars*, so ``grid[0]``
    is always a beat one and ``grid[k]`` is beat ``(k // division) % 4`` of a
    bar whatever the division. ``bar_phase`` relies on that.
    """
    beat = 60.0 / max(bpm, 1e-6)
    step = beat / max(division, 1)
    bar = 4.0 * beat
    first = first_downbeat - bar * np.ceil(first_downbeat / bar)
    n = int(np.floor((duration - first) / step)) + 1
    return first + step * np.arange(max(n, 1))


def phase_errors(onsets: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Signed seconds from each onset to the nearest grid point."""
    if not len(onsets) or not len(grid):
        return np.zeros(0)
    idx = np.clip(np.searchsorted(grid, onsets), 1, len(grid) - 1)
    lo, hi = grid[idx - 1], grid[idx]
    pick = np.where(np.abs(onsets - lo) <= np.abs(onsets - hi), lo, hi)
    return onsets - pick


def comb_offset(x: np.ndarray, sr: int, bpm: float, first_downbeat: float,
                hop: int = FINE_HOP, steps: int = 121) -> tuple[float, float, float]:
    """Global phase of a layer against the beat comb.

    Returns ``(offset_seconds, sharpness, half_beat_ratio)``. The offset is
    where a comb of beat impulses best explains the onset envelope, searched
    over +/- a quarter beat; sharpness is how much better the winner is than the
    mean of the sweep, which says whether the answer means anything at all (a
    pad has no sharpness).

    The half-beat question is asked separately, as a ratio rather than by
    widening the search, because widening it does not work. House offbeat hats
    are sharper than the kicks they sit between, so a comb allowed to roam half
    a beat happily lands on the offbeat of a loop that is perfectly on the grid
    -- two of the three kits built from real records did exactly that. A ratio
    with a threshold asks the real question: is the offbeat *so much* stronger
    that the layer is genuinely playing on the "and"?
    """
    env, fps = onset_envelope(x, sr, hop=hop)
    if env.size < 8 or env.max() <= 0:
        return 0.0, 0.0, 0.0
    kick, _ = F.kick_envelope(_mono(x), sr, hop=hop)
    n = min(len(env), len(kick)) if len(kick) else len(env)
    env, kick = env[:n], (kick[:n] if len(kick) else np.zeros(n))
    beat = 60.0 / max(bpm, 1e-6)
    duration = n / fps
    beats = grid_times(bpm, first_downbeat, duration, division=1)
    beats = beats[(beats >= 0) & (beats < duration)]
    if len(beats) < 4:
        return 0.0, 0.0, 0.0
    frames = F.attack_times(n, fps)

    def comb_of(sig: np.ndarray, o: float) -> float:
        return float(np.interp(beats + o, frames, sig, left=0.0, right=0.0).sum())

    # Ask the kick where the beat is whenever the kick has an opinion. Summed
    # over the whole spectrum a house open hat beats the kick it sits between
    # five to one, so a comb reading the attack envelope calls a perfectly
    # gridded loop half a beat out -- which is what two of three kits built
    # from real records did. The kick envelope's profile over a beat peaks
    # twenty times above its floor on a four-on-the-floor record, and is flat
    # on something with no kick, which is exactly when to ignore it.
    sweep = np.linspace(-beat / 2.0, beat / 2.0, 49)
    kc = np.array([comb_of(kick, o) for o in sweep])
    use_kick = kc.max() > 0 and kc.max() / max(kc.min(), 1e-9) >= 3.0
    sig = kick / max(float(kick.max()), 1e-9) if use_kick else env

    def comb(o: float) -> float:
        return comb_of(sig, o)

    offsets = np.linspace(-beat / 4.0, beat / 4.0, steps)
    scores = np.array([comb(o) for o in offsets])
    best = int(np.argmax(scores))
    mean = float(scores.mean())
    sharp = (scores[best] - mean) / max(scores[best], 1e-9)
    on_grid = max(comb(0.0), 1e-9)
    half = max(comb(beat / 2.0), comb(-beat / 2.0))
    return float(offsets[best]), float(sharp), float(half / on_grid)


def bar_phase(x: np.ndarray, sr: int, bpm: float, first_downbeat: float,
              hop: int = FINE_HOP) -> tuple[int, float]:
    """Which of the four beat offsets carries this layer's bar one.

    Returns ``(phase, margin)``. ``phase == 0`` means the layer's downbeat is the
    grid's downbeat; anything else means the song's bar one was placed on the
    remix's beat two, three or four, which is the kind of wrong a listener hears
    immediately even though every individual hit is on the lattice.

    The evidence is :func:`tempo.bar_phase_strength`, the same reading the beat
    tracker uses to find downbeats in the first place, so the gate and the
    analysis cannot disagree about what a bar one looks like.
    """
    from .tempo import bar_phase_strength, find_downbeats

    duration = len(x) / float(sr)
    beats = grid_times(bpm, first_downbeat, duration, division=1)
    lead = int(np.sum(beats < 0))
    beats = beats[int(np.ceil(lead / 4.0)) * 4:]      # keep beats[0] a bar one
    beats = beats[beats < duration]
    if len(beats) < 8:
        return 0, 0.0
    strength = bar_phase_strength(_mono(x), sr, beats)
    return find_downbeats(beats, strength)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Quantile of ``values`` weighted by ``weights``."""
    if not len(values):
        return 0.0
    order = np.argsort(values)
    v, w = values[order], np.maximum(weights[order], 1e-12)
    cum = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * float(np.sum(w)), cum, v))


def _layer_report(x: np.ndarray, sr: int, bpm: float, first_downbeat: float,
                  division: int = 4) -> dict:
    """Every phase measurement for one layer.

    The headline numbers are weighted by onset strength, because that is what
    "off beat" means to a listener: a snare landing 40 ms late is the complaint,
    and the fourth hat of a 32nd-note roll landing 40 ms from the nearest 16th
    is the music. On the drums of *Body* the unweighted 90th percentile is 44 ms
    and the weighted one is 12 ms, and the 12 ms is the honest description --
    the 44 ms is a hat roll being measured against a lattice it was never
    played on. Unweighted figures are reported alongside so the difference is
    always visible.
    """
    duration = len(x) / float(sr)
    onsets, strength = onset_times(x, sr)
    grid = grid_times(bpm, first_downbeat, duration, division=division)
    err = phase_errors(onsets, grid) * 1000.0
    absr = np.abs(err)
    w = np.asarray(strength, dtype=float)
    offset, sharp, half_ratio = comb_offset(x, sr, bpm, first_downbeat)
    phase, margin = bar_phase(x, sr, bpm, first_downbeat)
    beat_ms = 60_000.0 / max(bpm, 1e-6)
    off_ms = offset * 1000.0
    if half_ratio > HALF_BEAT_RATIO:
        where = "half-beat"
    elif abs(off_ms) < 0.10 * beat_ms:
        where = "grid"
    else:
        where = "offset"
    return {
        "onsets": int(len(onsets)),
        "median_ms": _weighted_quantile(absr, w, 0.5) if len(absr) else 0.0,
        "p90_ms": _weighted_quantile(absr, w, 0.9) if len(absr) else 0.0,
        "within_20ms": (float(np.sum(w[absr <= ON_GRID_MS]) / max(np.sum(w), 1e-12))
                        if len(absr) else 1.0),
        "median_ms_unweighted": float(np.median(absr)) if len(absr) else 0.0,
        "p90_ms_unweighted": float(np.percentile(absr, 90)) if len(absr) else 0.0,
        "within_20ms_unweighted": float(np.mean(absr <= ON_GRID_MS)) if len(absr) else 1.0,
        "mean_signed_ms": float(np.mean(err)) if len(err) else 0.0,
        "comb_offset_ms": off_ms,
        "comb_sharpness": sharp,
        "half_beat_ratio": half_ratio,
        "beat_alignment": where,
        "bar_phase": phase,
        "bar_phase_margin": margin,
        "grid_division": division,
    }


def span_overlap(spans: list[tuple[int, int]], n: int) -> dict:
    """Worst simultaneous-span count over a set of half-open sample ranges.

    The engine reports one span per arrangement slot. Two slots whose source
    audio is live at the same instant is the "everything was overlapping" bug:
    the count at every boundary must be exactly one while the arrangement is
    running (a crossfade is allowed, but it is handled inside a single slot's
    buffer, not by rendering two slots on top of each other).
    """
    if not spans:
        return {"max_concurrent": 0, "overlaps": [], "gaps": []}
    edges = sorted({e for s in spans for e in s} | {0, n})
    worst, overlaps, gaps = 0, [], []
    for a, b in zip(edges, edges[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2.0
        live = sum(1 for s, e in spans if s <= mid < e)
        worst = max(worst, live)
        if live > 1:
            overlaps.append({"start": a, "end": b, "count": live})
        elif live == 0 and 0 <= a < n:
            gaps.append({"start": a, "end": b})
    return {"max_concurrent": worst, "overlaps": overlaps[:8], "gaps": gaps[:8]}


def alignment_report(rendered_audio: np.ndarray, sr: int, bpm: float,
                     first_downbeat_sec: float = 0.0,
                     source_stem: np.ndarray | None = None,
                     kit_layer: np.ndarray | None = None,
                     spans: list[tuple[int, int]] | None = None,
                     division: int = 4, kit_is_sampled: bool = False) -> dict:
    """Measure whether a rendered remix sits on its own grid.

    ``rendered_audio`` is the finished mix. ``source_stem`` should be the source
    layer rendered on its own -- ideally the source's drums, which have the
    clearest onsets -- because in the full mix the kit's onsets swamp the
    source's and every number comes back flattering. ``kit_layer`` is the
    synthesised drum bus, whose numbers should be ~0 by construction and are
    therefore the control: if the kit reads badly the grid parameters passed in
    are wrong, not the render.

    Returns a dict of per-layer reports plus ``ok`` and a list of ``problems``
    naming, in plain words, whichever threshold failed.
    """
    out: dict = {
        "bpm": float(bpm),
        "first_downbeat_sec": float(first_downbeat_sec),
        "duration": len(rendered_audio) / float(sr),
        "mix": _layer_report(rendered_audio, sr, bpm, first_downbeat_sec, division),
    }
    if source_stem is not None:
        out["source"] = _layer_report(source_stem, sr, bpm, first_downbeat_sec, division)
    if kit_layer is not None:
        out["kit"] = _layer_report(kit_layer, sr, bpm, first_downbeat_sec, division)
    if spans is not None:
        out["spans"] = span_overlap(spans, len(rendered_audio))

    judged = out.get("source", out["mix"])
    problems: list[str] = []
    if judged["median_ms"] > MAX_MEDIAN_MS:
        problems.append(f"median phase error {judged['median_ms']:.1f} ms "
                        f"(limit {MAX_MEDIAN_MS:.0f})")
    if judged["p90_ms"] > MAX_P90_MS:
        problems.append(f"p90 phase error {judged['p90_ms']:.1f} ms "
                        f"(limit {MAX_P90_MS:.0f})")
    if judged["within_20ms"] < MIN_ON_GRID:
        problems.append(f"only {judged['within_20ms']:.1%} of source onsets within "
                        f"+/-{ON_GRID_MS:.0f} ms of the grid "
                        f"(limit {MIN_ON_GRID:.0%})")
    if judged["beat_alignment"] != "grid" and judged["comb_sharpness"] > 0.05:
        problems.append(f"the source sits {judged['comb_offset_ms']:+.0f} ms off the beat "
                        f"({judged['beat_alignment']})")
    if judged["bar_phase"] != 0 and judged["bar_phase_margin"] > 0.08:
        problems.append(f"the source's bar one lands on the grid's beat "
                        f"{judged['bar_phase'] + 1}")
    if "kit" in out:
        limit = 18.0 if kit_is_sampled else 6.0
        if out["kit"]["median_ms"] > limit:
            problems.append(
                f"the sampled loop is {out['kit']['median_ms']:.1f} ms off the grid "
                f"it was laid on (limit {limit:.0f})" if kit_is_sampled else
                f"the kit itself reads {out['kit']['median_ms']:.1f} ms off its own "
                "grid, so the grid parameters are wrong")
    if "spans" in out and out["spans"]["max_concurrent"] > 1:
        problems.append(f"{out['spans']['max_concurrent']} arrangement slots render "
                        "source audio at the same time")
    out["kit_is_sampled"] = bool(kit_is_sampled)
    out["problems"] = problems
    out["ok"] = not problems
    return out


# ---------------------------------------------------------------------------
# two rhythms at once
# ---------------------------------------------------------------------------

#: How close an onset has to be to a beat or an offbeat eighth to count as
#: playing the house grid rather than across it.
HOUSE_TOL = 0.025


def house_positions(bpm: float, first_downbeat: float, duration: float) -> np.ndarray:
    """Beats and offbeat eighths: where a house record puts things.

    Not the sixteenth lattice the phase gate measures against. That question is
    "is this in time"; this one is "is this playing the same rhythm as the kit",
    and a part can be perfectly in time on sixteenths while playing a pattern
    that fights four-on-the-floor all the way through.
    """
    return grid_times(bpm, first_downbeat, duration, division=2)


def _rhythm_onsets(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Onsets of a layer, measured with a detector that suits what it is.

    A bass layer has to be measured in the band it occupies. The attack
    envelope uses a 1024-sample window, which is one and a third cycles of a
    55 Hz note: the band energy ripples at the note's own frequency and the
    flux finds an "onset" on every ripple. A held bass note comes back as
    twenty-five onsets a bar that way, when it is one. The sub-band envelope
    looks at 30-120 Hz over a 46 ms window and reads the same note as one.
    """
    mono = _mono(np.asarray(x, dtype=np.float32))
    full = float(np.sqrt(np.mean(np.square(mono.astype(np.float64)))))
    if full <= 1e-9:
        return np.zeros(0), np.zeros(0)
    # "Is this a bass part" has to be asked as a ratio between two bands, not
    # as a share of total energy: music is 1/f, so a full-range bed with its
    # bottom rolled off at 105 Hz still holds most of its energy below 200 and
    # would be measured with a detector that only hears kicks.
    low = FL.apply(mono, "lowpass", sr, 200.0, q=0.707, order=2)
    high = FL.apply(mono, "highpass", sr, 200.0, q=0.707, order=2)
    low_db = 20.0 * np.log10(max(float(np.sqrt(np.mean(low ** 2))), 1e-9))
    high_db = 20.0 * np.log10(max(float(np.sqrt(np.mean(high ** 2))), 1e-9))
    if low_db - high_db < 12.0:
        return onset_times(x, sr)
    kick, fps = F.kick_envelope(mono, sr, lo=30.0, hi=160.0)
    if not len(kick) or kick.max() <= 0:
        return np.zeros(0), np.zeros(0)
    times = F.attack_times(len(kick), fps)
    ref = float(np.percentile(kick[kick > 0], 92)) if np.any(kick > 0) else 0.0
    peaks, _ = sps.find_peaks(kick, height=max(0.18 * ref, 1e-6),
                              distance=max(1, int(0.07 * fps)))
    return (times[peaks], kick[peaks]) if len(peaks) else (np.zeros(0), np.zeros(0))


def rhythm_report(x: np.ndarray, sr: int, bpm: float, first_downbeat: float = 0.0,
                  tol: float = HOUSE_TOL) -> dict:
    """How busy a layer is, and how much of it plays across the house grid.

    Returns onset density per bar -- both a count and an energy-weighted one --
    and the share of onset energy landing away from every beat and offbeat
    eighth. A pad reads as nearly no density. A sustained house bass reads as
    dense and entirely on the grid. A trap 808 reads as dense and substantially
    off it, and that is the measurement that says a layer is a second rhythm
    rather than part of the first.
    """
    duration = len(x) / float(sr)
    bars = max(duration / (4.0 * 60.0 / max(bpm, 1e-6)), 1e-9)
    onsets, strength = _rhythm_onsets(x, sr)
    rms = float(np.sqrt(np.mean(np.square(_mono(np.asarray(x, dtype=np.float64))))))
    out = {
        "rms_db": 20.0 * np.log10(max(rms, 1e-9)),
        "onsets": int(len(onsets)),
        "onsets_per_bar": float(len(onsets) / bars),
        "off_house": 0.0,
        "off_house_per_bar": 0.0,
        "strength_per_bar": 0.0,
    }
    if not len(onsets):
        return out
    grid = house_positions(bpm, first_downbeat, duration)
    err = np.abs(phase_errors(onsets, grid))
    w = np.asarray(strength, dtype=float)
    off = err > tol
    total = max(float(w.sum()), 1e-12)
    out["off_house"] = float(w[off].sum() / total)
    out["off_house_per_bar"] = float(np.sum(off) / bars)
    out["strength_per_bar"] = float(w.sum() / bars)
    return out


def overlap_report(layers: dict, sr: int, bpm: float, first_downbeat: float = 0.0,
                   sections: list | None = None, floor_db: float = -45.0) -> dict:
    """Which layers are carrying a rhythm, whole track and section by section.

    ``sections`` is ``[(label, start_seconds, end_seconds), ...]``; without it
    the whole track is one section. ``floor_db`` is the level below which a
    layer is treated as silent, because a layer 45 dB down is not what anybody
    is hearing two rhythms of.
    """
    out: dict = {"bpm": float(bpm), "layers": {}}
    for name, buf in layers.items():
        if buf is None:
            continue
        whole = rhythm_report(buf, sr, bpm, first_downbeat)
        entry = {"whole": whole, "silent": bool(whole["rms_db"] < floor_db),
                 "sections": {}}
        for label, a, b in (sections or []):
            seg = buf[int(a * sr):int(b * sr)]
            if len(seg) < sr // 4:
                continue
            entry["sections"][label] = rhythm_report(seg, sr, bpm, first_downbeat)
        out["layers"][name] = entry
    return out


def bass_collision(bass: np.ndarray, sr: int, bpm: float,
                   first_downbeat: float = 0.0, tol: float = 0.04) -> dict:
    """Does this bass part hit where a four-on-the-floor kick is going to hit?

    The complaint a listener makes is "two rhythms at once", and in a house
    remix of a hip-hop record it is almost always the same two: the kit's kick
    on every beat, and the source's 808 playing its own pattern in the same two
    octaves. An 808 is a melodic *percussion* instrument -- it is a kick with a
    pitch -- so under four-on-the-floor it reads as a second kick drum rather
    than as a bassline.

    What separates that from a bassline that belongs there is not how busy it
    is; the house records measured here run six to eight sub attacks a bar, the
    same as the trap ones. It is *where* the attacks land. A house bass dodges
    the kick, putting 5-25% of its sub-band attack energy on the beat. Don
    Toliver's *Body* puts 54% of it there, right on top of where the kick goes.

    Returns the attacks per bar, the share of attack energy on the beat and off
    the eighth-note grid, how much of the time the sub band is actually
    sounding, and the level.
    """
    mono = _mono(np.asarray(bass, dtype=np.float32))
    duration = len(mono) / float(sr)
    bars = max(duration / (4.0 * 60.0 / max(bpm, 1e-6)), 1e-9)
    rms = float(np.sqrt(np.mean(np.square(mono.astype(np.float64)))))
    out = {"per_bar": 0.0, "on_beat": 0.0, "off_eighth": 0.0, "sustain": 0.0,
           "rms_db": 20.0 * np.log10(max(rms, 1e-9))}
    kick, fps = F.kick_envelope(mono, sr, lo=30.0, hi=120.0)
    if not len(kick) or kick.max() <= 0:
        return out
    low = F.band_energy(mono, sr, 30.0, 120.0, hop=512)
    out["sustain"] = float(np.mean(low > 0.25 * max(float(low.max()), 1e-9)))
    times = F.attack_times(len(kick), fps)
    ref = float(np.percentile(kick[kick > 0], 92)) if np.any(kick > 0) else 0.0
    peaks, _ = sps.find_peaks(kick, height=max(0.18 * ref, 1e-6),
                              distance=max(1, int(0.07 * fps)))
    if not len(peaks):
        return out
    hits, w = times[peaks], kick[peaks]
    total = max(float(w.sum()), 1e-12)
    beats = grid_times(bpm, first_downbeat, duration, division=1)
    eighths = grid_times(bpm, first_downbeat, duration, division=2)
    on_beat = np.abs(phase_errors(hits, beats)) <= tol
    on_eighth = np.abs(phase_errors(hits, eighths)) <= tol
    out["per_bar"] = float(len(hits) / bars)
    out["on_beat"] = float(w[on_beat].sum() / total)
    out["off_eighth"] = float(w[~on_eighth].sum() / total)
    return out
