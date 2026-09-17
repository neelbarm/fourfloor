"""Bass and chord-stab synthesis driven by the source's detected chords.

The voicing, the note pattern and the levels here are measured from six
commercial house remixes (Demucs ``bass``/``drums`` stems, 32-bar drop window
each). Those measurements live in :data:`REFERENCE` and every default in this
module is set from them, so a change of taste is a change of one number rather
than a change of code.

The four things that make a bass line read as "house" rather than as "a synth
playing the root":

* **Nothing starts on the kick.** In all six references the four downbeat
  sixteenths carry about a quarter of the onsets the off-sixteenths do. The
  notes *sustain* across the beat -- per-sixteenth activity is close to uniform
  -- but none of them begins there, so the two attacks never collide.
* **It is a sixteenth-note roll, not offbeat eighths.** Four of the six run
  11-13 notes a bar: three sixteenths per beat, none of them on the beat.
* **It does not bounce octaves.** Measured octave-jump rate is 0.000 at the
  median and 1.8% at the worst; the whole line lives in about a 6.7-semitone
  span around a 50 Hz note. This module's predecessor lifted an octave every
  bar, which is the single most audible thing it did that no record does.
* **It is level with the kick, and mostly sub.** 40-120 Hz RMS within a dB of
  the kick's -- not tucked under it -- with energy above 200 Hz about 17 dB
  down: a clean sine fundamental plus a filtered saw layer for definition, not
  a bright saw with a little sub beneath it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsp import filters as FL


# ---------------------------------------------------------------------------
# Measured reference targets
# ---------------------------------------------------------------------------

#: What six commercial house remixes do, measured from their Demucs ``bass``
#: stems over the highest-energy 32-bar window of each (125-133 BPM). Values are
#: the median across the six with the observed spread beside them. Everything
#: else in this module is tuned to land inside these ranges; :func:`bass_report`
#: measures the synthesised bass the same way so the two can be compared.
REFERENCE = {
    "n_tracks": 6,
    "bpm_range": (125.0, 132.8),
    # 40-120 Hz RMS of the bass stem minus the same band of the drum stem.
    "bass_minus_kick_db": -0.8,
    "bass_minus_kick_db_range": (-4.9, 2.8),
    # Full band, the bass sits just under the whole drum bus.
    "bass_minus_drums_fullband_db": -1.1,
    # RMS above 200 Hz relative to the whole bass stem: saw-like vs sine-like.
    "above_200_rel_db": -16.8,
    "above_200_rel_db_range": (-23.6, -13.0),
    # Depth of the bass-envelope dip after each kick, and how long it takes to
    # climb back. Measuring this on a Demucs bass stem is the one number the
    # separation actively fights: the stem leaks kick, which *shallows* the
    # apparent dip, while a bar where the bass simply rests deepens it. The raw
    # corpus median is -28.8 dB here and -32 dB in the reference spec; the
    # trustworthy subset -- kicks where the bass is demonstrably sustaining
    # through -- gives -23 dB, and past about -20 dB it stops pumping and starts
    # gating. So the raw figure is recorded for honesty and the *target* below
    # is what anything should actually be built or tested against.
    "sidechain_dip_db_raw": -28.8,
    "sidechain_dip_db": -15.0,
    "sidechain_dip_db_range": (-18.0, -12.0),
    "sidechain_recovery_s": 0.085,
    "sidechain_recovery_s_range": (0.080, 0.110),
    # Note fundamentals: where the bass actually sits on a club system. The
    # whole line lives inside a 6.7-semitone span around a 50 Hz median note.
    "f0_median_hz": 50.0,
    "f0_range_hz": (33.6, 72.7),
    "span_semitones": 6.7,
    # The kick's own fundamental. The bass shares this band with it.
    "kick_f0_hz": 43.0,
    # Pattern.
    "onsets_per_bar": 12.4,
    "onsets_per_bar_range": (4.5, 13.2),
    "offbeat_ratio": 0.83,
    "note_len_sixteenths": 0.83,
    # Pitch behaviour: it follows the chord roots (the most common pitch class
    # is only a third of the notes) but literal octave jumps are rare.
    "distinct_pitch_classes": 5,
    "top_pitch_class_share": 0.34,
    # Octave bounces are effectively absent: median 0.000, worst case 0.018.
    "octave_jump_share": 0.0,
    "octave_jump_share_max": 0.018,
    "same_note_share": 0.47,
}

#: The note the whole line is built around: the references' median fundamental,
#: roughly G#1. Octaves are chosen by nearest approach to this, which is what
#: keeps the line inside one narrow register instead of bouncing octaves.
BASS_CENTRE_HZ = 50.0

#: How wide a span the line is allowed, in semitones, around
#: :data:`BASS_CENTRE_HZ`. The references stay inside 6.7; a couple of semitones
#: of slack absorbs progressions whose roots sit awkwardly against the centre.
BASS_SPAN_SEMITONES = 9.0

#: Hard window a fundamental must land in. Below it the note is felt but not
#: heard and it only steals headroom from the kick; above it, it stops reading
#: as bass. Nearest-to-centre octave choice cannot leave this window.
TARGET_F0_HZ = (36.0, 72.0)

#: The kick's own fundamental, measured across the corpus. The bass has to share
#: 40-120 Hz with it, so a root that lands here is the one worth watching in the
#: mix -- it is the same note as the kick, not a different instrument.
KICK_FUNDAMENTAL_HZ = 43.0

#: The band the kick and the bass share and fight over.
SUB_BAND = (40.0, 120.0)

#: **The number the mix stage wants.** Intended level of the bass bus inside
#: :data:`SUB_BAND`, in dB relative to the kick measured in the same band. The
#: references sit within a dB of their kicks; this is that target.
SUB_LEVEL_VS_KICK_DB = -0.8

#: How far either side of :data:`SUB_LEVEL_VS_KICK_DB` still counts as right.
#: Outside this the references disagree with each other, so it is a real miss.
SUB_LEVEL_TOLERANCE_DB = 2.5

#: Not "quieter than the kick": *level with* it. Spelling the pair out because
#: every instinct says to tuck a bass under a kick, and the records do not.
SUB_LEVEL_RANGE_DB = (-4.9, 2.8)

#: Per-sixteenth *onset* weight, normalised from the six references' pooled
#: onset histograms. Steps 0, 4, 8 and 12 -- the kick -- are the near-empty
#: ones; that hole is the whole point. It is a hole in note *starts*, not in
#: sound: see ``length`` in :data:`STYLES`.
STEP_WEIGHTS: tuple[float, ...] = (
    0.22, 0.87, 0.86, 0.47,
    0.21, 1.00, 0.78, 0.69,
    0.31, 0.95, 0.74, 0.63,
    0.19, 0.91, 0.87, 0.71,
)

#: Which sixteenths each style plays, how long a note lasts in sixteenths, and
#: how often it reaches for something other than the root.
#:
#: ``length`` is slightly over 1 on the rolling styles on purpose. Onsets avoid
#: the kick, but the *sound* does not stop there: the references show roughly
#: uniform per-sixteenth activity, so the note starting on step 3 has to sustain
#: across step 4. A line that actually went silent on the beat would measure a
#: 30 dB hole after every kick, which is gating, not pumping.
#:
#: ``octave_share`` is zero for the default. Octave bounces are the single most
#: recognisable thing fourfloor's old bass did that no reference does: the
#: corpus median octave-jump rate is 0.000 and the worst case 1.8%.
STYLES: dict[str, dict] = {
    # 12 notes a bar: three sixteenths per beat, none of them on the kick.
    "house":   {"min_weight": 0.40, "length": 1.06, "octave_share": 0.0,
                "fifth_share": 0.14, "cutoff": 760.0},
    # The same grid, brighter and shorter: more attack per note.
    "rolling": {"min_weight": 0.40, "length": 1.02, "octave_share": 0.02,
                "fifth_share": 0.18, "cutoff": 980.0},
    # Offbeat eighths only: steps 2, 6, 10, 14. One reference does this.
    "offbeat": {"steps": (2, 6, 10, 14), "length": 1.85, "octave_share": 0.0,
                "fifth_share": 0.10, "cutoff": 900.0},
    # Sparse and sustained, like the slowest of the references.
    "sparse":  {"steps": (2, 5, 8, 14), "length": 2.80, "octave_share": 0.0,
                "fifth_share": 0.12, "cutoff": 620.0},
}

DEFAULT_STYLE = "house"

#: Peak of a single :func:`bass_note`. Every note is normalised to it, so the
#: line does not get louder just because a root happens to sit where the filter
#: and the saw agree. It is set so that the engine's existing per-note loop, at
#: ``engine.BASS_GAIN``, already lands on :data:`SUB_LEVEL_VS_KICK_DB`.
NOTE_PEAK = 0.61


# ---------------------------------------------------------------------------
# Pitch helpers
# ---------------------------------------------------------------------------

def note_hz(pitch_class: int, octave: int = 2) -> float:
    """Frequency of a pitch class in a given octave (C2 = 65.41 Hz)."""
    midi = 12 * (octave + 1) + pitch_class
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def pick_bass_octave(root_pc: int, lo: float = TARGET_F0_HZ[0],
                     hi: float = TARGET_F0_HZ[1],
                     centre: float = BASS_CENTRE_HZ) -> int:
    """Octave whose fundamental sits closest to ``centre``, clamped to ``lo``-``hi``.

    Choosing by nearest approach to one centre note, rather than by "the first
    octave above 41 Hz", is what collapses the line into a single register: two
    roots a semitone apart can no longer land an octave apart, which is the
    mechanism behind the octave bouncing the references never do.
    """
    best, best_cost = 2, float("inf")
    for octave in (0, 1, 2, 3):
        f = note_hz(root_pc, octave)
        if not (lo <= f <= hi):
            continue
        cost = abs(np.log2(f / centre))
        if cost < best_cost:
            best, best_cost = octave, cost
    if best_cost < float("inf"):
        return best
    # Nothing inside the window (only possible if a caller narrowed it): fall
    # back to nearest approach and let the clamp be advisory.
    return min((0, 1, 2, 3), key=lambda o: abs(np.log2(note_hz(root_pc, o) / centre)))


def in_target_range(freq: float, lo: float = TARGET_F0_HZ[0],
                    hi: float = TARGET_F0_HZ[1]) -> bool:
    """Is this fundamental inside the measured bass window?"""
    return lo <= freq <= hi


def fit_to_register(pitch_class: int, reference_hz: float,
                    lo: float = TARGET_F0_HZ[0], hi: float = TARGET_F0_HZ[1],
                    max_semitones: float = 5.5) -> int | None:
    """Octave putting ``pitch_class`` nearest ``reference_hz``, or ``None``.

    ``None`` means the note cannot be placed near the reference without leaving
    the window -- the fifth of G, say, is either 36.7 Hz or 73.4 Hz and neither
    belongs next to a 49 Hz root. The caller is expected to drop the ornament
    and play the root instead, which is what keeps the line inside one register.

    The 5.5-semitone default admits the fifth *below* the root, which is how a
    bass plays a fifth, and refuses the fifth above, which would climb out of
    the register. Anything tighter than 5 rejects every fifth there is.
    """
    best, best_cost = None, float("inf")
    for octave in (0, 1, 2, 3):
        f = note_hz(pitch_class, octave)
        if not (lo <= f <= hi):
            continue
        cost = abs(12.0 * np.log2(f / reference_hz))
        if cost < best_cost:
            best, best_cost = octave, cost
    return best if best_cost <= max_semitones else None


def span_semitones(freqs) -> float:
    """Width of a set of fundamentals, in semitones. The references stay under 6.7."""
    f = [float(v) for v in freqs if v and v > 0]
    if len(f) < 2:
        return 0.0
    return float(12.0 * np.log2(max(f) / min(f)))


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------

def _phase(freqs: np.ndarray, sr: int) -> np.ndarray:
    """Running phase for a (possibly gliding) frequency curve, in radians."""
    return (2.0 * np.pi * np.cumsum(freqs) / sr).astype(np.float64)


def _saw_from_phase(phase: np.ndarray, base_hz: float, sr: int,
                    max_h: int = 24) -> np.ndarray:
    """Additive sawtooth from a phase curve, harmonics kept under Nyquist."""
    out = np.zeros(len(phase), dtype=np.float32)
    h = 1
    while base_hz * h < sr * 0.45 and h <= max_h:
        out += (np.sin(phase * h) / h).astype(np.float32)
        h += 1
    return out * (2.0 / np.pi)


def _saw(freq: float, n: int, sr: int, detune: float = 0.0) -> np.ndarray:
    """Band-limited-ish sawtooth by additive synthesis up to Nyquist."""
    f = freq * (1.0 + detune)
    ph = _phase(np.full(n, f, dtype=np.float64), sr)
    return _saw_from_phase(ph, f, sr)


def _amp_envelope(n: int, sr: int, seconds: float, sustain: float = 0.55,
                  attack: float = 0.004) -> np.ndarray:
    """Plucked AD(S)R: fast attack, quick decay to a sustain, clean release.

    A pure exponential decay reads as a "boop"; the short decay into a held
    sustain and a release that reaches true zero is what makes a sixteenth-note
    roll sound gated and tight rather than smeared.
    """
    a = min(max(2, int(attack * sr)), max(2, n // 4))
    rel = min(max(int(0.10 * n), int(0.008 * sr)), max(2, n - a - 1))
    dec = max(1, n - a - rel)

    env = np.empty(n, dtype=np.float32)
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32) ** 0.6
    tau = max(seconds * 0.16, 0.012)
    td = np.arange(dec) / sr
    env[a:a + dec] = (sustain + (1.0 - sustain)
                      * np.exp(-td / tau)).astype(np.float32)
    tail = float(env[a + dec - 1]) if dec else 1.0
    env[a + dec:] = np.linspace(tail, 0.0, n - a - dec, dtype=np.float32) ** 1.5
    return env


# ---------------------------------------------------------------------------
# One note
# ---------------------------------------------------------------------------

def bass_note(sr: int, freq: float, seconds: float, cutoff: float = 760.0,
              sub_mix: float = 0.92, drive: float = 1.35,
              glide_from: float | None = None, glide_time: float = 0.045,
              sustain: float = 0.55, velocity: float = 1.0,
              harmonics: float = 0.55, sub_h2: float = 0.35,
              sub_h3: float = 0.12) -> np.ndarray:
    """One house bass note: a clean sine sub plus a filtered, driven saw layer.

    The sub carries the note -- it is a sine at the *fundamental*, not an octave
    below it, so the energy lands in the 40-80 Hz the references occupy instead
    of in inaudible rumble. The saw layer is high-passed off the fundamental so
    it adds definition without muddying the sub, then swept by a per-note filter
    envelope that opens on the attack and closes over the note: that movement is
    the "plucked" forward motion of a house bass.

    ``harmonics`` sets how saw-like the result is, and ``sub_h2``/``sub_h3`` how
    much 2nd and 3rd the sub itself carries. Together the defaults put energy
    above 200 Hz about 15 dB under the note, where the references sit.

    The 2nd and 3rd matter more than they look. A 50 Hz note's harmonics land at
    100 and 150 Hz, which is the band a laptop or a phone can actually
    reproduce -- a pure sine sub is inaudible on both. It is also what keeps the
    bus from being so lopsided that the master's band-match EQ has to pull the
    whole sub down to reach the house balance, which costs the finished track
    half a dB of RMS.

    ``glide_from`` slides the pitch up or down from another frequency over
    ``glide_time`` seconds, used on chord changes.
    """
    n = max(8, int(seconds * sr))
    t = np.arange(n) / sr

    if glide_from and glide_from > 0 and glide_time > 0:
        g = min(max(2, int(glide_time * sr)), n)
        curve = np.concatenate([
            np.exp(np.linspace(np.log(glide_from), np.log(freq), g)),
            np.full(n - g, freq),
        ])
    else:
        curve = np.full(n, float(freq))
    ph = _phase(curve, sr)

    # -- sub: the note itself, plus a touch of 2nd so it survives small speakers
    sub = (np.sin(ph) + sub_h2 * np.sin(2.0 * ph)
           + sub_h3 * np.sin(3.0 * ph)).astype(np.float32) * sub_mix

    # -- saw layer: definition only, kept off the fundamental
    saw = (0.6 * _saw_from_phase(ph, freq, sr)
           + 0.4 * _saw_from_phase(ph * 1.006, freq, sr))
    saw = FL.apply(saw, "highpass", sr, max(freq * 1.6, 90.0), q=0.707, order=2)
    fenv = cutoff * (0.30 + 0.70 * np.exp(-t / max(seconds * 0.35, 0.03)))
    saw = FL.sweep(saw, "lowpass", sr, np.maximum(fenv, freq * 2.0), q=1.3, order=2)
    saw = np.tanh(drive * saw) / np.tanh(drive)

    amp = _amp_envelope(n, sr, seconds, sustain=sustain)
    out = (sub + harmonics * saw) * amp

    # Trim sub-bass rumble the glide or the envelope may have introduced, then
    # normalise so every note arrives at the same level whatever its pitch.
    out = FL.apply(out, "highpass", sr, 34.0, q=0.707, order=2)
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = out / peak
    return (out * NOTE_PEAK * float(velocity)).astype(np.float32)


def stab(sr: int, root_pc: int, minor: bool, seconds: float, octave: int = 4,
         cutoff: float = 2600.0) -> np.ndarray:
    """Short filtered triad stab for offbeat chords in the drops."""
    n = max(8, int(seconds * sr))
    t = np.arange(n) / sr
    intervals = (0, 3, 7, 10) if minor else (0, 4, 7, 11)
    sig = np.zeros(n, dtype=np.float32)
    for k, iv in enumerate(intervals):
        f = note_hz((root_pc + iv) % 12, octave + (1 if (root_pc + iv) >= 12 else 0))
        sig += _saw(f, n, sr, detune=0.004 * (k - 1.5)) * (0.9 ** k)
    amp = np.exp(-t / max(seconds * 0.3, 0.02)).astype(np.float32)
    a = max(2, int(0.004 * sr))
    amp[:a] *= np.linspace(0.0, 1.0, a)
    sig = FL.apply(sig, "lowpass", sr, cutoff, q=0.9, order=2)
    sig = FL.apply(sig, "highpass", sr, 220.0)
    out = sig * amp
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.3).astype(np.float32) if peak > 0 else out


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BassEvent:
    """One note on the sixteenth grid, ready to render."""

    bar: int
    step: int                      # 0-15, sixteenth within the bar
    length_steps: float            # note length, in sixteenths
    pitch_class: int
    octave: int
    velocity: float = 1.0
    role: str = "root"             # root | fifth | octave | passing
    slide_from_hz: float | None = None

    @property
    def freq(self) -> float:
        return note_hz(self.pitch_class, self.octave)

    def start_time(self, bar_dur: float) -> float:
        return (self.bar + self.step / 16.0) * bar_dur

    def seconds(self, bar_dur: float) -> float:
        return self.length_steps / 16.0 * bar_dur


def normalise_chords(chords) -> list[tuple[int, bool]]:
    """Accept ``[(root_pc, is_minor)]`` or ``key.chords_per_bar`` dicts."""
    out: list[tuple[int, bool]] = []
    for c in chords:
        if isinstance(c, dict):
            root = int(c.get("root_pc", 0)) % 12
            minor = str(c.get("quality", "min")).startswith("min")
        else:
            root = int(c[0]) % 12
            minor = bool(c[1]) if len(c) > 1 else True
        out.append((root, minor))
    return out


def _style(style: str) -> dict:
    return STYLES.get(style, STYLES[DEFAULT_STYLE])


def style_steps(style: str = DEFAULT_STYLE) -> tuple[int, ...]:
    """Which sixteenths a style plays."""
    spec = _style(style)
    if "steps" in spec:
        return tuple(spec["steps"])
    mw = spec["min_weight"]
    return tuple(i for i, w in enumerate(STEP_WEIGHTS) if w >= mw)


def bass_pattern(chords, bars: int | None = None, style: str = DEFAULT_STYLE,
                 seed: int = 0, phrase: int = 8,
                 lo_hz: float = TARGET_F0_HZ[0],
                 hi_hz: float = TARGET_F0_HZ[1]) -> list[BassEvent]:
    """Note events for ``bars`` bars over the per-bar chord roots.

    Notes come off the chord root, with a fifth or an octave lift sprinkled in
    at the rates the references show (octave jumps are rare -- 2% of notes --
    which is why the old "lift every bar on step 5" read as a tic). The first
    note of a bar whose root has changed slides in from the previous root, and
    the last sixteenth of a phrase takes a chromatic passing note into the next
    root. Deterministic for a given ``seed``.
    """
    chord_list = normalise_chords(chords) or [(0, True)]
    if bars is None:
        bars = len(chord_list)
    spec = _style(style)
    steps = style_steps(style)
    rng = np.random.default_rng(seed)

    events: list[BassEvent] = []
    prev_root: int | None = None
    prev_freq: float | None = None

    for bar in range(bars):
        root, _minor = chord_list[bar % len(chord_list)]
        octave = pick_bass_octave(root, lo_hz, hi_hz)
        fifth = (root + 7) % 12
        last_of_phrase = phrase > 0 and (bar % phrase) == phrase - 1
        changed = prev_root is not None and root != prev_root

        for i, step in enumerate(steps):
            pc, oct_, role = root, octave, "root"
            r = float(rng.random())
            # Octave lifts and fifths are seasoning, not the pattern.
            if (spec["octave_share"] > 0 and r < spec["octave_share"]
                    and step not in (0, 4, 8, 12)):
                oct_, role = octave + 1, "octave"
            elif r < spec["octave_share"] + spec["fifth_share"]:
                # The fifth is placed against the root, not against the global
                # centre: inheriting the root's octave outright would drop the
                # C under an F root to 32 Hz, and centring it independently
                # drops the D under a G root to 36.7 Hz. Either way the line
                # stops being one register. If it will not fit close by, the
                # ornament is dropped and the root plays again.
                fit = fit_to_register(fifth, note_hz(root, octave),
                                      max(lo_hz, 40.0), min(hi_hz, 70.0))
                if fit is not None:
                    pc, oct_, role = fifth, fit, "fifth"

            # Chromatic approach into the next phrase's root, last note only.
            # It is placed against the note it is approaching, not against the
            # global centre: a leading tone an octave away from its target is
            # not a leading tone.
            if last_of_phrase and i == len(steps) - 1 and bars > bar + 1:
                nxt = chord_list[(bar + 1) % len(chord_list)][0]
                nxt_hz = note_hz(nxt, pick_bass_octave(nxt, lo_hz, hi_hz))
                cand = (nxt - 1) % 12 if rng.random() < 0.5 else (nxt + 1) % 12
                fit = fit_to_register(cand, nxt_hz, lo_hz, hi_hz, max_semitones=2.0)
                if fit is not None:
                    pc, oct_, role = cand, fit, "passing"

            slide = prev_freq if (changed and i == 0) else None
            vel = 0.82 + 0.18 * STEP_WEIGHTS[step % 16]
            ev = BassEvent(bar=bar, step=step, length_steps=float(spec["length"]),
                           pitch_class=pc, octave=oct_, velocity=round(vel, 4),
                           role=role, slide_from_hz=slide)
            events.append(ev)
            prev_freq = ev.freq

        prev_root = root
    return events


#: 40-120 Hz RMS that :func:`render_bassline` normalises to, before the engine
#: applies its own bus gain. Normalising on the sub band rather than on the peak
#: is what makes the level predictable: a bass line's crest factor swings with
#: the pattern, so two styles normalised to the same peak arrive at the mix
#: several dB apart, while two normalised to the same sub RMS do not.
#:
#: The value is set so that with ``engine.BASS_GAIN`` (0.55) against
#: ``engine.DRUM_GAIN`` (0.72) on a four-to-the-floor ``drums.kick``, the line
#: lands on :data:`SUB_LEVEL_VS_KICK_DB`. If either bus gain moves, use
#: :func:`match_kick_level` instead of re-tuning this.
BUS_SUB_RMS_DB = -21.4

#: Ceiling the finished line is soft-clipped to. It should barely engage.
BUS_PEAK = 0.95

#: Gentle saturation across the finished line, the way a bass-bus limiter works:
#: it buys RMS at the same peak, and the harmonics it adds are what move the
#: above-200 Hz content onto the reference's -17 dB.
BUS_DRIVE = 1.9


def render_bassline(sr: int, bar_dur: float, chords, bars: int | None = None,
                    style: str = DEFAULT_STYLE, seed: int = 0, gain: float = 1.0,
                    stereo: bool = True, phrase: int = 8,
                    cutoff: float | None = None,
                    events: list[BassEvent] | None = None,
                    peak: float = BUS_PEAK, drive: float = BUS_DRIVE,
                    sub_rms_db: float | None = BUS_SUB_RMS_DB) -> np.ndarray:
    """Render a whole bass line: pattern, then a note per event.

    This is the entry point the engine should call instead of looping eighths
    itself -- the pattern, the octave choice, the slides and the passing notes
    all belong with the voice that has to play them.

    The finished line is saturated and normalised so its 40-120 Hz RMS is
    ``sub_rms_db``; see :data:`SUB_LEVEL_VS_KICK_DB` for where it is then
    supposed to sit against the kick, and :func:`match_kick_level` for locking
    it there regardless of the bus gains.
    """
    if events is None:
        events = bass_pattern(chords, bars=bars, style=style, seed=seed,
                              phrase=phrase)
    if bars is None:
        bars = (max((e.bar for e in events), default=0) + 1)
    cut = float(cutoff if cutoff is not None else _style(style)["cutoff"])

    n = max(1, int(round(bars * bar_dur * sr)) + int(0.5 * sr))
    out = np.zeros(n, dtype=np.float32)
    for ev in events:
        note = bass_note(sr, ev.freq, ev.seconds(bar_dur), cutoff=cut,
                         velocity=ev.velocity, glide_from=ev.slide_from_hz)
        a = int(round(ev.start_time(bar_dur) * sr))
        b = min(n, a + len(note))
        if a >= n or b <= a:
            continue
        out[a:b] += note[: b - a]

    if drive and drive > 0:
        p = float(np.max(np.abs(out)))
        if p > 0:
            out = (np.tanh(drive * (out / p)) / np.tanh(drive)).astype(np.float32) * p

    if sub_rms_db is not None:
        out = set_sub_level(out, sr, float(sub_rms_db))
    if peak and peak > 0:
        p = float(np.max(np.abs(out)))
        if p > peak:      # safety only; normalising on the sub band rarely trips it
            out = (np.tanh((out / p) * 2.0) / np.tanh(2.0) * peak).astype(np.float32)

    out = np.clip(out * float(gain), -1.0, 1.0).astype(np.float32)
    return np.column_stack([out, out]).astype(np.float32) if stereo else out


def sub_level_db(x: np.ndarray, sr: int) -> float:
    """RMS of ``x`` inside :data:`SUB_BAND`, in dBFS."""
    return _db(_rms(_band(_mono(np.asarray(x)), sr, *SUB_BAND)))


def set_sub_level(x: np.ndarray, sr: int, target_db: float = BUS_SUB_RMS_DB,
                  max_gain_db: float = 24.0) -> np.ndarray:
    """Scale ``x`` so its 40-120 Hz RMS is ``target_db``.

    A silent or near-silent buffer is returned untouched rather than amplified
    into noise, and the correction is capped so a nearly-empty breakdown bar
    cannot explode.
    """
    x = np.asarray(x, dtype=np.float32)
    cur = sub_level_db(x, sr)
    if not np.isfinite(cur) or cur <= -80.0:
        return x
    g = float(np.clip(target_db - cur, -max_gain_db, max_gain_db))
    return (x * (10.0 ** (g / 20.0))).astype(np.float32)


def match_kick_level(bass: np.ndarray, kick: np.ndarray, sr: int,
                     target_db: float = SUB_LEVEL_VS_KICK_DB,
                     max_gain_db: float = 12.0,
                     max_cut_db: float = 24.0) -> tuple[np.ndarray, float]:
    """Set the bass so it sits ``target_db`` against the kick inside 40-120 Hz.

    This is the level contract in executable form, and it is what the mix stage
    should call: it holds whatever the bus gains are and whatever the drum
    designer does to the kick, which a hard-coded gain does not. Returns the
    scaled bass and the gain applied, in dB.

    The two limits are deliberately asymmetric. Turning a bass *down* is always
    safe, so ``max_cut_db`` is generous; turning one *up* is how a near-silent
    breakdown bar gets dragged to kick level along with its noise floor, so
    ``max_gain_db`` is tight.
    """
    kb = sub_level_db(kick, sr)
    bb = sub_level_db(bass, sr)
    if not (np.isfinite(kb) and np.isfinite(bb)) or bb <= -80.0 or kb <= -80.0:
        return np.asarray(bass, dtype=np.float32), 0.0
    g = float(np.clip((kb + target_db) - bb, -abs(max_cut_db), abs(max_gain_db)))
    return (np.asarray(bass, dtype=np.float32) * 10.0 ** (g / 20.0)).astype(np.float32), g


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1).astype(np.float32) if x.ndim == 2 else x.astype(np.float32)


def _band(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """Band-limit with cascaded biquads (no SOS design at tiny Fc/Fs ratios)."""
    y = x
    if lo > 1.0:
        y = FL.apply(y, "highpass", sr, lo, q=0.707, order=3)
    if hi < sr * 0.49:
        y = FL.apply(y, "lowpass", sr, hi, q=0.707, order=3)
    return y


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64))) + 1e-20))


def _db(v: float) -> float:
    return float(20.0 * np.log10(max(float(v), 1e-9)))


def _envelope(x: np.ndarray, sr: int, hz: float = 400.0) -> tuple[np.ndarray, float]:
    """Peak-per-block amplitude envelope and its frame rate."""
    hop = max(1, int(sr / hz))
    n = (len(x) // hop) * hop
    if n < hop:
        return np.abs(x).astype(np.float32), float(sr)
    e = np.abs(x[:n]).reshape(-1, hop).max(axis=1).astype(np.float32)
    return e, sr / hop


def bass_report(x: np.ndarray, sr: int, kick_times=None,
                bar_dur: float | None = None,
                kick: np.ndarray | None = None) -> dict:
    """Measure a synthesised bass line the way the references were measured.

    Returns the same quantities as :data:`REFERENCE` so the two can be compared
    directly, plus the health checks a render has to pass: no NaN, no clipping.
    ``kick_times`` (seconds) enables the sidechain dip and recovery numbers, and
    a ``kick`` buffer enables the level-versus-kick number the mix stage wants.
    """
    mono = _mono(np.asarray(x))
    finite = np.isfinite(mono)
    report: dict = {
        "samples": int(len(mono)),
        "seconds": round(len(mono) / sr, 3),
        "has_nan": bool(not finite.all()),
        "peak": round(float(np.max(np.abs(mono[finite]))) if finite.any() else 0.0, 5),
        "clipping": bool(finite.any() and float(np.max(np.abs(mono[finite]))) > 1.0),
        "rms_db": round(_db(_rms(mono[finite] if finite.any() else mono)), 2),
    }
    if not finite.all() or len(mono) < 64:
        return report

    sub = _band(mono, sr, *SUB_BAND)
    hi = _band(mono, sr, 200.0, sr * 0.49)
    full = _rms(mono)
    report["sub_40_120_db"] = round(_db(_rms(sub)), 2)
    report["above_200_rel_db"] = round(_db(_rms(hi)) - _db(full), 2)
    report["above_200_in_reference_range"] = bool(
        REFERENCE["above_200_rel_db_range"][0] - 2.0
        <= report["above_200_rel_db"]
        <= REFERENCE["above_200_rel_db_range"][1] + 2.0)

    if kick is not None:
        ksub = _band(_mono(np.asarray(kick)), sr, *SUB_BAND)
        rel = _db(_rms(sub)) - _db(_rms(ksub))
        report["bass_minus_kick_db"] = round(rel, 2)
        report["level_target_db"] = SUB_LEVEL_VS_KICK_DB
        report["level_error_db"] = round(rel - SUB_LEVEL_VS_KICK_DB, 2)
        report["level_on_target"] = bool(
            abs(rel - SUB_LEVEL_VS_KICK_DB) <= SUB_LEVEL_TOLERANCE_DB)

    env, fps = _envelope(_band(mono, sr, 30.0, 300.0), sr)

    # -- fundamentals, from the sixteenth-note onsets we can find -------------
    d = np.diff(env, prepend=env[0])
    d[d < 0] = 0.0
    thr = max(float(np.percentile(d, 96)) * 0.35, float(env.max()) * 0.02)
    # Dedup to just under a sixteenth so one note cannot count twice.
    gap_s = (bar_dur / 16.0 * 0.85) if bar_dur and bar_dur > 0 else 0.05
    gap = max(2, int(fps * gap_s))
    idx = [i for i in range(1, len(d) - 1)
           if d[i] >= thr and d[i] >= d[i - 1] and d[i] > d[i + 1]]
    onsets = []
    for i in idx:
        if not onsets or i - onsets[-1] >= gap:
            onsets.append(i)
    times = np.asarray(onsets, dtype=float) / fps
    report["onsets"] = int(len(times))
    if bar_dur and bar_dur > 0:
        report["onsets_per_bar"] = round(len(times) / max(len(mono) / sr / bar_dur, 1e-6), 2)

    f0s = []
    for t in times:
        a = int(t * sr) + int(0.015 * sr)
        b = a + int(0.10 * sr)
        if b > len(mono):
            break
        seg = mono[a:b] * np.hanning(b - a)
        spec = np.abs(np.fft.rfft(seg, 1 << 15))
        fr = np.fft.rfftfreq(1 << 15, 1.0 / sr)
        # 34 Hz floor: below the lowest note the octave chooser can produce
        # (D1, 36.7 Hz) but above the difference tone two overlapping sixteenths
        # leave behind, which otherwise reads as a phantom 31 Hz fundamental.
        m = (fr > 34.0) & (fr < 190.0)
        if m.any():
            f0s.append(float(fr[m][int(np.argmax(spec[m]))]))
    if f0s:
        report["f0_median_hz"] = round(float(np.median(f0s)), 1)
        report["f0_min_hz"] = round(float(np.min(f0s)), 1)
        report["f0_max_hz"] = round(float(np.max(f0s)), 1)
        # Percentiles, because one mis-tracked frame should not decide whether
        # the line is in register.
        report["f0_p10_hz"] = round(float(np.percentile(f0s, 10)), 1)
        report["f0_p90_hz"] = round(float(np.percentile(f0s, 90)), 1)
        report["span_semitones"] = round(span_semitones(
            [np.percentile(f0s, 10), np.percentile(f0s, 90)]), 2)
        report["span_in_reference_range"] = bool(
            report["span_semitones"] <= BASS_SPAN_SEMITONES)
        share = float(np.mean([
            in_target_range(f, TARGET_F0_HZ[0] - 2.0, TARGET_F0_HZ[1] + 2.0)
            for f in f0s]))
        report["f0_in_target_share"] = round(share, 3)
        # Octave lifts are meant to sit above the window, so "on target" means
        # the bulk of the line is in it, not every single note.
        report["f0_in_target_range"] = bool(share >= 0.85)

    # -- sidechain behaviour -------------------------------------------------
    if kick_times is not None and len(np.asarray(kick_times)) and env.max() > 0:
        floor = float(np.percentile(env, 60)) * 0.05 + 1e-6
        ratios, recs = [], []
        for t in np.asarray(kick_times, dtype=float):
            i = int(t * fps)
            a, b = i - int(0.075 * fps), i - int(0.015 * fps)
            if a < 0 or i + int(0.45 * fps) >= len(env):
                continue
            pre = float(np.max(env[a:b]))
            if pre < float(env.max()) * 0.10:
                continue
            post = float(np.mean(env[i + int(0.006 * fps): i + int(0.035 * fps)]))
            ratios.append(max(post, floor) / pre)
            # Recovery is measured *from the bottom of the dip*, not from the
            # kick. The duck takes a few ms to bite, so searching from the kick
            # sample finds the envelope still on its way down and reports a
            # 5 ms recovery for a line that actually pumps for 100.
            tail = env[i:i + int(0.45 * fps)]
            if len(tail) < 3:
                continue
            bottom = int(np.argmin(tail[:max(2, int(0.05 * fps))]))
            over = np.flatnonzero(tail[bottom:] >= pre * 0.9)
            recs.append((bottom + float(over[0])) / fps if len(over) else 0.45)
        report["kicks_measured"] = len(ratios)
        if ratios:
            report["sidechain_dip_db"] = round(_db(float(np.median(ratios))), 2)
            report["sidechain_recovery_s"] = round(float(np.median(recs)), 3)
            lo_d, hi_d = REFERENCE["sidechain_dip_db_range"]
            # A little slack each way: the target is -12..-18 dB, but a render
            # measured over a whole drop mixes ducked notes with genuine rests.
            report["dip_in_reference_range"] = bool(
                lo_d - 5.0 <= report["sidechain_dip_db"] <= hi_d + 4.0)
            report["dip_target_db"] = REFERENCE["sidechain_dip_db"]
            lo_r, hi_r = REFERENCE["sidechain_recovery_s_range"]
            report["recovery_in_reference_range"] = bool(
                lo_r - 0.03 <= report["sidechain_recovery_s"] <= hi_r + 0.06)
    return report
