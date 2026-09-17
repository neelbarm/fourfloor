"""Real house drums, sampled out of a real house record.

A synthesised kit can be made convincing and this project's one is not bad, but
it will never have what a record has: the room, the compressor, the sample the
producer chose, the small unevenness that makes eight bars sound played rather
than placed. So ``fourfloor kit build`` takes a house remix you already like,
separates its drums, finds the cleanest eight bars of four-on-the-floor inside a
drop, warps those bars to a canonical 128 BPM and keeps them.

"Cleanest" is a measurement, not a guess. A window scores well when it has a
kick on every single beat at an even level, when its onsets sit on the sixteenth
lattice, when it is loud (drops are loud) and when nothing in it goes quiet. The
four are multiplied, so a window has to be good at all of them -- a loud eight
bars with a missing kick loses to a slightly quieter eight bars that has all
thirty-two.

Kits live in ``~/.fourfloor/kits/<name>/`` as ``loop.wav`` and ``meta.json``,
never in the repository: they are made of somebody else's record.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analysis import analyze
from .analysis import features as F
from .audio import decode, fit, write_wav
from .dsp import phasevocoder as PV

#: Every stored loop is warped to this, so a remix only ever has to stretch it
#: by the small amount between here and its own tempo.
CANONICAL_BPM = 128.0

#: How many bars a kit loop holds. Eight bars is the phrase a house record is
#: written in, so a loop of eight repeats without announcing itself.
KIT_BARS = 8

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def kits_home(home: str | Path | None = None) -> Path:
    """Where kits live: ``~/.fourfloor/kits`` unless told otherwise."""
    root = home or os.environ.get("FOURFLOOR_HOME") or (Path.home() / ".fourfloor")
    return Path(root) / "kits"


def slugify(text: str) -> str:
    """A filesystem-safe kit name from a track title."""
    out = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    out = re.sub(r"-+", "-", out)[:48].strip("-")
    return out or "kit"


@dataclass
class Kit:
    """One stored loop: eight bars of real house drums at 128 BPM."""

    name: str
    loop: np.ndarray            # (n, 2) float32 at CANONICAL_BPM
    sr: int
    bars: int = KIT_BARS
    source: str = ""
    source_bpm: float = 0.0
    kick_beats: list[float] = field(default_factory=list)   # seconds into the loop
    score: float = 0.0
    path: Path | None = None

    @property
    def bar_dur(self) -> float:
        return 4.0 * 60.0 / CANONICAL_BPM

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "source_bpm": round(self.source_bpm, 2),
            "canonical_bpm": CANONICAL_BPM,
            "bars": self.bars,
            "sample_rate": self.sr,
            "kick_beats": [round(t, 4) for t in self.kick_beats],
            "score": round(self.score, 4),
        }

    def at_bpm(self, target_bpm: float) -> tuple[np.ndarray, list[float]]:
        """The loop stretched to ``target_bpm``, exactly ``bars`` bars long.

        Returns the audio and the kick positions inside it, in seconds. The
        length is forced to the sample count the grid asks for rather than
        whatever the stretch happened to produce, because a loop one sample long
        is a loop that walks off the grid by the end of the track.
        """
        want_bars = 4.0 * 60.0 / target_bpm
        want = int(round(self.bars * want_bars * self.sr))
        rate = target_bpm / CANONICAL_BPM
        y = self.loop if abs(rate - 1.0) < 1e-6 else PV.time_stretch(self.loop, rate)
        scale = CANONICAL_BPM / target_bpm
        return fit(np.asarray(y, dtype=np.float32), want), [t * scale for t in self.kick_beats]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _four_on_the_floor(kick: np.ndarray, fps: float, beats: np.ndarray) -> float:
    """How reliably a kick lands on every one of these beats.

    The mean kick attack at the beats, divided by how uneven they are: a window
    with thirty-two even kicks beats a window with thirty strong ones and two
    missing, which is the whole point -- a loop with a hole in it is not a loop.

    It reads :func:`features.kick_envelope`, the *rise* of the bottom two
    octaves, and not their level. Level is mostly the bassline, which in house
    plays the offbeats, so a window whose kick sits between the beats scores
    just as well on level as one whose kick sits on them. Beat trackers do slip
    half a beat for a stretch of a track, and when *The Sweet Escape* slipped
    like that at 1:44 the level-based score picked precisely that stretch as the
    cleanest four-on-the-floor in the record.
    """
    if not len(beats) or not len(kick):
        return 0.0
    idx = np.clip(((beats + F.ATTACK_LATENCY) * fps).astype(int), 0, len(kick) - 1)
    hits = kick[idx]
    if hits.mean() <= 0:
        return 0.0
    evenness = 1.0 / (1.0 + float(np.std(hits) / max(hits.mean(), 1e-9)))
    weakest = float(np.min(hits) / max(hits.mean(), 1e-9))
    return float(hits.mean()) * evenness * (0.25 + 0.75 * min(weakest, 1.0))


def _on_lattice(env: np.ndarray, fps: float, beats: np.ndarray,
                tol: float = 0.025) -> float:
    """Share of the window's onset energy sitting on the sixteenth lattice.

    The lattice is built from the beats themselves rather than from a constant
    period, so a record that breathes is not marked down for breathing.
    """
    if not len(env) or len(beats) < 2:
        return 0.0
    a = int(max(0, beats[0] * fps))
    b = int(min(len(env), beats[-1] * fps))
    if b <= a:
        return 0.0
    total = float(env[a:b].sum())
    if total <= 0:
        return 0.0
    sixteenths = np.concatenate(
        [beats[i] + (beats[i + 1] - beats[i]) * np.arange(4) / 4.0
         for i in range(len(beats) - 1)])
    half = max(1, int(round(tol * fps)))
    on = 0.0
    for g in sixteenths:
        c = int(round((g + F.ATTACK_LATENCY) * fps))
        on += float(env[max(a, c - half):min(b, c + half + 1)].sum())
    return min(on / total, 1.0)


def find_loop(drums: np.ndarray, sr: int, beats: np.ndarray, downbeat_index: int,
              bars: int = KIT_BARS) -> tuple[int, float, list[float]]:
    """Index into ``beats`` where the best ``bars``-bar loop starts.

    Only bar lines are considered, so a loop always begins where the record
    begins a bar. Returns ``(beat_index, score, kick_offsets)``; the kick
    offsets are seconds from the start of the window, one per beat, taken from
    the low-band attack nearest each beat -- but only where there really is one,
    because a search window always has a maximum and taking it unconditionally
    invents a kick out of whatever noise was closest.
    """
    mono = drums.mean(axis=1) if drums.ndim == 2 else drums
    env, fps = F.attack_envelope(mono, sr)
    kick, _ = F.kick_envelope(mono, sr)
    n = min(len(env), len(kick))
    if n < 8 or len(beats) < bars * 4 + 1:
        return -1, -1.0, []
    env, kick = env[:n], kick[:n]
    rms = fit(F.rms_envelope(mono, hop=F.ATTACK_HOP), n)
    rms = rms / max(float(rms.max()), 1e-9)
    need = bars * 4

    best = (-1, -1.0, [])
    for j in range(int(downbeat_index) % 4, len(beats) - need, 4):
        win = beats[j:j + need + 1]
        if win[0] < 0 or win[-1] * fps >= n:
            continue
        floor_score = _four_on_the_floor(kick, fps, win[:-1])
        lattice = _on_lattice(env, fps, win)
        a, b = int(win[0] * fps), int(win[-1] * fps)
        loud = float(rms[a:b].mean())
        quietest = float(np.percentile(rms[a:b], 5)) / max(loud, 1e-9)
        score = floor_score * (0.4 + 0.6 * lattice) * loud * (0.3 + 0.7 * min(quietest, 1.0))
        if score > best[1]:
            # Presence from the kick envelope, position from the attack
            # envelope: the kick envelope is measuring a rise over a 46 ms
            # window and reads about 20 ms late, which is exactly the error a
            # reinforcement kick must not inherit.
            kicks = []
            half = max(1, int(round(0.05 * fps)))
            idx = np.clip(((win[:-1] + F.ATTACK_LATENCY) * fps).astype(int), 0, n - 1)
            ref = float(np.median(kick[idx]))
            for t in win[:-1]:
                c = int(round((t + F.ATTACK_LATENCY) * fps))
                lo, hi = max(0, c - half), min(n, c + half + 1)
                present = hi > lo and float(kick[lo:hi].max()) >= 0.35 * ref
                k = lo + int(np.argmax(env[lo:hi])) if present else c
                on_edge = k in (lo, hi - 1)
                kicks.append(float(t - win[0]) if (not present or on_edge)
                             else float(k / fps - F.ATTACK_LATENCY - win[0]))
            best = (j, float(score), kicks)
    return best


# ---------------------------------------------------------------------------
# building and loading
# ---------------------------------------------------------------------------

def build(path: str | Path, name: str | None = None, home: str | Path | None = None,
          bars: int = KIT_BARS, progress=None) -> Kit:
    """Separate a house record's drums and keep its best eight bars.

    The drums are separated from the *original* file, not from anything warped:
    a kit is a sample library, and it should carry the record's own timing into
    the 128 BPM grid rather than a timing something else already interfered
    with.
    """
    from .stems import separate_demucs

    step = progress or (lambda *_a, **_k: None)
    path = Path(path)
    step("analyse", f"reading {path.name}")
    a = analyze(path)
    if len(a.grid.beats) < bars * 4 + 1:
        raise RuntimeError(f"{path.name} is too short, or its beat grid is too "
                           f"patchy, to cut {bars} bars out of")

    step("separate", "demucs: pulling the drums out")
    stems = separate_demucs(a.clip.samples, a.sr)
    drums = stems.percussive

    step("find", "looking for the cleanest four-on-the-floor")
    beats = a.grid.beats
    j, score, kicks = find_loop(drums, a.sr, beats, a.grid.downbeat_index, bars=bars)
    if j < 0:
        raise RuntimeError(f"no {bars}-bar stretch of {path.name} held a steady "
                           "four-on-the-floor; try a different record")
    win = beats[j:j + bars * 4 + 1]
    start = float(win[0])

    # Warp through the record's own beat times rather than a constant period:
    # over fifteen seconds a tempo estimate that is 0.1% out is 15 ms of drift,
    # and a kit that drifts is a kit that argues with the grid it is laid on.
    step("warp", f"{a.grid.bpm:.2f} -> {CANONICAL_BPM:.0f} BPM")
    beat = 60.0 / a.grid.bpm
    out_times = (60.0 / CANONICAL_BPM) * np.arange(len(win))
    want = int(round(bars * 4 * 60.0 / CANONICAL_BPM * a.sr))
    head = int(start * a.sr)
    tail = int(min(len(drums), (float(win[-1]) + beat) * a.sr))
    seg = drums[head:tail]
    loop = PV.warp(seg, win - start, out_times, a.sr, want)
    loop = fit(np.asarray(loop, dtype=np.float32), want)

    peak = float(np.max(np.abs(loop)))
    if peak > 0:
        loop = (loop / peak * 0.89).astype(np.float32)
    # kick offsets are measured in source seconds; the loop runs at 128
    scale = (60.0 / CANONICAL_BPM) / max(float(np.median(np.diff(win))), 1e-9)
    kit = Kit(name=(name or slugify(path.stem)), loop=loop, sr=a.sr, bars=bars,
              source=path.name, source_bpm=a.grid.bpm, score=score,
              kick_beats=[k * scale for k in kicks])
    if not NAME_RE.match(kit.name):
        raise ValueError(f"{kit.name!r} is not a usable kit name; use letters, "
                         "digits, dashes and underscores")
    step("write", str(save(kit, home)))
    return kit


def save(kit: Kit, home: str | Path | None = None) -> Path:
    """Write a kit to ``~/.fourfloor/kits/<name>/`` and return the folder."""
    folder = kits_home(home) / kit.name
    folder.mkdir(parents=True, exist_ok=True)
    write_wav(folder / "loop.wav", kit.loop, kit.sr)
    meta = kit.to_dict()
    meta["built"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf8")
    kit.path = folder
    return folder


def load(name: str, home: str | Path | None = None) -> Kit:
    """Read one kit by name."""
    folder = kits_home(home) / name
    meta_path, wav = folder / "meta.json", folder / "loop.wav"
    if not (meta_path.is_file() and wav.is_file()):
        raise FileNotFoundError(
            f"no kit called {name!r} in {kits_home(home)}. "
            "Build one with `fourfloor kit build <a house remix.mp3>`."
        )
    meta = json.loads(meta_path.read_text(encoding="utf8"))
    clip = decode(wav)
    return Kit(name=meta.get("name", name), loop=clip.samples, sr=clip.sr,
               bars=int(meta.get("bars", KIT_BARS)), source=meta.get("source", ""),
               source_bpm=float(meta.get("source_bpm", 0.0)),
               kick_beats=[float(t) for t in meta.get("kick_beats", [])],
               score=float(meta.get("score", 0.0)), path=folder)


def catalogue(home: str | Path | None = None) -> list[dict]:
    """Every stored kit, newest first, as its metadata plus ``built``."""
    root = kits_home(home)
    if not root.is_dir():
        return []
    out = []
    for folder in root.iterdir():
        meta = folder / "meta.json"
        if not (folder.is_dir() and meta.is_file() and (folder / "loop.wav").is_file()):
            continue
        try:
            d = json.loads(meta.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError):
            continue
        d.setdefault("name", folder.name)
        d["mtime"] = meta.stat().st_mtime
        out.append(d)
    return sorted(out, key=lambda d: -d["mtime"])


def resolve(name: str | None, home: str | Path | None = None) -> Kit | None:
    """The kit a remix should use: the one asked for, the newest, or none.

    ``"none"`` is how a caller says "use the synthesised kit even though real
    ones exist"; anything else is a name; ``None`` means "the most recently
    built, if there is one".
    """
    if name and name.lower() == "none":
        return None
    if name:
        return load(name, home)
    rows = catalogue(home)
    return load(rows[0]["name"], home) if rows else None
