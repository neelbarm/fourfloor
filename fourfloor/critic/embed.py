"""Window embeddings and the reference cache behind the similarity score.

Two backends, picked at runtime:

``clap``
    LAION-CLAP, a contrastive audio/text model whose audio tower was
    trained on music. It is the backend that actually knows what "house
    record" sounds like. It cannot live in the project venv -- its
    dependency set pins ``numpy<2``, and fourfloor runs numpy 2 -- so it
    is installed once into a sidecar venv under ``~/.fourfloor/critic``
    and driven as a subprocess. The audio crosses the boundary as a
    ``.npy`` of 10-second windows; nothing else does.

``mfcc``
    A hand-built timbre-plus-rhythm vector: MFCC mean/std/delta, the
    onset autocorrelation profile, and octave band ratios. No model, no
    download, always available. It is a real fallback, not a stub, but it
    scores texture and pulse rather than genre, so it separates a house
    remix from its original far less confidently than CLAP does.

Reference embeddings are cached under ``~/.fourfloor/critic/cache`` keyed
by backend, file identity and window layout -- never inside the repo, and
never the audio itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import ANALYSIS_SR, spectral

WINDOW_SEC = 10.0
CLAP_SR = 48000
HOME = Path(os.environ.get("FOURFLOOR_HOME", Path.home() / ".fourfloor")) / "critic"
SIDECAR = HOME / "clap-venv" / "bin" / "python"
_WORKER = Path(__file__).with_name("_clap_worker.py")


def cache_dir() -> Path:
    d = HOME / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _file_key(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

class MfccRhythm:
    """Timbre and pulse descriptor, computed in process from numpy alone."""

    name = "mfcc-rhythm"
    sr = ANALYSIS_SR
    #: (best a non-house original managed, mean leave-one-out cosine the
    #: references reach against each other). Fitted on the calibration set.
    anchors = (0.643, 0.814)
    #: Half weight, and the critique says why. On the calibration set this
    #: backend gets the important pair backwards: ``body.fourfloor``, the
    #: render Neel called garbage, scores 0.946 -- higher than every real
    #: reference -- while ``body.classic`` scores 0.610, below the non-house
    #: originals. MFCCs and an autocorrelation profile describe texture and
    #: pulse density, and a dense bright mix has the texture of a house
    #: record whether or not its parts agree about where the beat is. It
    #: still separates house from non-house, which is worth something, so
    #: it contributes -- at half weight, with a warning.
    weight_factor = 0.5
    caveat = ("the mfcc fallback embedding ranks texture, not arrangement; on "
              "the calibration set it scored the worst render highest. Install "
              "CLAP (`python -m fourfloor.critic.install_clap`) for a "
              "similarity score worth trusting.")

    def embed(self, windows: list[np.ndarray]) -> np.ndarray:
        return np.stack([self._one(w) for w in windows]) if windows else np.zeros((0, 1))

    def _one(self, x: np.ndarray) -> np.ndarray:
        sp = spectral(x, self.sr)
        log_mel = np.log(sp.mel + 1e-8)
        n_mel = log_mel.shape[1]
        k = np.arange(20)[:, None]
        dct = np.cos(np.pi * k * (2 * np.arange(n_mel)[None, :] + 1) / (2 * n_mel))
        mfcc = log_mel @ dct.T                      # (frames, 20)
        mfcc = mfcc[:, 1:]                          # drop the loudness term
        timbre = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0),
                                 np.abs(np.diff(mfcc, axis=0)).mean(axis=0)])
        env = sp.onset
        from .features import autocorr
        ac = autocorr(env)
        lo, hi = int(0.10 * sp.fps), int(2.0 * sp.fps)
        seg = ac[lo: hi] if hi > lo and hi <= len(ac) else np.zeros(48)
        rhythm = np.interp(np.linspace(0, len(seg) - 1, 48),
                           np.arange(len(seg)), seg) if len(seg) > 1 else np.zeros(48)
        edges = np.linspace(0, n_mel, 9).astype(int)
        bands = np.array([sp.mel[:, a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
        bands = np.log(bands / max(bands.sum(), 1e-9) + 1e-6)
        return np.concatenate([_unit(timbre), _unit(rhythm), 0.7 * _unit(bands)])


class Clap:
    """LAION-CLAP audio embeddings, computed in the sidecar interpreter."""

    name = "laion-clap"
    sr = CLAP_SR
    #: Fitted on the calibration set: the six references reach 0.857-0.881
    #: against each other, the two non-house originals 0.592-0.606, and the
    #: two rated renders land where Neel put them (0.770 / 0.618).
    anchors = (0.606, 0.869)
    weight_factor = 1.0
    caveat = ""

    def __init__(self, python: Path) -> None:
        self.python = python

    def embed(self, windows: list[np.ndarray]) -> np.ndarray:
        if not windows:
            return np.zeros((0, 1))
        n = int(WINDOW_SEC * CLAP_SR)
        block = np.stack([_fit(w, n) for w in windows]).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.npy", Path(tmp) / "out.npy"
            np.save(src, block)
            proc = subprocess.run([str(self.python), str(_WORKER), str(src), str(dst)],
                                  capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not dst.exists():
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                raise RuntimeError(f"clap worker failed: {tail[-1] if tail else '?'}")
            return np.load(dst)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=x.dtype)])


def available_backends() -> list[str]:
    out = []
    if SIDECAR.exists() and _WORKER.exists():
        out.append("clap")
    out.append("mfcc")
    return out


def load_backend(prefer: str = "auto"):
    """Return the best available backend, honouring an explicit request."""
    if prefer in ("clap", "auto") and SIDECAR.exists() and _WORKER.exists():
        return Clap(SIDECAR)
    if prefer == "clap":
        raise RuntimeError(
            "the CLAP backend is not installed. Run "
            "`python -m fourfloor.critic.install_clap` or drop --embed clap."
        )
    return MfccRhythm()


# ---------------------------------------------------------------------------
# window selection
# ---------------------------------------------------------------------------

def drop_windows(mono: np.ndarray, sr: int, cues: list[float] | None = None,
                 limit: int = 4) -> list[tuple[float, np.ndarray]]:
    """The track's loudest, bassiest stretches: its drops.

    Session cues of kind ``drop`` win when a render has them. References
    never do, so they fall back to the energy rule -- which is the same
    rule, just derived rather than declared: rank non-overlapping
    10-second windows by RMS weighted toward low end, take the top few.
    """
    n = int(WINDOW_SEC * sr)
    if len(mono) < n:
        return [(0.0, _fit(mono, n))]
    picked: list[float] = []
    if cues:
        for t in cues[:limit]:
            start = min(max(0.0, t + 2.0), (len(mono) - n) / sr)
            picked.append(start)
    if not picked:
        hop = n // 2
        starts = np.arange(0, len(mono) - n + 1, hop)
        # low-band weighting: a house drop is where the kick and bass are
        lo = np.array([float(np.sqrt(np.mean(np.square(
            np.convolve(mono[s: s + n: 8], np.ones(16) / 16, mode="same")))))
            for s in starts])
        full = np.array([float(np.sqrt(np.mean(np.square(mono[s: s + n])))) for s in starts])
        rank = np.argsort(-(full * (0.5 + lo / max(lo.max(), 1e-9))))
        for i in rank:
            s = float(starts[i]) / sr
            if all(abs(s - p) >= WINDOW_SEC for p in picked):
                picked.append(s)
            if len(picked) >= limit:
                break
    picked.sort()
    return [(s, _fit(mono[int(s * sr): int(s * sr) + n], n)) for s in picked]


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------

@dataclass
class RefBank:
    vectors: np.ndarray            # (n_windows, dim)
    files: list[str]               # display names, one per window
    backend: str

    def __len__(self) -> int:
        return len(self.vectors)


def file_windows(path: Path, backend, cues: list[float] | None = None,
                 limit: int = 4) -> list[np.ndarray]:
    """Drop windows for one file, decoded at the backend's rate."""
    from ..audio import decode

    mono = decode(path, sr=backend.sr).mono.astype(np.float64)
    return [w for _, w in drop_windows(mono, backend.sr, cues, limit)]


def embed_file(path: Path, backend, cues: list[float] | None = None,
               limit: int = 4) -> np.ndarray:
    """Drop-window embeddings for one file."""
    return backend.embed(file_windows(path, backend, cues, limit))


def reference_bank(folder: Path, backend, limit: int = 4,
                   on_file=None) -> RefBank:
    """Embed every audio file in ``folder``, caching per file on disk.

    Every uncached reference goes through the backend in one call. With
    CLAP that matters a great deal: each call is a subprocess that loads
    a 2 GB checkpoint, so embedding six references one at a time cost six
    model loads and over two minutes. Batched, it is one.
    """
    from ..audio import find_audio

    files = find_audio(folder)
    if not files:
        raise RuntimeError(f"no audio files in {folder}")
    cached: dict[Path, np.ndarray] = {}
    pending: list[Path] = []
    for path in files:
        key = cache_dir() / f"{backend.name}-{_file_key(path)}-{limit}.npy"
        if key.exists():
            cached[path] = np.load(key)
        else:
            pending.append(path)

    if pending:
        if on_file:
            on_file(f"{len(pending)} reference{'s' if len(pending) > 1 else ''}")
        batch: list[np.ndarray] = []
        spans: list[tuple[Path, int]] = []
        for path in pending:
            wins = file_windows(path, backend, limit=limit)
            spans.append((path, len(wins)))
            batch.extend(wins)
        out = backend.embed(batch)
        at = 0
        for path, n in spans:
            v = out[at: at + n]
            at += n
            cached[path] = v
            np.save(cache_dir() / f"{backend.name}-{_file_key(path)}-{limit}.npy", v)

    vecs, names = [], []
    for path in files:
        v = cached[path]
        vecs.append(v)
        names.extend([path.stem] * len(v))
    return RefBank(vectors=np.concatenate(vecs), files=names, backend=backend.name)


def cosine_to_bank(vectors: np.ndarray, bank: RefBank) -> tuple[float, list[float]]:
    """Mean over the render's windows of its best match in the bank.

    Best-match rather than mean-to-all: a remix should sound like *a*
    reference drop, not like the average of five different records, and
    averaging over unrelated references just compresses every score into
    the middle.
    """
    if not len(vectors) or not len(bank):
        return 0.0, []
    a = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    b = bank.vectors / np.maximum(np.linalg.norm(bank.vectors, axis=1, keepdims=True), 1e-9)
    sims = a @ b.T
    per_window = sims.max(axis=1)
    return float(per_window.mean()), [float(v) for v in per_window]
