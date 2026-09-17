"""Audio I/O: decode/encode through ffmpeg, plus small buffer helpers.

Everything inside fourfloor is float32, 44.1 kHz, shaped ``(n_samples, 2)`` for
stereo or ``(n_samples,)`` for mono. ffmpeg is used as the codec layer so we can
read anything it reads (mp3, m4a, wav, flac, aiff, ogg) without a Python codec
dependency.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 44100
"""Project-wide sample rate. Everything is resampled to this on decode."""


def _tool(name: str) -> str:
    """Locate an ffmpeg-family binary, preferring the Homebrew arm64 prefix."""
    for candidate in (f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(
        f"{name} not found. Install it with `brew install ffmpeg` and retry."
    )


AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".aiff", ".aif", ".ogg", ".opus", ".wma"}


@dataclass(frozen=True)
class Clip:
    """A decoded audio buffer plus provenance."""

    samples: np.ndarray  # (n, 2) float32
    sr: int
    path: str

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sr)

    @property
    def mono(self) -> np.ndarray:
        """Mono mixdown used for every analysis stage."""
        if self.samples.ndim == 1:
            return self.samples
        return self.samples.mean(axis=1)


def probe_duration(path: str | Path) -> float:
    """Return container duration in seconds via ffprobe (0.0 if unknown)."""
    out = subprocess.run(
        [
            _tool("ffprobe"), "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(json.loads(out.stdout)["format"]["duration"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return 0.0


def decode(path: str | Path, sr: int = SR) -> Clip:
    """Decode any ffmpeg-readable file to float32 stereo at ``sr``.

    Mono sources are duplicated to two channels; >2 channel sources are
    downmixed by ffmpeg's default matrix.
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    proc = subprocess.run(
        [
            _tool("ffmpeg"), "-v", "error", "-nostdin", "-i", path,
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "-",
        ],
        capture_output=True, check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode("utf8", "replace").strip().splitlines()
        raise RuntimeError(f"ffmpeg could not decode {path}: {msg[-1] if msg else 'no output'}")
    buf = np.frombuffer(proc.stdout, dtype="<f4")
    n = len(buf) // 2
    samples = np.ascontiguousarray(buf[: n * 2].reshape(n, 2).astype(np.float32))
    return Clip(samples=samples, sr=sr, path=path)


def write_wav(path: str | Path, x: np.ndarray, sr: int = SR) -> Path:
    """Write a 24-bit WAV (soundfile) with clamping to [-1, 1]."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(np.atleast_2d(x.T).T if x.ndim == 1 else x, -1.0, 1.0)
    sf.write(str(path), y.astype(np.float32), sr, subtype="PCM_24")
    return path


def write_mp3(path: str | Path, x: np.ndarray, sr: int = SR, bitrate: str = "320k") -> Path:
    """Encode a float buffer straight to MP3 through ffmpeg's libmp3lame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(x if x.ndim == 2 else np.stack([x, x], axis=1), -1.0, 1.0).astype("<f4")
    proc = subprocess.run(
        [
            _tool("ffmpeg"), "-v", "error", "-y", "-nostdin",
            "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
            "-codec:a", "libmp3lame", "-b:a", bitrate, str(path),
        ],
        input=y.tobytes(), capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mp3 encode failed: {proc.stderr.decode('utf8', 'replace')}")
    return path


def find_audio(folder: str | Path) -> list[Path]:
    """All audio files directly inside ``folder``, sorted, hidden files skipped."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith(".")
    )


# ---------------------------------------------------------------------------
# buffer helpers
# ---------------------------------------------------------------------------

def to_stereo(x: np.ndarray) -> np.ndarray:
    """Promote a mono buffer to stereo; pass stereo through."""
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    return x


def fit(x: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad or truncate ``x`` (mono or stereo) to exactly ``n`` frames."""
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    pad = n - len(x)
    if x.ndim == 1:
        return np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
    return np.concatenate([x, np.zeros((pad, x.shape[1]), dtype=x.dtype)])


def add_at(dst: np.ndarray, src: np.ndarray, offset: int, gain: float = 1.0) -> None:
    """Mix ``src`` into ``dst`` at sample ``offset``, clipped to bounds, in place."""
    if gain == 0.0 or len(src) == 0:
        return
    start = max(0, offset)
    end = min(len(dst), offset + len(src))
    if end <= start:
        return
    s0 = start - offset
    chunk = src[s0: s0 + (end - start)]
    if dst.ndim == 2 and chunk.ndim == 1:
        chunk = chunk[:, None]
    dst[start:end] += gain * chunk


def db(x: np.ndarray) -> float:
    """Peak level of a buffer in dBFS (-inf guarded at -120)."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return 20.0 * np.log10(max(peak, 1e-6))


def rms_db(x: np.ndarray) -> float:
    """RMS level in dBFS."""
    if not len(x):
        return -120.0
    return 20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(x)))), 1e-6))


def xfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Equal-power crossfade: last ``n`` frames of ``a`` into the head of ``b``."""
    n = int(min(n, len(a), len(b)))
    if n <= 0:
        return np.concatenate([a, b])
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out, fade_in = np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
    if a.ndim == 2:
        fade_out, fade_in = fade_out[:, None], fade_in[:, None]
    head = a[:-n] if n < len(a) else a[:0]
    mid = a[len(a) - n:] * fade_out + b[:n] * fade_in
    return np.concatenate([head, mid, b[n:]])
