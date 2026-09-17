"""Source separation: built-in HPSS, or demucs when it is installed.

The built-in path is median-filtering HPSS (``dsp.hpss``), which is always
available and needs no model. ``--stems demucs`` runs Meta's Hybrid Transformer
Demucs (Rouard, Massa & Défossez, "Hybrid Transformers for Music Source
Separation", ICASSP 2023) if the optional ``[stems]`` extra is installed, giving
a real four-way split and therefore a true vocal-house remix.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .audio import SR, decode, fit
from .dsp.hpss import hpss_stereo
from .house.engine import Stems

DEMUCS_MODEL = "htdemucs"


def demucs_available() -> bool:
    """True when the demucs package can be imported in this interpreter."""
    try:
        import demucs  # noqa: F401
    except ImportError:
        return False
    return True


def separate_hpss(x: np.ndarray, sr: int) -> Stems:
    """Split into harmonic (vocals + chords) and percussive (original drums)."""
    harm, perc = hpss_stereo(x, sr)
    return Stems(harmonic=harm, percussive=perc, source_name="hpss")


def separate_demucs(path: str | Path, sr: int = SR, model: str = DEMUCS_MODEL,
                    jobs: int = 4) -> Stems:
    """Run demucs and fold its four stems into fourfloor's two buses.

    ``vocals`` and ``other`` become the harmonic bed (the house kit supplies the
    rhythm section); ``drums`` becomes the percussive bed, used only where the
    plan asks for original-drum texture. The original ``bass`` stem is
    deliberately discarded -- replacing the low end with a synthesised rolling
    bass at the target key is the point of the remix.
    """
    if not demucs_available():
        raise RuntimeError(
            "demucs is not installed. `pip install 'fourfloor[stems]'` (pulls torch), "
            "or use --stems hpss."
        )
    tmp = Path(tempfile.mkdtemp(prefix="fourfloor-demucs-"))
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "demucs", "-n", model, "-j", str(jobs),
             "-o", str(tmp), str(path)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"demucs failed: {proc.stderr.strip()[-400:]}")
        stem_dir = next((p for p in (tmp / model).iterdir() if p.is_dir()), None)
        if stem_dir is None:
            raise RuntimeError("demucs produced no output directory")

        parts: dict[str, np.ndarray] = {}
        for name in ("vocals", "other", "drums", "bass"):
            f = stem_dir / f"{name}.wav"
            parts[name] = decode(f, sr).samples if f.is_file() else None

        ref = next((v for v in parts.values() if v is not None), None)
        if ref is None:
            raise RuntimeError("demucs produced no stems")
        n = len(ref)
        zero = np.zeros_like(ref)
        vocals = parts["vocals"] if parts["vocals"] is not None else zero
        other = parts["other"] if parts["other"] is not None else zero
        drums = parts["drums"] if parts["drums"] is not None else zero
        return Stems(harmonic=fit(vocals + other, n).astype(np.float32),
                     percussive=fit(drums, n).astype(np.float32),
                     source_name="demucs")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def separate(path: str | Path, x: np.ndarray, sr: int, mode: str = "hpss") -> Stems:
    """Separate by ``mode``, falling back to HPSS if demucs is unavailable."""
    if mode == "demucs":
        return separate_demucs(path, sr)
    return separate_hpss(x, sr)
