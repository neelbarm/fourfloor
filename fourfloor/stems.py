"""Source separation: built-in HPSS, or demucs when it is installed.

The built-in path is median-filtering HPSS (``dsp.hpss``), which is always
available and needs no model. ``--stems demucs`` runs Meta's Hybrid Transformer
Demucs (Rouard, Massa & Défossez, "Hybrid Transformers for Music Source
Separation", ICASSP 2023) if the optional ``[stems]`` extra is installed, giving
a real four-way split and therefore a true vocal-house remix.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .audio import SR, decode, fit, write_wav
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


def separate_hpss(x: np.ndarray, sr: int, want_bass: bool = True) -> Stems:
    """Split into harmonic (vocals + chords), percussive (drums) and low end.

    HPSS cannot isolate a bass guitar, but the bottom of the harmonic half *is*
    the bass: below 180 Hz there is almost nothing else in a mix but the bass
    and the kick's tail, and the kick is being replaced anyway. It is a coarser
    answer than demucs gives and it is still the song's own low end rather than
    a synthesiser playing a chord estimate.
    """
    harm, perc = hpss_stereo(x, sr)
    bass = None
    if want_bass:
        from .dsp import filters as FL
        bass = FL.apply(harm, "lowpass", sr, 180.0, q=0.707, order=2)
        harm = FL.apply(harm, "highpass", sr, 150.0, q=0.707, order=2)
    return Stems(harmonic=harm, percussive=perc, source_name="hpss",
                 bass=bass, bass_name="hpss low band" if want_bass else "synth")


def separate_demucs(x: np.ndarray, sr: int = SR, model: str = DEMUCS_MODEL,
                    jobs: int = 2, want_bass: bool = True) -> Stems:
    """Run demucs and fold its four stems into fourfloor's two buses.

    ``x`` is the source *after* the beat-by-beat warp onto the target grid, not
    the original file: the arrangement addresses stems in warped seconds, so
    separating the untouched file would hand the engine a bed at the source
    tempo and every cut would land off the grid. Demucs is tempo-agnostic, so
    running it on the warped audio costs nothing in quality.

    ``vocals`` and ``other`` become the harmonic bed (the house kit supplies the
    rhythm section); ``drums`` becomes the percussive bed, used only where the
    plan asks for original-drum texture; ``bass`` becomes the bass bed.

    That last one used to be thrown away, on the theory that replacing the low
    end with a synthesised rolling bass at the target key was the point of the
    remix. It is not. The synthesised bass follows a chord estimate, and a chord
    estimate off a dense trap mix is frequently wrong -- wrong by a third, which
    is a wrong chord, played loudly, under the vocal that is telling you what
    the chord actually is. The song already knows its own bassline.
    """
    if not demucs_available():
        raise RuntimeError(
            "demucs is not installed. `pip install 'fourfloor[stems]'` (pulls torch), "
            "or use --stems hpss."
        )
    tmp = Path(tempfile.mkdtemp(prefix="fourfloor-demucs-"))
    try:
        src = write_wav(tmp / "warped.wav", x, sr)
        # torch defaults to one thread per core *inside each* of the -j workers,
        # which oversubscribes badly on a laptop; give each worker a fair share.
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // jobs)))
        proc = subprocess.run(
            [sys.executable, "-m", "demucs", "-n", model, "-j", str(jobs),
             "-o", str(tmp), str(src)],
            capture_output=True, text=True, check=False, env=env,
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
        bass = parts["bass"] if parts["bass"] is not None else zero
        return Stems(vocals=fit(vocals, n).astype(np.float32),
                     other=fit(other, n).astype(np.float32),
                     harmonic=fit(vocals + other, n).astype(np.float32),
                     percussive=fit(drums, n).astype(np.float32),
                     source_name="demucs",
                     bass=fit(bass, n).astype(np.float32) if want_bass else None,
                     bass_name="demucs bass" if want_bass else "synth")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def separate(x: np.ndarray, sr: int, mode: str = "hpss",
             want_bass: bool = True) -> Stems:
    """Separate the warped source by ``mode``.

    Both engines see the same warped buffer, so the stems they return are
    already on the target grid that the arrangement addresses.
    """
    if mode == "demucs":
        return separate_demucs(x, sr, want_bass=want_bass)
    return separate_hpss(x, sr, want_bass=want_bass)
