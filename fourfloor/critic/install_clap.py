"""Install LAION-CLAP into a sidecar venv under ``~/.fourfloor/critic``.

    python -m fourfloor.critic.install_clap            # install, fetch weights, slim
    python -m fourfloor.critic.install_clap --slim     # slim an existing install
    python -m fourfloor.critic.install_clap --force    # rebuild from scratch
    python -m fourfloor.critic.install_clap --no-weights

CLAP does not go in the project venv on purpose. Its dependency closure
pins ``numpy<2`` and fourfloor runs numpy 2, so installing it next to the
remixer would silently downgrade the array library the DSP is tested
against. A sidecar interpreter costs disk and buys total isolation: the
critic talks to it over a ``.npy`` file and a subprocess.

laion-clap 1.1.7 under-declares its own requirements -- torch and
torchvision are imported unconditionally (``clap_module/utils.py`` needs
``torchvision.ops.misc.FrozenBatchNorm2d``) but neither is in its metadata
-- so they are named explicitly here. torchaudio is *not* installed: the
only import is in ``training/data.py`` inside ``try/except ImportError``,
and the inference path the critic uses (``CLAP_Module.load_ckpt`` and
``get_audio_embedding_from_data``) never reaches it.

Where the disk actually goes (measured on Apple silicon, Python 3.12):

* The sidecar is **2.0 GB**: about 1.45 GB of packages (torch 582 MB,
  llvmlite + numba 154 MB for librosa, transformers 110 MB, scipy 97 MB)
  plus a **635 MB** checkpoint.
* It was 3.2 GB before slimming. The published checkpoint,
  ``630k-audioset-best.pt``, is 1.78 GB because 1.23 GB of it is the Adam
  optimizer state from training. ``clap_module.factory.load_state_dict``
  reads only ``checkpoint["state_dict"]``, so :func:`slim_checkpoints`
  rewrites the file without the optimizer. Embeddings are bit-identical.
* The PyTorch CPU-only index (``download.pytorch.org/whl/cpu``) does not
  help on macOS: it serves the same 127 MB arm64 wheel PyPI does, because
  macOS wheels never carried CUDA. It is worth using on Linux x86_64,
  where the default wheel bundles CUDA libraries; see ``TORCH_INDEX``.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import venv
from pathlib import Path

from .embed import HOME, SIDECAR

#: Imported unconditionally by laion-clap but missing from its metadata.
EXTRA = ["torch", "torchvision"]

#: CPU-only wheels. A large saving on Linux, where the default torch wheel
#: bundles CUDA; a no-op on macOS, where the wheels are identical.
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

# Runs inside the sidecar: fetch the default non-fusion checkpoint the worker
# uses, so the first critique does not stall on a 1.8 GB download.
_FETCH = """
import laion_clap
m = laion_clap.CLAP_Module(enable_fusion=False)
m.load_ckpt(verbose=False)
"""

# Runs inside the sidecar: rewrite every checkpoint in the laion_clap package
# keeping only what inference reads. Atomic, so an interrupted run leaves the
# original file in place.
_SLIM = """
import os, sys, tempfile
import torch
import laion_clap
pkg = os.path.dirname(laion_clap.__file__)
saved = 0
for name in sorted(os.listdir(pkg)):
    if not name.endswith(".pt"):
        continue
    path = os.path.join(pkg, name)
    before = os.path.getsize(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "state_dict" in ck):
        print(f"{name}: not a training checkpoint, left alone")
        continue
    extra = [k for k in ck if k not in ("state_dict", "epoch", "name")]
    if not extra:
        print(f"{name}: already slim ({before / 1e6:.0f} MB)")
        continue
    slim = {k: ck[k] for k in ("state_dict", "epoch", "name") if k in ck}
    del ck
    fd, tmp = tempfile.mkstemp(dir=pkg, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(slim, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    after = os.path.getsize(path)
    saved += before - after
    print(f"{name}: {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB (dropped {', '.join(extra)})")
print(f"reclaimed {saved / 1e6:.0f} MB")
"""


def _pip_install(pip: list[str], packages: list[str], *, torch: bool = False) -> None:
    """Install ``packages``; on Linux pull torch from the CPU-only index."""
    cmd = pip + packages
    if torch and platform.system() == "Linux":
        cmd += ["--index-url", TORCH_INDEX, "--extra-index-url", "https://pypi.org/simple"]
    subprocess.run(cmd, check=True)


def slim_checkpoints() -> None:
    """Strip training-only state from the sidecar's CLAP checkpoints.

    The published checkpoints carry the Adam optimizer state (about two
    thirds of the file). Inference reads ``state_dict`` alone, so dropping
    the rest changes nothing the critic computes.
    """
    if not SIDECAR.exists():
        raise FileNotFoundError(f"no sidecar at {SIDECAR}; run the installer first")
    subprocess.run([str(SIDECAR), "-c", _SLIM], check=True)


def fetch_weights() -> None:
    """Download the checkpoint the worker loads, then slim it."""
    print("fetching the CLAP checkpoint (1.8 GB download, 635 MB kept)")
    subprocess.run([str(SIDECAR), "-c", _FETCH], check=True)
    slim_checkpoints()


def install(force: bool = False, weights: bool = True) -> Path:
    """Create the sidecar and install CLAP into it. Returns the interpreter."""
    if SIDECAR.exists() and not force:
        return SIDECAR
    HOME.mkdir(parents=True, exist_ok=True)
    target = HOME / "clap-venv"
    print(f"creating {target}")
    venv.EnvBuilder(with_pip=True, clear=force).create(target)
    pip = [str(target / "bin" / "pip"), "install", "--upgrade", "--no-cache-dir"]
    subprocess.run(pip + ["pip"], check=True)
    print("installing torch (this is the slow part)")
    _pip_install(pip, EXTRA, torch=True)
    print("installing laion-clap")
    _pip_install(pip, ["laion-clap"])
    subprocess.run([str(SIDECAR), "-c", "import laion_clap"], check=True)
    if weights:
        fetch_weights()
        print(f"ready: {SIDECAR}")
    else:
        print(f"ready: {SIDECAR}\nthe model weights download on the first critique;"
              " run with --slim afterwards to reclaim 1.2 GB.")
    return SIDECAR


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        if "--slim" in argv:
            slim_checkpoints()
        else:
            install(force="--force" in argv, weights="--no-weights" not in argv)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"install failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
