"""Install LAION-CLAP into a sidecar venv under ``~/.fourfloor/critic``.

    python -m fourfloor.critic.install_clap

CLAP does not go in the project venv on purpose. Its dependency closure
pins ``numpy<2`` and fourfloor runs numpy 2, so installing it next to the
remixer would silently downgrade the array library the DSP is tested
against. A sidecar interpreter costs disk and buys total isolation: the
critic talks to it over a ``.npy`` file and a subprocess.

laion-clap 1.1.7 also under-declares its own requirements -- torch,
torchvision and torchaudio are all imported but none are in its metadata
-- so they are named explicitly here.
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

from .embed import HOME, SIDECAR

EXTRA = ["torch", "torchvision", "torchaudio"]


def install(force: bool = False) -> Path:
    """Create the sidecar and install CLAP into it. Returns the interpreter."""
    if SIDECAR.exists() and not force:
        return SIDECAR
    HOME.mkdir(parents=True, exist_ok=True)
    target = HOME / "clap-venv"
    print(f"creating {target}")
    venv.EnvBuilder(with_pip=True, clear=force).create(target)
    pip = [str(target / "bin" / "pip"), "install", "--upgrade"]
    subprocess.run(pip + ["pip"], check=True)
    print("installing torch (this is the slow part)")
    subprocess.run(pip + EXTRA, check=True)
    print("installing laion-clap")
    subprocess.run(pip + ["laion-clap"], check=True)
    subprocess.run([str(SIDECAR), "-c", "import laion_clap"], check=True)
    print(f"ready: {SIDECAR}\nthe model weights download on the first critique.")
    return SIDECAR


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        install(force="--force" in argv)
    except subprocess.CalledProcessError as exc:
        print(f"install failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
