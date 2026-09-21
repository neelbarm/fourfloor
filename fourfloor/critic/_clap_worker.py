"""Run LAION-CLAP in the sidecar interpreter and hand back embeddings.

This file is never imported by fourfloor. It is executed by
``~/.fourfloor/critic/clap-venv/bin/python``, which has numpy 1.x and the
CLAP dependency tree that the project venv deliberately does not. Keep it
to numpy and laion_clap only -- importing anything from ``fourfloor``
would drag numpy 2 code into a numpy 1 interpreter.

    python _clap_worker.py in.npy out.npy

``in.npy`` is float32 ``(n_windows, 480000)``: 10-second mono windows at
48 kHz. ``out.npy`` is float32 ``(n_windows, 512)``.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    import numpy as np
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    windows = np.load(argv[1]).astype(np.float32)
    if windows.ndim == 1:
        windows = windows[None, :]

    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()          # cached under ~/.cache after the first run
    model.eval()
    with torch.no_grad():
        out = model.get_audio_embedding_from_data(x=windows, use_tensor=False)
    np.save(argv[2], np.asarray(out, dtype=np.float32))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
