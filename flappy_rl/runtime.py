"""Device selection, random seeding and output folders."""
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

OUTPUT_DIR_NAME = "flappy_bird_output"
COLAB_DRIVE = Path("/content/drive/MyDrive")


def resolve_device(name="auto"):
    """Return a torch.device; "auto" picks CUDA when available."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def set_seed(seed=42, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)  # seeds CPU without initializing CUDA
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)


def output_root():
    """Where runs are saved: $FLAPPY_OUTPUT, else Google Drive on Colab, else ./flappy_bird_output.

    Saving to Drive on Colab keeps checkpoints when the runtime disconnects.
    """
    if os.environ.get("FLAPPY_OUTPUT"):
        return Path(os.environ["FLAPPY_OUTPUT"])
    if COLAB_DRIVE.is_dir():
        return COLAB_DRIVE / OUTPUT_DIR_NAME
    return Path(OUTPUT_DIR_NAME)


def new_run_dir(mode, root=None):
    """Create <root>/<mode>_<YYYYmmdd_HHMMSS>/ and return it."""
    root = Path(root) if root is not None else output_root()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"{mode}_{stamp}"
    suffix = 1
    while path.exists():  # two runs in the same second
        path = root / f"{mode}_{stamp}_{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path
