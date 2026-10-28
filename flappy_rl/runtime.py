"""Device selection and random seeding."""
import random

import numpy as np
import torch


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
