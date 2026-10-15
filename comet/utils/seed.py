import os
import random

import numpy as np
import torch

from config import RANDOM_STATE


def set_seed(seed=RANDOM_STATE):
    """Seed every source of randomness COMET touches (python/numpy/torch) and force
    deterministic torch kernels, so a run from the same code + data reproduces bit-for-bit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
