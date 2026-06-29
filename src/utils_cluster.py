import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch


def configure_runtime():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dir(output_dir):
    path = Path(output_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total_params, trainable_params


def safe_perplexity(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def write_summary(output_dir, summary):
    summary_path = Path(output_dir) / "summary.json"

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")

    return summary_path
