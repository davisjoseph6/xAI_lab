"""Shared utilities for the XAI laboratory.

These functions should remain small and easy to inspect. Avoid hiding
important methodological choices inside generic helper functions.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr


def ensure_directory(path: str | Path) -> Path:
    """Create a directory, including its parents, if necessary."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_global_seed(seed: int) -> None:
    """Set available Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # These choices favour reproducibility over maximum performance.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def json_default(value: Any) -> Any:
    """Convert common scientific-Python objects into JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if hasattr(value, "to_dict"):
        return value.to_dict()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serialisable."
    )


def save_json(data: Any, path: str | Path) -> None:
    """Save data as formatted JSON."""
    destination = Path(path)
    ensure_directory(destination.parent)

    with destination.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )


def sha256_file(path: str | Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate Spearman correlation, returning NaN for invalid inputs."""
    first_flat = np.asarray(first, dtype=float).ravel()
    second_flat = np.asarray(second, dtype=float).ravel()

    if first_flat.shape != second_flat.shape:
        raise ValueError(
            "Spearman inputs have different shapes: "
            f"{first_flat.shape} versus {second_flat.shape}."
        )

    if np.all(first_flat == first_flat[0]):
        return float("nan")

    if np.all(second_flat == second_flat[0]):
        return float("nan")

    correlation = spearmanr(first_flat, second_flat).statistic
    return float(correlation)


def top_k_jaccard(
    first_features: Iterable[str],
    second_features: Iterable[str],
    k: int = 5,
) -> float:
    """Calculate the Jaccard overlap between two top-k feature sets."""
    first_set = set(list(first_features)[:k])
    second_set = set(list(second_features)[:k])

    union = first_set | second_set

    if not union:
        return 1.0

    return len(first_set & second_set) / len(union)


def normalise_for_display(values: np.ndarray) -> np.ndarray:
    """Normalise a numerical map to [0, 1] for display only.

    Never use this display-normalised array for additivity checks,
    attribution totals, or model-randomisation correlations.
    """
    array = np.asarray(values, dtype=float)
    array = array - np.nanmin(array)
    maximum = np.nanmax(array)

    if maximum == 0 or not np.isfinite(maximum):
        return np.zeros_like(array)

    return array / maximum
