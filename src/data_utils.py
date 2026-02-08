from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_names(class_names_json: str | Path) -> List[str]:
    path = Path(class_names_json)
    if not path.exists():
        raise FileNotFoundError(
            f"class_names_json not found: {path}. "
            "Run `python -m src.prepare_mimic ...` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    names = obj.get("class_names", None)
    if not names:
        raise ValueError(f"Invalid class_names.json: missing 'class_names' field: {path}")
    return list(names)


def load_split_csv(csv_path: str | Path, class_names: List[str]) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}")
    df = pd.read_csv(path)

    required = {"rel_path"} | set(class_names)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def compute_pos_weight(train_df: pd.DataFrame, class_names: List[str]) -> np.ndarray:
    """Compute BCEWithLogitsLoss pos_weight per class: (N_neg / N_pos).

    Returns a float32 numpy array of shape (C,).
    """
    labels = train_df[class_names].to_numpy(dtype=np.float32)  # (N, C)
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    # Avoid division by zero
    pos_weight = neg / np.clip(pos, 1.0, None)
    return pos_weight.astype(np.float32)
