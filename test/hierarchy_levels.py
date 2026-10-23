from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from tools.shared_defaults import DEFAULT_HIERARCHY_ORGANIZE_LEVEL


IntArray = NDArray[np.intp]

DEFAULT_ORGANIZE_LEVEL = DEFAULT_HIERARCHY_ORGANIZE_LEVEL


def cluster_level_key(level_number: int) -> str:
    return f"cluster_level{max(1, int(level_number))}"


def cluster_level_key_from_index(level_index: int) -> str:
    return cluster_level_key(int(level_index) + 1)


def cluster_level_display(level_index: int) -> str:
    return f"L{int(level_index) + 1}"


def default_organize_level() -> str:
    return DEFAULT_ORGANIZE_LEVEL


def parse_cluster_level_index(name: str | None) -> int | None:
    text = str(name or "").strip().lower()
    match = re.fullmatch(r"cluster_level(\d+)", text)
    if match is None:
        return None
    level_number = int(match.group(1))
    if level_number <= 0:
        return None
    return level_number - 1


def parse_hierarchy_label_index(name: str | None) -> int | None:
    text = str(name or "").strip().lower()
    cluster_idx = parse_cluster_level_index(text)
    if cluster_idx is not None:
        return cluster_idx
    if text in {"cluster_level0", "level0", "node_clusters", "raw_clusters", "micro_clusters"}:
        return 0
    if text in {"level1", "top", "level2_refined", "cluster_level2_refined", "refined_level2"}:
        return 1
    match = re.fullmatch(r"level(\d+)", text)
    if match is None:
        return None
    return max(0, int(match.group(1)))


def canonicalize_organize_level(name: str | None, *, default: str = DEFAULT_ORGANIZE_LEVEL) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return default
    if text in {"cluster_level0", "level0", "node_clusters", "raw_clusters", "micro_clusters"}:
        return "cluster_level1"
    if text in {"stable", "stable_tokens"}:
        return "stable_tokens"
    if text in {"macro", "coarse", "macro_tokens", "coarse_tokens"}:
        return "macro_tokens"
    if text in {"macro_labels", "level2_refined", "cluster_level2_refined", "refined_level2", "top"}:
        return "cluster_level2"
    cluster_idx = parse_cluster_level_index(text)
    if cluster_idx is not None:
        return cluster_level_key_from_index(cluster_idx)
    match = re.fullmatch(r"level(\d+)", text)
    if match is not None:
        return cluster_level_key_from_index(int(match.group(1)))
    return text


def canonicalize_hierarchy_outputs(raw_outputs: Mapping[str, Any]) -> dict[str, IntArray]:
    base: dict[str, IntArray] = {}
    for key, value in raw_outputs.items():
        arr = np.asarray(value, dtype=int)
        if arr.ndim == 1:
            base[str(key)] = arr.copy()

    outputs: dict[str, IntArray] = {key: value.copy() for key, value in base.items()}

    for key, value in base.items():
        match = re.fullmatch(r"level(\d+)", key.lower())
        if match is None:
            continue
        outputs.setdefault(cluster_level_key_from_index(int(match.group(1))), value.copy())

    if "level0" in base:
        outputs["cluster_level1"] = base["level0"].copy()
    if "cluster_level2_refined" in base:
        outputs["cluster_level2_refined"] = base["cluster_level2_refined"].copy()
    if "cluster_level2" in base:
        outputs["cluster_level2"] = base["cluster_level2"].copy()
    elif "level1" in base:
        outputs["cluster_level2"] = base["level1"].copy()
        outputs.setdefault("cluster_level2_refined", base["level1"].copy())
    elif "cluster_level2" in outputs:
        outputs["cluster_level2_refined"] = outputs["cluster_level2"].copy()
    if "stable_tokens" in base:
        outputs["stable_tokens"] = base["stable_tokens"].copy()
    if "macro_tokens" in base:
        outputs["macro_tokens"] = base["macro_tokens"].copy()
    if "cluster_level1" in outputs:
        outputs["level0"] = outputs["cluster_level1"].copy()
    if "cluster_level2" in outputs:
        outputs["level1"] = outputs["cluster_level2"].copy()

    return outputs
