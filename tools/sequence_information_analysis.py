from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from unified_experiments import create_model, infer_default_env_mode, resolve_dataset_path


EPS = 1e-12
POISSON_RATE_95_FOR_FIVE_EVENTS = 9.2


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D transition matrix, got shape {arr.shape}")
    clipped = np.clip(arr, 0.0, None)
    row_sums = clipped.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= EPS
    if np.any(zero_rows):
        clipped[zero_rows] = 1.0 / float(clipped.shape[1])
        row_sums = clipped.sum(axis=1, keepdims=True)
    return clipped / np.clip(row_sums, EPS, None)


def _entropy(probs: np.ndarray) -> float:
    arr = np.asarray(probs, dtype=float).reshape(-1)
    arr = arr[arr > EPS]
    if arr.size == 0:
        return 0.0
    return float(-np.sum(arr * np.log2(arr)))


def _mutual_information_from_joint(joint: np.ndarray) -> float:
    joint_arr = np.asarray(joint, dtype=float)
    total = float(np.sum(joint_arr))
    if total <= EPS:
        return 0.0
    joint_norm = joint_arr / total
    px = np.sum(joint_norm, axis=1, keepdims=True)
    py = np.sum(joint_norm, axis=0, keepdims=True)
    denom = px @ py
    valid = joint_norm > EPS
    return float(np.sum(joint_norm[valid] * np.log2(joint_norm[valid] / np.clip(denom[valid], EPS, None))))


def _stationary_distribution(
    transition: np.ndarray,
    *,
    max_iter: int = 10000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = int(transition.shape[0])
    dist = np.full(n, 1.0 / float(max(1, n)), dtype=float)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        next_dist = dist @ transition
        if float(np.max(np.abs(next_dist - dist))) <= tol:
            dist = next_dist
            converged = True
            break
        dist = next_dist
    dist = np.clip(dist, 0.0, None)
    total = float(np.sum(dist))
    if total <= EPS:
        dist[:] = 1.0 / float(max(1, n))
    else:
        dist /= total
    return dist, {
        "converged": bool(converged),
        "iterations": int(iterations),
        "tolerance": float(tol),
    }


def _label_levels(env: Any) -> list[tuple[str, np.ndarray]]:
    if bool(getattr(env, "is_hierarchical", False)) and hasattr(env, "get_true_label_levels"):
        levels = np.asarray(env.get_true_label_levels(), dtype=int)
        return [(f"cluster_level{idx + 1}", np.asarray(levels[:, idx], dtype=int)) for idx in range(levels.shape[1])]
    if hasattr(env, "trueLabel"):
        return [("cluster_level1", np.asarray(env.trueLabel(), dtype=int))]
    return []


def _safe_unique_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(labels, dtype=int).reshape(-1)
    valid = arr >= 0
    return arr, valid


def _weighted_centroids(rows: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    unique_labels = sorted({int(v) for v in labels.tolist() if int(v) >= 0})
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    centroids = np.zeros((len(unique_labels), rows.shape[1]), dtype=float)
    label_probs = np.zeros(len(unique_labels), dtype=float)
    for label, centroid_idx in label_to_index.items():
        members = np.flatnonzero(labels == int(label))
        member_weights = np.asarray(weights[members], dtype=float)
        weight_sum = float(np.sum(member_weights))
        if weight_sum <= EPS:
            member_weights = np.full(members.size, 1.0 / float(max(1, members.size)), dtype=float)
            weight_sum = 1.0
        else:
            member_weights = member_weights / weight_sum
        centroids[centroid_idx] = np.sum(rows[members] * member_weights[:, None], axis=0)
        label_probs[centroid_idx] = float(np.sum(weights[members]))
    return centroids, label_to_index, label_probs


def _level_analysis(transition: np.ndarray, stationary: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    raw_labels, valid = _safe_unique_labels(labels)
    if not np.any(valid):
        return {
            "label_count": 0,
            "label_entropy": 0.0,
            "mutual_information_to_next_state": 0.0,
            "mutual_information_to_next_label": 0.0,
        }

    rows = np.asarray(transition[valid], dtype=float)
    weights = np.asarray(stationary[valid], dtype=float)
    weights = weights / np.clip(float(np.sum(weights)), EPS, None)
    valid_labels = np.asarray(raw_labels[valid], dtype=int)
    centroids, label_to_index, label_probs = _weighted_centroids(rows, valid_labels, weights)
    class_count = int(len(label_to_index))

    mapped = np.array([label_to_index[int(v)] for v in valid_labels], dtype=int)
    label_distribution = np.zeros(class_count, dtype=float)
    np.add.at(label_distribution, mapped, weights)

    joint_label_next = np.zeros((class_count, transition.shape[1]), dtype=float)
    for label_idx in range(class_count):
        members = np.flatnonzero(mapped == label_idx)
        joint_label_next[label_idx] = np.sum((weights[members, None] * rows[members]), axis=0)

    next_labels = np.zeros((class_count, class_count), dtype=float)
    membership = np.zeros((transition.shape[0], class_count), dtype=float)
    for label, label_idx in label_to_index.items():
        membership[:, label_idx] = (np.asarray(labels, dtype=int) == int(label)).astype(float)
    base_joint = stationary[:, None] * transition
    next_labels = membership.T @ base_joint @ membership
    off_diagonal = next_labels[~np.eye(class_count, dtype=bool)] if class_count > 1 else np.zeros(0)
    positive_cross_rates = off_diagonal[off_diagonal > EPS]
    min_cross_rate = float(np.min(positive_cross_rates)) if positive_cross_rates.size else None
    cross_event_flow = float(np.sum(off_diagonal))

    dist_to_centroid = np.linalg.norm(rows[:, None, :] - centroids[None, :, :], axis=2)
    pred = np.argmin(dist_to_centroid, axis=1)
    centroid_distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    if class_count > 1:
        centroid_mask = ~np.eye(class_count, dtype=bool)
        centroid_min = float(np.min(centroid_distances[centroid_mask]))
        centroid_mean = float(np.mean(centroid_distances[centroid_mask]))
    else:
        centroid_min = 0.0
        centroid_mean = 0.0

    own_distance = dist_to_centroid[np.arange(dist_to_centroid.shape[0]), mapped]
    within_rms = float(np.sqrt(np.sum(weights * np.square(own_distance)) / np.clip(float(np.sum(weights)), EPS, None)))
    weighted_accuracy = float(np.sum(weights[pred == mapped]))
    label_entropy = _entropy(label_distribution)
    mi_to_next_state = _mutual_information_from_joint(joint_label_next)
    mi_to_next_label = _mutual_information_from_joint(next_labels)

    return {
        "label_count": class_count,
        "label_entropy": label_entropy,
        "mutual_information_to_next_state": mi_to_next_state,
        "mutual_information_to_next_label": mi_to_next_label,
        "normalized_next_state_capture": float(mi_to_next_state / max(EPS, label_entropy)),
        "normalized_next_label_capture": float(mi_to_next_label / max(EPS, label_entropy)),
        "cross_relation_evidence": {
            "cross_event_flow": cross_event_flow,
            "minimum_positive_pair_event_rate": min_cross_rate,
            "memory_steps_for_five_events_at_95pct": (
                None
                if min_cross_rate is None
                else float(POISSON_RATE_95_FOR_FIVE_EVENTS / (2.0 * min_cross_rate))
            ),
            "global_steps_for_five_events_at_95pct": (
                None
                if min_cross_rate is None
                else float(POISSON_RATE_95_FOR_FIVE_EVENTS / min_cross_rate)
            ),
        },
        "row_centroid_separation": {
            "centroid_distance_min": centroid_min,
            "centroid_distance_mean": centroid_mean,
            "within_label_rms": within_rms,
            "separation_ratio": float(centroid_min / max(EPS, within_rms)),
        },
        "oracle_row_centroid_classifier": {
            "weighted_accuracy": weighted_accuracy,
            "nmi": float(normalized_mutual_info_score(mapped, pred)),
            "ari": float(adjusted_rand_score(mapped, pred)),
        },
        "stationary_label_mass": {
            str(label): float(label_distribution[idx])
            for label, idx in sorted(label_to_index.items(), key=lambda item: item[1])
        },
    }


def analyze_environment_sequence(env: Any) -> dict[str, Any]:
    if not hasattr(env, "a"):
        raise ValueError("Environment does not expose transition matrix 'a'.")
    transition = _normalize_rows(np.asarray(getattr(env, "a"), dtype=float))
    stationary, stationary_meta = _stationary_distribution(transition)
    next_dist = stationary @ transition
    conditional_entropy = float(np.sum(stationary * np.array([_entropy(row) for row in transition], dtype=float)))
    state_joint = stationary[:, None] * transition
    level_results = {
        key: _level_analysis(transition, stationary, level_labels)
        for key, level_labels in _label_levels(env)
    }
    return {
        "state_count": int(transition.shape[0]),
        "is_hierarchical": bool(getattr(env, "is_hierarchical", False)),
        "hierarchy_depth": int(getattr(env, "hierarchy_depth", 1)),
        "stationary_distribution": {
            **stationary_meta,
            "entropy": _entropy(stationary),
            "effective_support": float(2.0 ** _entropy(stationary)),
        },
        "dynamics": {
            "state_entropy": _entropy(stationary),
            "next_state_entropy": _entropy(next_dist),
            "conditional_entropy": conditional_entropy,
            "mutual_information": _mutual_information_from_joint(state_joint),
            "mean_self_transition": float(np.sum(stationary * np.diag(transition))),
            "mean_row_entropy": float(np.mean(np.array([_entropy(row) for row in transition], dtype=float))),
        },
        "levels": level_results,
    }


def analyze_dataset_sequence(
    dataset: str,
    *,
    strategy: str = "rg",
    env_mode: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_path = resolve_dataset_path(dataset)
    effective_env_mode = infer_default_env_mode(dataset_path, strategy) if env_mode in {None, "", "auto"} else str(env_mode)
    _model, env = create_model(
        dataset_path,
        "nsgd",
        {
            **dict(params or {}),
            "env_mode": effective_env_mode,
        },
        record_trajectory=False,
    )
    return {
        "dataset": dataset_path,
        "env_mode": effective_env_mode,
        "analysis": analyze_environment_sequence(env),
    }


def print_sequence_information_summary(result: dict[str, Any]) -> None:
    analysis = dict(result.get("analysis", result))
    dynamics = dict(analysis.get("dynamics", {}))
    print("-" * 90)
    print("Sequence Information Summary")
    print("-" * 90)
    print(
        f"  states={int(analysis.get('state_count', 0))}  "
        f"hierarchical={bool(analysis.get('is_hierarchical', False))}  "
        f"depth={int(analysis.get('hierarchy_depth', 1))}"
    )
    print(
        f"  H(X)={float(dynamics.get('state_entropy', 0.0)):.4f}  "
        f"H(X'|X)={float(dynamics.get('conditional_entropy', 0.0)):.4f}  "
        f"I(X;X')={float(dynamics.get('mutual_information', 0.0)):.4f}"
    )
    for level_key, info in dict(analysis.get("levels", {})).items():
        classifier = dict(info.get("oracle_row_centroid_classifier", {}))
        separation = dict(info.get("row_centroid_separation", {}))
        evidence = dict(info.get("cross_relation_evidence", {}))
        min_event_rate = evidence.get("minimum_positive_pair_event_rate")
        evidence_steps = evidence.get("global_steps_for_five_events_at_95pct")
        evidence_suffix = ""
        if min_event_rate is not None and evidence_steps is not None:
            evidence_suffix = (
                f"  q_min={float(min_event_rate):.8g}"
                f"  N95(k=5)={float(evidence_steps):.0f}"
            )
        print(
            f"  {level_key:16s} labels={int(info.get('label_count', 0)):>2d}  "
            f"I(L;X')={float(info.get('mutual_information_to_next_state', 0.0)):.4f}  "
            f"capture={float(info.get('normalized_next_state_capture', 0.0)):.4f}  "
            f"oracle_acc={float(classifier.get('weighted_accuracy', 0.0)):.4f}  "
            f"sep={float(separation.get('separation_ratio', 0.0)):.4f}  "
            f"{evidence_suffix}"
        )
    print("-" * 90)
