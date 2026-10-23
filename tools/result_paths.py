from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

# Anchor to the repo root so output always lands in the canonical results/ tree
# regardless of the current working directory.
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
EXPERIMENT_DIR = RESULTS_ROOT / "experiment"
PICTURE_DIR = RESULTS_ROOT / "picture"
ANIMATION_DIR = RESULTS_ROOT / "animations"
CSV_DIR = RESULTS_ROOT / "csv"
HTML_DIR = RESULTS_ROOT / "html"
MD_DIR = RESULTS_ROOT / "md"
SNAPSHOT_DIR = RESULTS_ROOT / "snapshots"

OutputTokenMode = Literal["auto", "disabled", "path"]


def ensure_results_dirs() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    MD_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def result_dir(kind: str | None = None) -> Path:
    ensure_results_dirs()
    normalized = (kind or "").strip().lower()
    if normalized in {"", "json", "results", "root"}:
        return RESULTS_ROOT
    if normalized in {"experiment", "experiments", "archive", "archives"}:
        return EXPERIMENT_DIR
    if normalized in {"picture", "png", "plot", "plots", "image", "images"}:
        return PICTURE_DIR
    if normalized in {"animation", "animations", "gif", "video"}:
        return ANIMATION_DIR
    if normalized == "csv":
        return CSV_DIR
    if normalized == "html":
        return HTML_DIR
    if normalized in {"md", "markdown"}:
        return MD_DIR
    if normalized in {"snapshot", "snapshots"}:
        return SNAPSHOT_DIR
    raise ValueError(f"Unsupported results directory kind: {kind}")


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def classify_output_token(value: str | int | bool | None) -> OutputTokenMode:
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "auto" if value else "disabled"
    text = str(value).strip()
    if text == "":
        return "auto"
    lowered = text.lower()
    if lowered in {"auto", "1", "true", "yes", "y", "on"}:
        return "auto"
    if lowered in {"0", "false", "no", "n", "off"}:
        return "disabled"
    return "path"


def result_path(filename: str, kind: str | None = None) -> Path:
    return result_dir(kind) / filename


def unique_result_path(filename: str, kind: str | None = None) -> Path:
    base = result_path(filename, kind=kind)
    if not base.exists():
        return base
    stem = base.stem
    suffix = base.suffix
    counter = 1
    while True:
        candidate = base.with_name(f"{stem}_{counter:02d}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def timestamped_result_path(prefix: str, extension: str, *, kind: str | None = None, timestamp: str | None = None) -> Path:
    ts = timestamp or timestamp_now()
    ext = extension.lstrip(".")
    return unique_result_path(f"{prefix}_{ts}.{ext}", kind=kind)


def experiment_archive_dir(timestamp: str | None = None) -> Path:
    ts = str(timestamp or timestamp_now()).strip()
    if not ts:
        ts = timestamp_now()
    archive_dir = result_dir("experiment") / ts
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def experiment_archive_path(filename: str, *, timestamp: str | None = None) -> Path:
    archive_dir = experiment_archive_dir(timestamp)
    return archive_dir / str(filename).strip()


def normalize_result_prefix(prefix: str | None, *, fallback: str = "RESULT") -> str:
    text = str(prefix or "").strip().replace(" ", "_")
    return text.upper() or fallback


def dataset_result_stem(dataset: str | Path | None, *, fallback: str = "dataset") -> str:
    text = str(dataset or "").strip()
    stem = Path(text).stem if text else ""
    return stem or fallback


def build_run_stem(
    prefix: str | None,
    dataset: str | Path | None,
    artifact: str,
    step_count: int | None = None,
    *,
    timestamp: str | None = None,
    qualifier: str | None = None,
) -> str:
    norm_prefix = normalize_result_prefix(prefix)
    dataset_stem = dataset_result_stem(dataset)
    norm_artifact = str(artifact).strip().lower() or "artifact"
    parts = [norm_prefix]
    if qualifier not in {None, ""}:
        parts.append(normalize_result_prefix(qualifier, fallback="RUN"))
    parts.extend([timestamp or timestamp_now(), dataset_stem, norm_artifact])
    if step_count is not None:
        parts.append(str(int(step_count)))
    return "_".join(parts)


def trace_result_paths(
    strategy: str | None,
    dataset: str | Path | None,
    steps: int,
    *,
    timestamp: str | None = None,
    qualifier: str | None = None,
) -> tuple[Path, Path]:
    stem = build_run_stem(strategy, dataset, "score", int(steps), timestamp=timestamp, qualifier=qualifier)
    json_out = unique_result_path(f"{stem}.json", kind="snapshots")
    plot_stem = stem.replace("_score_", "_curve_", 1)
    plot_out = unique_result_path(f"{plot_stem}.png", kind="picture")
    return json_out, plot_out


def trace_plot_path_from_json(json_path: str | Path) -> Path:
    json_stem = Path(json_path).stem
    plot_stem = json_stem.replace("_score_", "_curve_", 1)
    return result_path(f"{plot_stem}.png", kind="picture")


def sequence_info_result_path(dataset: str | Path | None, *, timestamp: str | None = None) -> Path:
    stem = build_run_stem("SEQINFO", dataset, "sequence", timestamp=timestamp)
    return unique_result_path(f"{stem}.json", kind="snapshots")


def artifact_result_path(
    prefix: str | None,
    dataset: str | Path | None,
    artifact: str,
    extension: str,
    *,
    step_count: int | None = None,
    kind: str | None = None,
    timestamp: str | None = None,
    qualifier: str | None = None,
) -> Path:
    stem = build_run_stem(prefix, dataset, artifact, step_count=step_count, timestamp=timestamp, qualifier=qualifier)
    return unique_result_path(f"{stem}.{extension.lstrip('.')}", kind=kind)


def experiment_artifact_path(
    prefix: str | None,
    dataset: str | Path | None,
    artifact: str,
    extension: str,
    *,
    step_count: int | None = None,
    timestamp: str | None = None,
    qualifier: str | None = None,
) -> Path:
    stem = build_run_stem(prefix, dataset, artifact, step_count=step_count, timestamp=timestamp, qualifier=qualifier)
    return experiment_archive_path(f"{stem}.{extension.lstrip('.')}", timestamp=timestamp)
