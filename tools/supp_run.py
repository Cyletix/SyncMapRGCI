#!/usr/bin/env python3
"""Supplementary experiments: community orientation (flat) and hierarchy difficulty.

    python tools/supp_run.py pilot
    python tools/supp_run.py run [--workers 3]

Each job is one `main.py --task 2` call for a (dataset, model variant, hierarchy decay) cell with
five paired seeds; its CSV is copied to results/supp/cells/ so an interrupted run resumes.
Learning curves are sampled every 2,000 steps and late_mean_ari averages the last five points.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "supp"
STEPS, TRIALS, INTERVAL = 100_000, 5, 2_000
_START_LOCK = threading.Lock()
_last_start = 0.0

FLAT = ["datasets/supp/fixed3_100.dot", "datasets/fixed5_60.dot", "datasets/supp/fixed10_30.dot",
        "datasets/supp/fixed15_20.dot", "datasets/supp/fixed20_15.dot", "datasets/supp/fixed30_10.dot",
        "datasets/fixed60_5.dot"]
HIER = ["datasets/hierarchy/DCH1_fixed_chunk_6_3.dot", "datasets/supp/HIMB3.dot",
        "datasets/supp/H4x4x5.dot", "datasets/supp/H3x3x3x4.dot"]
DECAYS = [0.25, 0.5, 0.75]
# variant -> (profile, extra --set overrides); rg_l0 is the RG kernel without recursion
VARIANTS = {"syncmap": ("syncmap", []), "symmetric": ("symmetric", []), "tsf": ("tsf", []),
            "nsgd": ("nsgd", []), "rg": ("rg", []), "rg_l0": ("rg", ["level1_enabled=false"])}


def jobs(pilot: bool) -> list[tuple[str, str, float | None]]:
    if pilot:
        return [("datasets/supp/H4x4x5.dot", v, 0.25) for v in ("rg", "nsgd")] + \
               [("datasets/supp/H3x3x3x4.dot", v, 0.75) for v in ("rg", "nsgd")] + \
               [("datasets/supp/fixed15_20.dot", v, None) for v in ("rg", "nsgd")]
    out: list[tuple[str, str, float | None]] = []
    for d in HIER:  # long RG cells first so the pool stays busy
        for decay in DECAYS:
            out += [(d, v, decay) for v in ("rg", "rg_l0", "nsgd", "tsf", "symmetric", "syncmap")]
    for d in FLAT:
        out += [(d, v, None) for v in ("rg", "nsgd", "tsf", "symmetric", "syncmap")]
    return out


def cell_name(dataset: str, variant: str, decay: float | None) -> str:
    return f"{Path(dataset).stem}__{variant}" + ("" if decay is None else f"__d{decay:g}")


def run_cell(dataset: str, variant: str, decay: float | None, trials: int, cells: Path) -> None:
    name = cell_name(dataset, variant, decay)
    target = cells / f"{name}.csv"
    if target.exists():
        return
    profile, sets = VARIANTS[variant]
    if decay is not None:
        sets = sets + [f"env_hierarchy_level_switch_decay={decay}"]
    args = ["--task", "2", "--profiles", profile, "--datasets", dataset,
            "--comparison-sequence-lengths", str(STEPS), "--comparison-trials", str(trials),
            "--metric", "ari", "--seed", "0", "--resume-mode", "off",
            "--comparison-trace-json", "--comparison-record-interval", str(INTERVAL)]
    for s in sets:
        args += ["--set", s]
    global _last_start
    with _START_LOCK:  # task 2 names its archive by start second, so starts are kept apart
        wait = _last_start + 2.5 - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_start = time.time()
    t0 = time.time()
    proc = subprocess.run([sys.executable, "main.py", *args], cwd=ROOT, capture_output=True, text=True,
                          env=dict(os.environ, MPLBACKEND="Agg"))
    (OUT / "logs" / f"{name}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(f"FAILED {name} ({proc.returncode})", flush=True)
        return
    archive = Path(re.findall(r"Results saved to: (\S+)", proc.stdout)[-1])
    import pandas as pd
    df = pd.read_csv(archive / "comparison_full_results.csv", low_memory=False)
    df = df.drop(columns=[c for c in ("predicted_labels", "true_labels") if c in df.columns])
    df.insert(0, "variant", variant)
    df.insert(1, "decay", decay)
    df.to_csv(target, index=False)
    print(f"done {name:40s} {time.time() - t0:7.0f}s  ari={df['ari_score'].mean():.3f}", flush=True)


SHOWCASE = [("datasets/supp/H4x4x5.dot", 0.25), ("datasets/supp/H4x4x5.dot", 0.75)]


def run_showcase(dataset: str, variant: str, decay: float) -> None:
    """Seed-0 run that keeps the final 3-D coordinates, for the latent-space figure."""
    import json

    name = cell_name(dataset, variant, decay)
    target = OUT / "showcase" / f"{name}.json"
    if target.exists():
        return
    profile, sets = VARIANTS[variant]
    args = ["--task", "0", "--profile", profile, "--dataset", dataset, "--steps", str(STEPS), "--seed", "0",
            "--record-interval", str(INTERVAL), "--save-json", "true", "--save-png", "false", "--save-plot", "false",
            "--save-gif", "false"]
    for s in sets + [f"env_hierarchy_level_switch_decay={decay}"]:
        args += ["--set", s]
    proc = subprocess.run([sys.executable, "main.py", *args], cwd=ROOT, capture_output=True, text=True,
                          env=dict(os.environ, MPLBACKEND="Agg"))
    (OUT / "logs" / f"showcase_{name}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(f"FAILED showcase {name} ({proc.returncode})", flush=True)
        return
    outputs = dict(re.findall(r"(saved_\w+)=(\S+)", proc.stdout.split("outputs:")[-1]))
    snapshot = json.loads(Path(outputs["saved_snapshot"]).read_text(encoding="utf-8"))
    record = json.loads(Path(outputs["saved_json"]).read_text(encoding="utf-8"))
    levels = {k: record[k] for k in record if re.fullmatch(r"cluster_level\d+_ari_score", k)}
    target.write_text(json.dumps({
        "dataset": dataset, "variant": variant, "decay": decay, "ari": record.get("ari_score"), "level_ari": levels,
        "positions": [node["position"] for node in snapshot["nodes"]],
        "true_label_levels": snapshot["true_label_levels"],
    }), encoding="utf-8")
    print(f"showcase {name:40s} ari={record.get('ari_score'):.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pilot", "run", "showcase"])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "showcase":
        for sub in (OUT / "showcase", OUT / "logs"):
            sub.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for f in [pool.submit(run_showcase, d, v, decay) for d, decay in SHOWCASE for v in ("rg", "rg_l0", "nsgd")]:
                f.result()
        return
    pilot = args.mode == "pilot"
    cells = OUT / ("pilot" if pilot else "cells")
    for sub in (cells, OUT / "logs"):
        sub.mkdir(parents=True, exist_ok=True)
    todo = jobs(pilot)
    print(f"{len(todo)} cells, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for d, v, decay in todo:
            if (cells / f"{cell_name(d, v, decay)}.csv").exists():
                continue
            futures.append(pool.submit(run_cell, d, v, decay, 1 if pilot else TRIALS, cells))
        for f in futures:
            f.result()
    import pandas as pd
    frames = [pd.read_csv(p, low_memory=False) for p in sorted(cells.glob("*.csv"))]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUT / f"{args.mode}_results.csv", index=False)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
