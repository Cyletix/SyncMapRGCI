"""Unified graph-switch CLI.

One entrypoint for the whole graph-switch scenario: run the probe, then build
the diagnostic report from the probe archive.

Subcommands
-----------
- probe    graph-switch plasticity probe (was graph_switch_plasticity_probe.py)
- report   build markdown report from a probe archive
           (was graph_switch_diagnostics_report.py)

Examples
--------
    python tools/switch.py probe --datasets fixed5_6.dot --seconds 20
    python tools/switch.py report results/experiment/<timestamp>/
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PROBES = Path(__file__).resolve().parent / "_probes"

_REGISTRY = {
    "probe": "graph_switch_plasticity_probe.py",
    "report": "graph_switch_diagnostics_report.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("subcommand", choices=sorted(_REGISTRY))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO)

    if len(sys.argv) >= 2 and sys.argv[1] in _REGISTRY and any(
        token in ("-h", "--help") for token in sys.argv[2:]
    ):
        return subprocess.call(
            [sys.executable, str(_PROBES / _REGISTRY[sys.argv[1]]), "-h"], env=env
        )
    args, remaining = parser.parse_known_args()
    script = _PROBES / _REGISTRY[args.subcommand]
    return subprocess.call([sys.executable, str(script), *remaining], env=env)


if __name__ == "__main__":
    raise SystemExit(main())