"""Unified convergence-curve comparison CLI.

One entrypoint for both equal-step and equal-wall-clock convergence curves.
Two scenarios, one script.

Subcommands
-----------
- steps   equal-step convergence comparison (was convergence_compare.py)
- time    equal-wall-clock convergence comparison (was convergence_time_compare.py)

Examples
--------
    python tools/conv.py steps --datasets fixed5_6.dot --strategies strict rg --steps 20000
    python tools/conv.py time  --datasets fixed5_6.dot --strategies strict rg --seconds 60
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
    "steps": "convergence_compare.py",
    "time": "convergence_time_compare.py",
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