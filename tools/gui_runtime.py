"""Shared process environment setup for interactive Qt/Matplotlib viewers."""

from __future__ import annotations

import os
from pathlib import Path


def configure_gui_environment() -> None:
    """Apply the WSLg-safe environment used by every task-1 GUI path."""
    runtime_dir = Path("/tmp/syncmaprgci-runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(runtime_dir, 0o700)
    os.environ["XDG_RUNTIME_DIR"] = str(runtime_dir)
    os.environ["QT_XCB_GL_INTEGRATION"] = "none"

    mpl_config_dir = Path("/tmp/syncmaprgci-mpl")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
