"""Streamlit Cloud entrypoint.

Streamlit reruns this file on every interaction, so we call ``render()``
explicitly rather than relying on module-import side effects (imports are
cached in ``sys.modules`` and don't re-execute on rerun).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alpaca_dashboard.app import render  # noqa: E402

render()
