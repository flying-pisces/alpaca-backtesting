"""Streamlit Cloud entrypoint.

Streamlit Cloud auto-detects ``streamlit_app.py`` at the repo root, so we
keep this file at the top level and import the real home page from the
``alpaca_dashboard`` package under ``src/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Import the home page module — running its top-level code is what Streamlit
# actually renders, so we call the renderer directly.
from alpaca_dashboard import app  # noqa: E402,F401  (side-effect import)
