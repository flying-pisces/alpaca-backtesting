"""Streamlit Cloud entrypoint.

Streamlit reruns this file on every interaction, so we call ``render()``
explicitly rather than relying on module-import side effects (imports are
cached in ``sys.modules`` and don't re-execute on rerun).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import streamlit as st

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from alpaca_dashboard.app import render
    render()
except Exception:   # noqa: BLE001 — top-level so users see *why*
    st.set_page_config(page_title="Error", layout="wide")
    st.error("⚠️ Entrypoint crashed — full traceback below:")
    st.code(traceback.format_exc(), language="python")
    st.caption("sys.path:")
    st.code("\n".join(sys.path), language="text")
