#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec streamlit run src/alpaca_dashboard/app.py "$@"
