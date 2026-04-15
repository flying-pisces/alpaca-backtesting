"""Self-driving QA for the live Streamlit app + Turso backend.

Runs headed so you can watch on the left monitor. Each pass:
  1. Drives the UI via Playwright.
  2. Verifies backend state via direct Turso HTTP queries.
  3. Saves screenshots to /tmp/qa_*.png.

Usage:
    .venv/bin/python scripts/qa.py home
    .venv/bin/python scripts/qa.py admin_run
    .venv/bin/python scripts/qa.py dashboard
    .venv/bin/python scripts/qa.py planned
    .venv/bin/python scripts/qa.py all
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from alpaca_dashboard import store  # noqa: E402
from playwright.sync_api import Page, Playwright, expect, sync_playwright  # noqa: E402

APP_URL = "https://alpaca-backtesting.streamlit.app"
SHOT_DIR = Path("/tmp")


def shot(page: Page, name: str) -> Path:
    ts = datetime.now().strftime("%H%M%S")
    p = SHOT_DIR / f"qa_{name}_{ts}.png"
    page.screenshot(path=str(p), full_page=True)
    return p


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def app_frame(page: Page):
    """Streamlit Cloud wraps the app in an iframe at ``/~/+/``. Interactions
    with the sidebar, buttons, tables, etc. must all go through that frame."""
    # Walk frames; the app frame is the one whose URL contains '/~/+/'
    for f in page.frames:
        if "/~/+/" in f.url:
            return f
    return page.main_frame


def wait_for_content(page: Page, text: str, timeout_ms: int = 120_000) -> bool:
    """Poll the app iframe for visible text (survives cold-start + rerun)."""
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        f = app_frame(page)
        try:
            body = f.evaluate("() => document.body && document.body.innerText || ''")
        except Exception:
            body = ""
        if text in (body or ""):
            return True
        page.wait_for_timeout(1000)
    return False


def text_on_page(page: Page) -> str:
    """Visible innerText of the app iframe (not the top frame)."""
    try:
        return app_frame(page).evaluate(
            "() => document.body && document.body.innerText || ''"
        )
    except Exception:
        return ""


# ── Pass 1: home / secrets / algo tables ─────────────────────────────────────
def qa_home(page: Page) -> dict:
    log("→ /")
    page.goto(APP_URL, wait_until="domcontentloaded", timeout=90_000)
    # The tables render first (no API); wait for that as the "app alive" signal.
    got_tables = wait_for_content(page, "Ready algos", timeout_ms=180_000)
    # Then wait a bit more for the Live paper accounts section + snapshots.
    got_snaps = wait_for_content(page, "Live paper accounts", timeout_ms=30_000)
    page.wait_for_timeout(5000)   # let snapshots paint after ThreadPool returns
    p = shot(page, "home")

    body = text_on_page(page)
    results = {
        "screenshot": str(p),
        "tables_loaded": got_tables,
        "snaps_section_loaded": got_snaps,
        "body_chars": len(body),
    }

    if "No accounts configured" in body:
        results["secrets_ok"] = False
        results["error"] = "Secrets not configured in Streamlit Cloud UI"
        return results

    results["secrets_ok"] = True

    # Account names are inside <metric> widgets — check each
    for name in ["Degen", "Surge", "Moderate", "Sentinel", "Fortress",
                 "Reddit Play", "ER Play", "Dividend Play"]:
        if name not in body:
            results.setdefault("missing", []).append(name)

    results["ready_table"] = "Ready algos" in body
    results["planned_table"] = "Planned algos" in body
    return results


# ── Pass 2: admin — run 20-pulse Moderate batch ──────────────────────────────
def qa_admin_run(page: Page) -> dict:
    log("→ /Admin")
    page.goto(f"{APP_URL}/Admin", wait_until="networkidle", timeout=60_000)
    page.wait_for_timeout(3000)

    baseline = len(store.pulses_for_algo("moderate", limit=100000))
    log(f"Turso baseline Moderate rows: {baseline}")

    # Lower target for a fast QA run (sidebar slider)
    # Set sidebar "Target pulses per run" to a small number via keyboard
    # Streamlit sliders are fiddly; simplest path: set "History window" and
    # trust the default 120 target, but 120 is slow. We'll click the slider
    # label and type Left arrows to drop it. Quicker: just run defaults.

    shot(page, "admin_before")

    # Click the Moderate expander header to open it
    moderate_label = page.get_by_text("⚖️ Moderate", exact=False).first
    try:
        moderate_label.scroll_into_view_if_needed()
        moderate_label.click(timeout=10_000)
    except Exception as e:
        log(f"couldn't click Moderate expander: {e}")
    page.wait_for_timeout(800)

    shot(page, "admin_moderate_expanded")

    # Click the ▶ Run button scoped to moderate's expander
    # Streamlit renders buttons as <button>, visible text includes "▶ Run"
    try:
        page.get_by_role("button", name="▶ Run").first.click(timeout=10_000)
    except Exception as e:
        log(f"couldn't click ▶ Run: {e}")
        shot(page, "admin_run_click_fail")
        return {"click_ok": False, "error": str(e)}

    log("clicked ▶ Run, waiting for job to produce rows...")
    # Poll Turso until baseline + at least 1 row shows up, up to 180s
    deadline = time.time() + 180
    final_count = baseline
    while time.time() < deadline:
        time.sleep(3)
        final_count = len(store.pulses_for_algo("moderate", limit=100000))
        if final_count > baseline:
            break

    shot(page, "admin_after_run")
    return {
        "click_ok": True,
        "baseline": baseline,
        "final_count": final_count,
        "wrote_rows": final_count - baseline,
    }


# ── Pass 3: dashboard render + data match ────────────────────────────────────
def qa_dashboard(page: Page) -> dict:
    log("→ /Dashboard")
    page.goto(f"{APP_URL}/Dashboard", wait_until="networkidle", timeout=60_000)
    page.wait_for_timeout(3000)
    p = shot(page, "dashboard")

    content = page.content()
    if "No backtest pulses stored yet" in content:
        return {"has_data": False, "screenshot": str(p)}

    # Find the headline count (first line says "N trades · M tickers ·  ...")
    # We also check that key chart headings render.
    turso_total = len(store.all_pulses(limit=100000))
    return {
        "has_data": True,
        "turso_total": turso_total,
        "has_cum_pnl": "Cumulative P&L" in content,
        "has_drawdown": "Drawdown" in content,
        "has_trade_log": "Trade log" in content,
        "screenshot": str(p),
    }


# ── Pass 4: planned algos disabled + coef save ───────────────────────────────
def qa_planned(page: Page) -> dict:
    log("→ /Admin (planned)")
    page.goto(f"{APP_URL}/Admin", wait_until="networkidle", timeout=60_000)
    page.wait_for_timeout(3000)
    shot(page, "admin_top")

    # Scroll to "Planned algos · 3"
    heading = page.get_by_text("Planned algos · 3").first
    try:
        heading.scroll_into_view_if_needed(timeout=10_000)
    except Exception as e:
        return {"planned_heading": False, "error": str(e)}

    page.wait_for_timeout(500)
    shot(page, "admin_planned")

    content = page.content()
    # Our code renders "▶ Run (disabled)" buttons for planned algos
    disabled_count = content.count("Run (disabled)")
    return {
        "planned_heading": True,
        "disabled_run_buttons": disabled_count,
    }


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run(pw: Playwright, pass_name: str) -> None:
    browser = pw.chromium.launch(headless=False, args=[
        "--window-size=1600,1000",
        "--window-position=40,60",
    ])
    context = browser.new_context(viewport={"width": 1600, "height": 1000})
    page = context.new_page()

    # Surface browser-side errors so we see WHY a page is blank.
    def _console(msg):
        if msg.type in ("error", "warning"):
            log(f"[console.{msg.type}] {msg.text[:300]}")
    def _pageerr(exc):
        log(f"[pageerror] {str(exc)[:300]}")
    page.on("console", _console)
    page.on("pageerror", _pageerr)
    try:
        if pass_name == "home" or pass_name == "all":
            r = qa_home(page)
            log(f"PASS 1 (home): {r}")
            if not r.get("secrets_ok"):
                log("→ human needed: paste secrets in Streamlit Cloud UI")
                return
        if pass_name == "admin_run" or pass_name == "all":
            r = qa_admin_run(page)
            log(f"PASS 2 (admin_run): {r}")
        if pass_name == "dashboard" or pass_name == "all":
            r = qa_dashboard(page)
            log(f"PASS 3 (dashboard): {r}")
        if pass_name == "planned" or pass_name == "all":
            r = qa_planned(page)
            log(f"PASS 4 (planned): {r}")
    finally:
        page.wait_for_timeout(2000)
        context.close()
        browser.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pass_name",
                    choices=["home", "admin_run", "dashboard", "planned", "all"],
                    default="home", nargs="?")
    args = ap.parse_args()
    with sync_playwright() as pw:
        run(pw, args.pass_name)


if __name__ == "__main__":
    main()
