"""QA: click Run on Moderate in /Remote_Control, verify data lands in Turso.

Strategy:
  1. Navigate to /Remote_Control
  2. Wait for cards to render
  3. Take baseline row count from Turso
  4. Click "▶ Run 50" in the Moderate card
  5. Poll Turso until rows_added >= some_threshold OR 3 min elapse
  6. Screenshot before/after, report counts
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from alpaca_dashboard import store  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

URL = "https://alpaca-backtesting.streamlit.app/Remote_Control"


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def app_frame(page):
    for f in page.frames:
        if "/~/+/" in f.url:
            return f
    return page.main_frame


with sync_playwright() as pw:
    browser = pw.chromium.launch(
        headless=False,
        args=["--window-size=1400,1100", "--window-position=40,60"],
    )
    page = browser.new_context(viewport={"width": 1400, "height": 1100}).new_page()

    log(f"→ {URL}")
    page.goto(URL, wait_until="domcontentloaded", timeout=60_000)

    # Wait for card headings to render inside the app iframe
    log("waiting for cards to render…")
    deadline = time.time() + 90
    ready = False
    while time.time() < deadline:
        f = app_frame(page)
        try:
            t = f.evaluate("() => document.body && document.body.innerText || ''")
        except Exception:
            t = ""
        if "▶ Run 50" in t and "Moderate" in t:
            ready = True
            break
        page.wait_for_timeout(2000)
    log(f"cards ready: {ready}")

    page.screenshot(path="/tmp/qa_rc_before.png", full_page=True)

    baseline = len(store.pulses_for_algo("moderate", limit=100_000))
    log(f"baseline Moderate pulses: {baseline}")

    # Click Run 50 inside the Moderate card. Each card has its own "▶ Run 50"
    # button. We target by role+name within the app iframe, then click nth(2)
    # (order: Degen, Surge, Moderate — Moderate is index 2).
    frame = app_frame(page)
    run_buttons = frame.get_by_role("button", name="▶ Run 50")
    count = run_buttons.count()
    log(f"found {count} '▶ Run 50' buttons; clicking index 2 (Moderate)")
    if count < 3:
        log("not enough buttons; aborting")
        browser.close()
        sys.exit(1)

    run_buttons.nth(2).click(timeout=10_000)
    log("clicked — waiting for rows to appear in Turso")

    # Poll Turso for new rows — jobs are in a background thread on Cloud
    deadline = time.time() + 180
    final = baseline
    last_ui_done = 0
    while time.time() < deadline:
        time.sleep(4)
        final = len(store.pulses_for_algo("moderate", limit=100_000))
        # also read UI progress for logging
        try:
            t = app_frame(page).evaluate("() => document.body.innerText")
        except Exception:
            t = ""
        import re
        m = re.search(r"🏃 (\d+)/50", t)
        ui_done = int(m.group(1)) if m else 0
        if ui_done != last_ui_done:
            log(f"  UI progress: {ui_done}/50 · Turso rows: {final} (+{final-baseline})")
            last_ui_done = ui_done
        if final > baseline + 8:
            log("enough rows written — stopping poll early")
            break

    page.screenshot(path="/tmp/qa_rc_after.png", full_page=True)

    # Summary
    print("\n" + "=" * 60)
    print(f"RC RUN QA RESULT")
    print(f"  cards_rendered:   {ready}")
    print(f"  baseline_rows:    {baseline}")
    print(f"  final_rows:       {final}")
    print(f"  rows_added:       {final - baseline}")
    print(f"  before_shot:      /tmp/qa_rc_before.png")
    print(f"  after_shot:       /tmp/qa_rc_after.png")
    print("=" * 60)

    page.wait_for_timeout(2000)
    browser.close()
