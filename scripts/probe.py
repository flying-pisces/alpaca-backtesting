"""One-shot DOM inspector — print top-frame + all iframe innerText."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from playwright.sync_api import sync_playwright

URL = "https://alpaca-backtesting.streamlit.app"

with sync_playwright() as pw:
    b = pw.chromium.launch(headless=False, args=["--window-size=1400,900"])
    p = b.new_context(viewport={"width": 1400, "height": 900}).new_page()
    p.goto(URL, wait_until="domcontentloaded", timeout=60_000)
    p.wait_for_timeout(15_000)  # give Streamlit time to wake
    print("=== top frame innerText (first 800 chars) ===")
    print((p.evaluate("() => document.body.innerText") or "")[:800])
    print("\n=== frame tree ===")
    for f in p.frames:
        print(f"  frame: name={f.name!r} url={f.url[:80]!r}")
    print("\n=== each frame innerText (first 300 chars) ===")
    for i, f in enumerate(p.frames):
        try:
            t = f.evaluate("() => document.body && document.body.innerText || ''")
        except Exception as e:
            t = f"(evaluate failed: {e})"
        print(f"--- frame {i} ({f.url[:60]}) ---")
        print((t or "")[:300])
    p.wait_for_timeout(1500)
    b.close()
