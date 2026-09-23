#!/usr/bin/env python3
"""Visit a Streamlit Cloud app with a real browser to prevent Zzz sleep.

Streamlit Community Cloud only counts "viewer activity" when a browser
actually loads the page (JavaScript + WebSocket). A plain HTTP GET only
returns the static HTML shell and does NOT wake or keep the app alive.

This script uses Playwright (Chromium) to really load the page. If the
app shows the "Zzzz..." sleep screen, it clicks the wake-up button and
waits for the app to come back up.
"""
from __future__ import annotations

import asyncio
import os
import sys

from playwright.async_api import async_playwright

APP_URL = os.environ.get(
    "STREAMLIT_APP_URL",
    "https://montecarlo-portfolio-simulator-a7d5pfukvvvoy9tytzhc96.streamlit.app/",
)
TIMEOUT_MS = 120_000  # 2 min per page load (sleep -> wake can be slow)
WAKE_WAIT_MS = 90_000  # wait for the app process to actually start


async def visit(page, url: str) -> bool:
    """Load the app page. Click the wake-up button if it's sleeping."""
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        # Give the JS a moment to render either the sleep screen or the app
        await page.wait_for_timeout(5000)

        wake_btn = page.get_by_role("button", name="Yes, get this app back up!")
        if await wake_btn.count() > 0:
            print(f"WAKE {url}")
            await wake_btn.click()
            await page.wait_for_timeout(WAKE_WAIT_MS)
        else:
            print(f"OK {url}")
        return True
    except Exception as e:  # noqa: BLE001 - log and report failure, don't crash the job
        print(f"NG {url} -- {e}")
        return False


async def main() -> int:
    print(f"Visiting {APP_URL} ...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()
        ok = await visit(page, APP_URL)
        await browser.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
