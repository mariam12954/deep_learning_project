"""
seif-online.com Web Scraper
============================
Scrapes two datasets:
  1. products_data.csv  — product name, price, SKU, description, category, subcategory, URL
  2. images/            — all product images saved locally + images_index.csv with metadata

Requirements:
    pip install playwright pandas requests tqdm
    python -m playwright install chromium

Usage:
    python seif_scraper.py
"""

import asyncio
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# ── Config ───────────────────────────────────────────────────────────────────
BASE_URL     = "https://seif-online.com/en"
OUTPUT_DIR   = Path("seif_output")
IMAGES_DIR   = OUTPUT_DIR / "images"
PRODUCTS_CSV = OUTPUT_DIR / "products_data.csv"
IMAGES_CSV   = OUTPUT_DIR / "images_index.csv"
LOG_FILE     = OUTPUT_DIR / "scraper.log"
HEADLESS     = True       # Set False to watch the browser open
PAGE_TIMEOUT = 30_000     # ms per page load
NAV_PAUSE    = 1.5        # seconds between navigations (be polite)
MAX_RETRIES  = 3
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

_start_time = time.time()
_log_handle = open(LOG_FILE, "w", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _elapsed() -> str:
    s = int(time.time() - _start_time)
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"

def log(msg: str, level: str = "INFO"):
    line = f"[{_ts()}] [{level}] {msg}"
    print(line, flush=True)
    _log_handle.write(line + "\n")
    _log_handle.flush()

def log_step(step: int, total: int, title: str):
    bar = "-" * 56
    log(f"\n{bar}")
    log(f"  STEP {step}/{total}  --  {title}")
    log(f"{bar}")

def log_ok(msg: str):
    log(f"OK  {msg}", "OK")

def log_warn(msg: str):
    log(f"!!  {msg}", "WARN")

def log_err(msg: str):
    log(f"XX  {msg}", "ERROR")

def log_progress(current: int, total: int, label: str):
    pct = (current / total * 100) if total else 0
    log(f"  [{current:>4} / {total}]  ({pct:5.1f}%)  {label}")


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_-]+", "_", text).strip("_")[:80]


def download_image(url: str, dest: Path, session: requests.Session) -> bool:
    if dest.exists():
        return True
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=20, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                log_err(f"Failed to download {url}: {e}")
                return False
            time.sleep(1)
    return False


async def safe_goto(page, url: str) -> bool:
    for attempt in range(MAX_RETRIES):
        try:
            await page.goto(url, wait_until="networkidle", timeout=PAGE_TIMEOUT)
            return True
        except PlaywrightTimeoutError:
            log_warn(f"Timeout on {url}  (attempt {attempt + 1}/{MAX_RETRIES})")
            await asyncio.sleep(2)
        except Exception as e:
            log_err(f"Navigation error on {url}: {e}")
            return False
    log_err(f"Giving up on {url} after {MAX_RETRIES} attempts.")
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1 — Discover categories & subcategories
# ═══════════════════════════════════════════════════════════════════════════

async def get_categories(page) -> list[dict]:
    log_step(1, 3, "Discovering categories & subcategories")
    log(f"Opening homepage -> {BASE_URL}")
    await safe_goto(page, BASE_URL)
    await asyncio.sleep(NAV_PAUSE)
    log_ok("Homepage loaded. Scanning nav links...")

    nav_links: list[dict] = []
    anchors = await page.query_selector_all("a[href]")
    log(f"   Found {len(anchors)} total anchor elements on the page.")
    seen = set()

    for a in anchors:
        href = await a.get_attribute("href") or ""
        text = (await a.inner_text()).strip()
        if not href:
            continue
        full = href if href.startswith("http") else urljoin(BASE_URL, href)
        if "seif-online.com" not in full:
            continue
        if any(seg in full for seg in ["/categories/", "/category/", "/c/", "/shop/", "/collection"]):
            if full not in seen and text:
                seen.add(full)
                nav_links.append({"raw_url": full, "link_text": text})

    log(f"   Filtered to {len(nav_links)} category-style links.")

    categories = []
    for item in nav_links:
        url   = item["raw_url"].rstrip("/")
        parts = urlparse(url).path.strip("/").split("/")
        parts = [p for p in parts if p not in ("en", "ar")]

        if len(parts) >= 3:
            category    = parts[1].replace("-", " ").title()
            subcategory = parts[2].replace("-", " ").title() if len(parts) >= 4 else None
        elif len(parts) == 2:
            category    = parts[1].replace("-", " ").title()
            subcategory = None
        else:
            continue

        categories.append({"category": category, "subcategory": subcategory, "url": item["raw_url"]})

    if not categories:
        log_warn("Nav approach found nothing -- trying /en/categories/ page as fallback...")
        categories = await get_categories_from_listing_page(page)

    if categories:
        log_ok(f"Discovered {len(categories)} category URLs:")
        for c in categories:
            sub = f"  ->  {c['subcategory']}" if c["subcategory"] else ""
            log(f"     * {c['category']}{sub}   ({c['url']})")
    else:
        log_err("No categories found at all. The site structure may have changed.")

    return categories


async def get_categories_from_listing_page(page) -> list[dict]:
    url = "https://seif-online.com/en/categories/"
    log(f"   Loading fallback page -> {url}")
    await safe_goto(page, url)
    await asyncio.sleep(NAV_PAUSE)

    cards = await page.query_selector_all("a[href]")
    results, seen = [], set()
    for card in cards:
        href = await card.get_attribute("href") or ""
        text = (await card.inner_text()).strip()
        if not href or not text:
            continue
        full = href if href.startswith("http") else urljoin(BASE_URL, href)
        if full in seen:
            continue
        seen.add(full)
        results.append({"category": text, "subcategory": None, "url": full})

    log(f"   Fallback found {len(results)} category links.")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — Collect product URLs from category pages
# ═══════════════════════════════════════════════════════════════════════════

async def scrape_products_from_category(
    page, cat: dict, cat_idx: int, cat_total: int
) -> list[dict]:
    label = cat["category"] + (f" / {cat['subcategory']}" if cat["subcategory"] else "")
    log(f"\n  [{cat_idx}/{cat_total}] Category: {label}")
    log(f"         URL: {cat['url']}")

    product_entries = []
    url = cat["url"]
    page_num = 1

    while url:
        log(f"       Loading page {page_num}...")
        ok = await safe_goto(page, url)
        if not ok:
            log_warn(f"       Skipping page {page_num} (failed to load).")
            break
        await asyncio.sleep(NAV_PAUSE)

        anchors = await page.query_selector_all("a[href]")
        found_on_page = 0
        seen_on_page = set()

        for a in anchors:
            href = await a.get_attribute("href") or ""
            full = href if href.startswith("http") else urljoin(BASE_URL, href)
            if "seif-online.com" not in full:
                continue
            if any(seg in full for seg in ["/products/", "/product/", "/p/", "/item/"]):
                if full not in seen_on_page:
                    seen_on_page.add(full)
                    found_on_page += 1
                    product_entries.append({
                        "product_url": full,
                        "category":    cat["category"],
                        "subcategory": cat["subcategory"],
                    })

        log(f"       Page {page_num}: {found_on_page} product links found"
            f"  (running total: {len(product_entries)})")

        next_btn = await page.query_selector(
            "a[aria-label='Next'], a.next, a[rel='next'], "
            "button.next, [class*='pagination'] a:last-child"
        )
        if next_btn:
            next_href = await next_btn.get_attribute("href")
            if next_href and next_href != url:
                url = next_href if next_href.startswith("http") else urljoin(BASE_URL, next_href)
                page_num += 1
                continue
        break

    log_ok(f"       Done -- {len(product_entries)} product URLs from \"{label}\".")
    return product_entries


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 — Scrape individual product pages
# ═══════════════════════════════════════════════════════════════════════════

async def scrape_product_detail(page, entry: dict) -> dict | None:
    ok = await safe_goto(page, entry["product_url"])
    if not ok:
        return None
    await asyncio.sleep(0.8)

    async def text(selector: str) -> str:
        el = await page.query_selector(selector)
        return (await el.inner_text()).strip() if el else ""

    async def attr(selector: str, attribute: str) -> str:
        el = await page.query_selector(selector)
        return (await el.get_attribute(attribute) or "").strip() if el else ""

    name = await text("h1") or await text("[class*='product-name']") or await text("[class*='title']")
    price = (
        await text("[class*='price']:not([class*='old']):not([class*='was'])")
        or await text("[class*='amount']")
        or await text("[itemprop='price']")
    )
    sku = (
        await text("[class*='sku']")
        or await text("[class*='barcode']")
        or await attr("[itemprop='sku']", "content")
    )
    description = (
        await text("[class*='description']")
        or await text("[itemprop='description']")
        or await text("[class*='details']")
    )
    brand        = await text("[class*='brand']") or await attr("[itemprop='brand']", "content")
    rating       = await attr("[itemprop='ratingValue']", "content") or await text("[class*='rating']")
    availability = (
        await text("[class*='stock']")
        or await text("[class*='availab']")
        or await attr("[itemprop='availability']", "content")
    )

    img_elements = await page.query_selector_all(
        "[class*='product'] img, [class*='gallery'] img, [class*='slider'] img, "
        "[id*='product'] img, main img"
    )
    image_urls = []
    for img in img_elements:
        src = (
            await img.get_attribute("src")
            or await img.get_attribute("data-src")
            or await img.get_attribute("data-lazy")
            or ""
        )
        if src and src.startswith("http") and src not in image_urls:
            if not any(skip in src.lower() for skip in ["logo", "icon", "placeholder", "spinner", "loading"]):
                image_urls.append(src)

    return {
        "name":         name,
        "price":        price,
        "sku":          sku,
        "brand":        brand,
        "category":     entry["category"],
        "subcategory":  entry["subcategory"],
        "description":  description[:500].replace("\n", " ") if description else "",
        "rating":       rating,
        "availability": availability,
        "image_urls":   "|".join(image_urls),
        "product_url":  entry["product_url"],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4 — Download images
# ═══════════════════════════════════════════════════════════════════════════

def download_all_images(products: list[dict]) -> list[dict]:
    log_step(3, 3, "Downloading product images")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; SeifScraper/1.0)"})

    total_images = sum(
        len([u for u in p.get("image_urls", "").split("|") if u])
        for p in products
    )
    log(f"   {len(products)} products -> estimated {total_images} images to download.")
    log(f"   Saving to: {IMAGES_DIR.resolve()}\n")

    image_records = []
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    for prod_idx, product in enumerate(products, 1):
        if not product.get("image_urls"):
            continue

        urls = [u for u in product["image_urls"].split("|") if u]
        product_slug = slugify(product.get("name") or "product")
        name_short   = (product.get("name") or "?")[:50]

        log(f"  [{prod_idx:>4}/{len(products)}]  {name_short}  ({len(urls)} image(s))")

        for idx, img_url in enumerate(urls):
            ext      = Path(urlparse(img_url).path).suffix or ".jpg"
            filename = f"{product_slug}_{idx + 1}{ext}"
            dest     = IMAGES_DIR / filename

            already_existed = dest.exists()
            success = download_image(img_url, dest, session)

            if success and already_existed:
                skipped_count += 1
                status = "skip (exists)"
            elif success:
                downloaded_count += 1
                status = "downloaded"
            else:
                failed_count += 1
                status = "FAILED"

            log(f"         img {idx + 1}: [{status}]  {filename}")

            image_records.append({
                "product_name": product.get("name", ""),
                "category":     product.get("category", ""),
                "subcategory":  product.get("subcategory", ""),
                "image_index":  idx + 1,
                "image_url":    img_url,
                "local_file":   str(dest) if success else "",
                "downloaded":   success,
                "product_url":  product.get("product_url", ""),
            })

    log(f"\n   Image download summary:")
    log(f"     Downloaded : {downloaded_count}")
    log(f"     Skipped    : {skipped_count}  (already existed on disk)")
    log(f"     Failed     : {failed_count}")
    return image_records


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    log("=" * 60)
    log("  seif-online.com Scraper  --  started " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log(f"  Output folder : {OUTPUT_DIR.resolve()}")
    log(f"  Log file      : {LOG_FILE.resolve()}")
    log(f"  Headless mode : {HEADLESS}")
    log("=" * 60)

    log("\nLaunching Chromium browser...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()
        log_ok("Browser launched successfully.")

        # ── Step 1: Categories ─────────────────────────────────────────────
        categories = await get_categories(page)
        if not categories:
            log_err("No categories found -- aborting. Try setting HEADLESS=False to debug.")
            await browser.close()
            _log_handle.close()
            return

        # ── Step 2: Collect product URLs ───────────────────────────────────
        log_step(2, 3, "Collecting product URLs from every category")
        all_entries: list[dict] = []

        for i, cat in enumerate(categories, 1):
            entries = await scrape_products_from_category(page, cat, i, len(categories))
            all_entries.extend(entries)

        # Deduplicate
        seen_urls: dict = {}
        deduped = []
        for e in all_entries:
            if e["product_url"] not in seen_urls:
                seen_urls[e["product_url"]] = True
                deduped.append(e)

        dupes = len(all_entries) - len(deduped)
        log(f"\n   Total collected  : {len(all_entries)}")
        log(f"   Duplicates removed: {dupes}")
        log_ok(f"{len(deduped)} unique product pages to scrape.")

        # ── Step 3: Scrape product detail pages ────────────────────────────
        log(f"\n   Scraping {len(deduped)} product pages. This may take a while...\n")

        products: list[dict] = []
        failed = 0

        for idx, entry in enumerate(deduped, 1):
            product = await scrape_product_detail(page, entry)
            name_display = (entry["product_url"].split("/")[-1] or entry["product_url"])[:55]

            if product:
                products.append(product)
                name_display = (product.get("name") or name_display)[:55]
                log_progress(
                    idx, len(deduped),
                    f"{name_display:<55}  price: {product.get('price') or '-'}"
                )
            else:
                failed += 1
                log_warn(f"  [{idx:>4}/{len(deduped)}]  FAILED -> {entry['product_url']}")

            # Checkpoint save every 50 products so you don't lose progress
            if idx % 50 == 0 and products:
                pd.DataFrame(products).to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")
                log(f"   [CHECKPOINT] Saved {len(products)} products so far -> {PRODUCTS_CSV}")

        await browser.close()
        log_ok("Browser closed.")

    # ── Save final products CSV ────────────────────────────────────────────
    log(f"\n   Total scraped : {len(products)}  |  Failed : {failed}")

    if products:
        pd.DataFrame(products).to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")
        log_ok(f"Products CSV saved -> {PRODUCTS_CSV}")
        log("\n   First 5 products preview:")
        log(f"   {'Name':<40}  {'Price':<12}  Category / Subcategory")
        log(f"   {'-'*40}  {'-'*12}  {'-'*30}")
        for p in products[:5]:
            sub = p.get("subcategory") or "-"
            log(f"   {(p.get('name') or '?')[:40]:<40}  {(p.get('price') or '?'):<12}  {p.get('category','?')} / {sub}")
    else:
        log_warn("No products scraped -- CSV not written.")

    # ── Download images ────────────────────────────────────────────────────
    if products:
        image_records = download_all_images(products)
        pd.DataFrame(image_records).to_csv(IMAGES_CSV, index=False, encoding="utf-8-sig")
        log_ok(f"Images index CSV saved -> {IMAGES_CSV}")

    # ── Final summary ──────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("  FINISHED")
    log(f"  Total run time  : {_elapsed()}")
    log(f"  Products scraped: {len(products)}")
    log(f"  Output folder   : {OUTPUT_DIR.resolve()}")
    log(f"  Log file        : {LOG_FILE.resolve()}")
    log("=" * 60)

    _log_handle.close()


if __name__ == "__main__":
    asyncio.run(main())