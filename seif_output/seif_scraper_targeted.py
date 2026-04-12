"""
seif_scraper_targeted.py
=========================
Scrapes only dangerous drug categories from seif-online.com.
Targets: heart, diabetes, blood pressure, medical devices.
Produces:
    seif_output/products_data.csv
    seif_output/images_index.csv
    seif_output/images/

Requirements:
    pip install playwright pandas requests tqdm
    playwright install chromium
"""

import asyncio
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from tqdm import tqdm
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# -- Config -------------------------------------------------------------------
BASE_URL     = "https://seif-online.com/en"
OUTPUT_DIR   = Path("seif_output")
IMAGES_DIR   = OUTPUT_DIR / "images"
PRODUCTS_CSV = OUTPUT_DIR / "products_data.csv"
IMAGES_CSV   = OUTPUT_DIR / "images_index.csv"
HEADLESS     = True
PAGE_TIMEOUT = 30_000
NAV_PAUSE    = 2.0
MAX_RETRIES  = 3

# -- Target categories to scrape (edit these URLs after checking the site) ---
# These are the direct category URLs for dangerous drugs on seif-online.com
# If any URL returns 0 products, open the site and find the correct URL
TARGET_CATEGORIES = [
    {
        "url":         "https://seif-online.com/en/categories/medicine/heart-medications",
        "category":    "Medicine",
        "subcategory": "Heart",
        "label":       "heart",
    },
    {
        "url":         "https://seif-online.com/en/categories/medicine/diabetes-medications",
        "category":    "Medicine",
        "subcategory": "Diabetes",
        "label":       "diabetes",
    },
    {
        "url":         "https://seif-online.com/en/categories/medicine/blood-pressure",
        "category":    "Medicine",
        "subcategory": "Blood Pressure",
        "label":       "blood_pressure",
    },
    {
        "url":         "https://seif-online.com/en/categories/medical-devices",
        "category":    "Medical Devices",
        "subcategory": "General",
        "label":       "medical_device",
    },
    {
        "url":         "https://seif-online.com/en/categories/medicine",
        "category":    "Medicine",
        "subcategory": "General",
        "label":       "general_medicine",
    },
]
# -----------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", str(text).lower())
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
                print(f"Failed to download {url}: {e}")
                return False
            time.sleep(1)
    return False


async def safe_goto(page, url: str) -> bool:
    for attempt in range(MAX_RETRIES):
        try:
            await page.goto(url, wait_until="networkidle", timeout=PAGE_TIMEOUT)
            return True
        except PlaywrightTimeoutError:
            print(f"Timeout on {url}, retry {attempt + 1}/{MAX_RETRIES}")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Error on {url}: {e}")
            return False
    return False


async def discover_real_urls(page) -> list[dict]:
    """
    Visit the main medicine category page and collect all subcategory URLs.
    This runs first to find the real URLs on the site.
    """
    print("Discovering real category URLs...")
    found = []

    medicine_pages = [
        "https://seif-online.com/en/categories/medicine",
        "https://seif-online.com/en/categories/",
        BASE_URL,
    ]

    seen = set()
    for start_url in medicine_pages:
        ok = await safe_goto(page, start_url)
        if not ok:
            continue
        await asyncio.sleep(NAV_PAUSE)

        anchors = await page.query_selector_all("a[href]")
        for a in anchors:
            href = await a.get_attribute("href") or ""
            text = (await a.inner_text()).strip()
            full = href if href.startswith("http") else urljoin(BASE_URL, href)

            if "seif-online.com" not in full:
                continue
            if full in seen:
                continue
            if not any(seg in full for seg in ["/categories/", "/category/", "/c/"]):
                continue

            seen.add(full)
            found.append({"url": full, "text": text})
            print(f"  Found: [{text}] {full}")

    return found


async def scrape_category(page, cat: dict) -> list[dict]:
    """Collect all product URLs from a category, handling pagination."""
    entries = []
    url = cat["url"]
    page_num = 1

    while url:
        ok = await safe_goto(page, url)
        if not ok:
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
                    entries.append({
                        "product_url": full,
                        "category":    cat["category"],
                        "subcategory": cat["subcategory"],
                        "label":       cat.get("label", "general_medicine"),
                    })

        print(f"  Page {page_num}: {found_on_page} products from {cat['subcategory']}")

        next_btn = await page.query_selector(
            "a[aria-label='Next'], a.next, a[rel='next'], [class*='pagination'] a:last-child"
        )
        if next_btn:
            next_href = await next_btn.get_attribute("href")
            if next_href and next_href != url:
                url = next_href if next_href.startswith("http") else urljoin(BASE_URL, next_href)
                page_num += 1
                continue
        break

    return entries


async def scrape_product(page, entry: dict) -> dict | None:
    ok = await safe_goto(page, entry["product_url"])
    if not ok:
        return None
    await asyncio.sleep(0.8)

    async def get_text(selector: str) -> str:
        el = await page.query_selector(selector)
        return (await el.inner_text()).strip() if el else ""

    async def get_attr(selector: str, attribute: str) -> str:
        el = await page.query_selector(selector)
        return (await el.get_attribute(attribute) or "").strip() if el else ""

    name        = await get_text("h1") or await get_text("[class*='product-name']")
    price       = await get_text("[class*='price']:not([class*='old'])") or await get_text("[itemprop='price']")
    sku         = await get_text("[class*='sku']") or await get_attr("[itemprop='sku']", "content")
    description = await get_text("[class*='description']") or await get_text("[itemprop='description']")
    brand       = await get_text("[class*='brand']") or await get_attr("[itemprop='brand']", "content")

    img_elements = await page.query_selector_all(
        "[class*='product'] img, [class*='gallery'] img, main img"
    )
    image_urls = []
    for img in img_elements:
        src = (
            await img.get_attribute("src") or
            await img.get_attribute("data-src") or
            await img.get_attribute("data-lazy") or ""
        )
        if src and src.startswith("http") and src not in image_urls:
            if not any(skip in src.lower() for skip in ["logo", "icon", "placeholder", "spinner"]):
                image_urls.append(src)

    return {
        "name":        name,
        "price":       price,
        "sku":         sku,
        "brand":       brand,
        "category":    entry["category"],
        "subcategory": entry["subcategory"],
        "label":       entry["label"],
        "description": description[:500].replace("\n", " ") if description else "",
        "image_urls":  "|".join(image_urls),
        "product_url": entry["product_url"],
    }


def download_all_images(products: list[dict]) -> list[dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; SeifScraper/1.0)"})
    records = []

    for product in tqdm(products, desc="Downloading images"):
        if not product.get("image_urls"):
            continue

        urls = [u for u in product["image_urls"].split("|") if u]
        slug = slugify(product.get("name") or "product")

        label_dir = IMAGES_DIR / product.get("label", "general_medicine")
        label_dir.mkdir(exist_ok=True)

        for idx, img_url in enumerate(urls):
            ext      = Path(urlparse(img_url).path).suffix or ".jpg"
            filename = f"{slug}_{idx + 1}{ext}"
            dest     = label_dir / filename
            success  = download_image(img_url, dest, session)

            records.append({
                "product_name": product.get("name", ""),
                "category":     product.get("category", ""),
                "subcategory":  product.get("subcategory", ""),
                "label":        product.get("label", ""),
                "image_index":  idx + 1,
                "image_url":    img_url,
                "local_file":   str(dest) if success else "",
                "downloaded":   success,
                "product_url":  product.get("product_url", ""),
            })

    return records


async def main():
    print("=" * 55)
    print("  seif_scraper_targeted.py")
    print("=" * 55)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        # step 1: discover real URLs on the site
        real_urls = await discover_real_urls(page)
        print(f"\nTotal URLs discovered: {len(real_urls)}")
        print("Check above list and update TARGET_CATEGORIES if needed.\n")

        # step 2: collect product links from target categories
        all_entries: list[dict] = []
        for cat in TARGET_CATEGORIES:
            print(f"Scraping: {cat['subcategory']} -> {cat['url']}")
            entries = await scrape_category(page, cat)
            all_entries.extend(entries)
            print(f"  Total so far: {len(all_entries)}")

        # deduplicate
        seen_urls: dict = {}
        deduped = []
        for e in all_entries:
            if e["product_url"] not in seen_urls:
                seen_urls[e["product_url"]] = True
                deduped.append(e)

        print(f"\nUnique products: {len(deduped)}")

        if not deduped:
            print("No products found.")
            print("Open the site and check the correct category URLs.")
            print("Update TARGET_CATEGORIES in this file then run again.")
            await browser.close()
            return

        # step 3: scrape product details
        products: list[dict] = []
        failed = 0
        for entry in tqdm(deduped, desc="Scraping products"):
            product = await scrape_product(page, entry)
            if product:
                products.append(product)
            else:
                failed += 1

        await browser.close()

    print(f"\nScraped: {len(products)} products ({failed} failed)")

    # step 4: save CSV
    df = pd.DataFrame(products)
    df.to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved -> {PRODUCTS_CSV}")

    print("\nLabel distribution:")
    for lbl, cnt in df["label"].value_counts().items():
        print(f"  {lbl:<22} {cnt}")

    # step 5: download images
    print("\nDownloading images...")
    records = download_all_images(products)
    df_img = pd.DataFrame(records)
    df_img.to_csv(IMAGES_CSV, index=False, encoding="utf-8-sig")
    total = df_img["downloaded"].sum()
    print(f"Downloaded: {total}/{len(records)} images -> {IMAGES_DIR}/")

    print("\n" + "=" * 55)
  
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())