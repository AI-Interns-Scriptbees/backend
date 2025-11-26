"""
SCRIPTBEES SITEMAP SCRAPER
Fully working version for Next.js website.
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

START_SITEMAP = "https://scriptbees.com/sitemap.xml"
OUTPUT_DIR = "content"
DELAY = 0.8
MAX_PAGES = 200   # ScriptBees has many pages

# -----------------------------------------------------
# XML SITEMAP PARSER
# -----------------------------------------------------

def extract_sitemap_urls(xml_text):
    """Extract all <loc> URLs from any sitemap XML."""
    soup = BeautifulSoup(xml_text, "xml")
    urls = [loc.get_text() for loc in soup.find_all("loc")]
    return urls

def fetch_sitemap_urls():
    """Fetch root sitemap and all child sitemaps."""
    print("🔍 Fetching main sitemap:", START_SITEMAP)

    xml = requests.get(START_SITEMAP, timeout=10).text
    sitemap_urls = extract_sitemap_urls(xml)

    all_urls = []

    for sm in sitemap_urls:
        if sm.endswith(".xml"):
            print("📄 Fetching sub-sitemap:", sm)
            sub_xml = requests.get(sm, timeout=10).text
            page_urls = extract_sitemap_urls(sub_xml)
            all_urls.extend(page_urls)

    # Remove duplicates
    all_urls = list(dict.fromkeys(all_urls))

    print(f"🟡 Found {len(all_urls)} URLs in sitemap")
    return all_urls

# -----------------------------------------------------
# HTML PAGE SCRAPER
# -----------------------------------------------------

def clean_text(text):
    return " ".join(text.split())

def scrape_page(url):
    """Extract readable text from an HTML page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)

        if "text/html" not in r.headers.get("content-type", ""):
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        # remove scripts/css
        for bad in soup(["script", "style", "nav", "footer", "header"]):
            bad.decompose()

        text = clean_text(soup.get_text(separator=" "))

        if len(text) < 100:
            return None

        title = soup.title.string.strip() if soup.title else "Untitled"

        return {
            "url": url,
            "title": title,
            "text": text[:8000]
        }
    except:
        return None

# -----------------------------------------------------
# MAIN SCRAPING PIPELINE
# -----------------------------------------------------

def scrape_website():
    print("\n============================================================")
    print("🕷️  SCRIPTBEES SITEMAP SCRAPER")
    print("============================================================\n")

    urls = fetch_sitemap_urls()

    scraped = []

    pbar = tqdm(total=min(MAX_PAGES, len(urls)), desc="Scraping", unit="page")

    for url in urls[:MAX_PAGES]:
        result = scrape_page(url)
        if result:
            scraped.append(result)
            pbar.update(1)

        time.sleep(DELAY)

    pbar.close()

    print(f"\n✓ Scraped {len(scraped)} pages")
    return scraped

# -----------------------------------------------------
# SAVE RESULTS
# -----------------------------------------------------

def save_pages(pages):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "pages.json")

    # Add ID field
    for i, p in enumerate(pages):
        p["id"] = i

    with open(path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved {len(pages)} pages → {path}\n")

# -----------------------------------------------------
# RUN
# -----------------------------------------------------

def main():
    pages = scrape_website()
    save_pages(pages)

    print("Next step:")
    print("  python embeddings/embedder.py\n")

if __name__ == "__main__":
    main()
