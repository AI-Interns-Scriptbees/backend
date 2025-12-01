#!/usr/bin/env python3
"""
build_index.py - GenAI-only scraper + embedder (auto-sitemap).

Behavior:
  - If --sitemap is provided, use it.
  - Otherwise defaults to SITEMAP_DEFAULT ('https://scriptbees.com/sitemap.xml').
  - Expands sitemap index recursively to discover real page URLs.
  - Scrapes pages, extracts text, embeds with Google GenAI (no local models).
  - Writes under content/: pages.json, pages_meta.json, embeddings.npy, embeddings_info.json, pages.faiss (optional).

Usage:
  python build_index.py
  python build_index.py --sitemap https://example.com/sitemap.xml --max-pages 50

Environment:
  - GEMINI_API_KEY (optional if ADC available)
  - EMBED_MODEL (default: text-embedding-004)
"""
import os
import sys
import time
import argparse
import json
import requests
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup

# GenAI client
try:
    from google import genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

# Faiss (optional)
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

import numpy as np

# Default sitemap if none provided
SITEMAP_DEFAULT = os.getenv("SITEMAP_URL", "https://scriptbees.com/sitemap.xml")

# ---------- helper utilities ----------
def safe_get_text_from_html(html: str) -> str:
    bs = BeautifulSoup(html, "html.parser")
    for s in bs(["script", "style", "noscript"]):
        s.decompose()
    text = bs.get_text(separator=" ", strip=True)
    text = " ".join(text.split())
    return text

def safe_fetch(url: str, timeout: float = 10.0) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"build-index-bot/1.0"})
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  ⚠️ fetch error for {url}: {e}")
        return None

# ---------- sitemap expansion (recursive) ----------
def fetch_sitemap_links(sitemap_url: str, timeout: float = 10.0, max_depth: int = 4) -> List[str]:
    """
    Expand a sitemap index into all page URLs (recursively).
    """
    print("📡 Expanding sitemap:", sitemap_url)
    found = []
    seen = set()

    def _fetch(url: str, depth: int):
        if depth > max_depth:
            return
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent":"build-index-bot/1.0"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "xml")
        except Exception as e:
            print(f"  ⚠️ sitemap fetch error for {url}: {e}")
            return

        # sitemap index?
        sitemaps = soup.find_all("sitemap")
        if sitemaps:
            for sm in sitemaps:
                loc = sm.find("loc")
                if loc and loc.text:
                    child = loc.text.strip()
                    if child not in seen:
                        seen.add(child)
                        _fetch(child, depth + 1)
            return

        # url sitemap?
        urls = soup.find_all("url")
        if urls:
            for u in urls:
                loc = u.find("loc")
                if loc and loc.text:
                    page_url = loc.text.strip()
                    if page_url not in seen:
                        seen.add(page_url)
                        found.append(page_url)
            return

        # fallback: any <loc> tags
        locs = soup.find_all("loc")
        for loc in locs:
            if loc and loc.text:
                page_url = loc.text.strip()
                if page_url.startswith("http") and page_url not in seen:
                    seen.add(page_url)
                    found.append(page_url)
        return

    _fetch(sitemap_url, 0)
    print(f"→ Expanded sitemap to {len(found)} page URLs")
    return found

# ---------- genai embedding ----------
def extract_vector_from_genai_item(item):
    """Robustly extract numeric vector from various genai response shapes."""
    if item is None:
        raise RuntimeError("empty genai item")
    if isinstance(item, (list, tuple, np.ndarray)):
        return [float(x) for x in item]
    if isinstance(item, dict):
        for key in ("embedding", "values", "vector", "embedding_vector", "data"):
            if key in item and item[key] is not None:
                return extract_vector_from_genai_item(item[key])
        for key in ("data", "results", "candidates"):
            if key in item and isinstance(item[key], (list, tuple)) and item[key]:
                return extract_vector_from_genai_item(item[key][0])
    try:
        if hasattr(item, "embedding"):
            return extract_vector_from_genai_item(getattr(item, "embedding"))
        if hasattr(item, "values"):
            return extract_vector_from_genai_item(getattr(item, "values"))
        if hasattr(item, "data"):
            return extract_vector_from_genai_item(getattr(item, "data"))
    except Exception:
        pass
    # fallback: parse repr
    import re
    try:
        r = repr(item)
        m = re.search(r"\[[-0-9eE\.,\s]{50,}\]", r)
        if m:
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", m.group(0))]
            if nums:
                return nums
    except Exception:
        pass
    raise RuntimeError("Unable to extract numeric embedding from GenAI item. repr(item)[:400]=" + repr(item)[:400])

def embed_texts_genai(model_name: str, texts: List[str], client, batch_size: int = 16, sleep_between: float = 0.05):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = client.models.embed_content(model=model_name, contents=batch)
        except Exception as e:
            try:
                resp = client.embeddings.create(model=model_name, input=batch)
            except Exception as e2:
                raise RuntimeError(f"GenAI embed failed for batch {i}: {e}; fallback: {e2}")
        if hasattr(resp, "embeddings"):
            results = resp.embeddings
        elif isinstance(resp, dict) and "embeddings" in resp:
            results = resp["embeddings"]
        elif isinstance(resp, dict) and "data" in resp:
            results = resp["data"]
        elif hasattr(resp, "data"):
            results = resp.data
        elif isinstance(resp, (list, tuple)):
            results = resp
        else:
            results = [resp]
        for item in results:
            vec = extract_vector_from_genai_item(item)
            out.append(vec)
        print(f"  → embedded {min(i+batch_size, len(texts))}/{len(texts)}")
        time.sleep(sleep_between)
    arr = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sitemap", type=str, default="", help="Sitemap URL (index or sitemap)")
    parser.add_argument("--content-dir", type=str, default=os.getenv("CONTENT_DIR", "content"), help="Output content dir")
    parser.add_argument("--embed-model", type=str, default=os.getenv("EMBED_MODEL", "text-embedding-004"), help="GenAI embed model")
    parser.add_argument("--batch", type=int, default=int(os.getenv("GENAI_BATCH", "16")), help="GenAI embed batch size")
    parser.add_argument("--max-pages", type=int, default=0, help="Max pages to fetch (0 = all)")
    parser.add_argument("--sleep", type=float, default=float(os.getenv("SLEEP_BETWEEN_BATCHES", "0.05")), help="Sleep between genai batches")
    parser.add_argument("--timeout", type=float, default=float(os.getenv("REQUEST_TIMEOUT", "10")), help="HTTP request timeout")
    parser.add_argument("--no-faiss", action="store_true", help="Do not build faiss even if faiss installed")
    args = parser.parse_args()

    sitemap_url = args.sitemap.strip() or SITEMAP_DEFAULT
    content_dir = Path(args.content_dir)
    content_dir.mkdir(parents=True, exist_ok=True)
    embed_model = args.embed_model
    gemini_key = os.getenv("GEMINI_API_KEY")

    print("Using sitemap:", sitemap_url)
    print("GenAI embed model:", embed_model)

    if not HAS_GENAI:
        print("ERROR: google-genai not installed. pip install google-genai")
        sys.exit(2)

    client = genai.Client(api_key=gemini_key) if gemini_key else genai.Client()

    urls = fetch_sitemap_links(sitemap_url, timeout=args.timeout)
    if args.max_pages and args.max_pages > 0:
        urls = urls[: args.max_pages]
    print(f"Will scrape {len(urls)} pages")

    pages = []
    pages_meta = []
    texts = []

    for idx, url in enumerate(urls):
        print(f"Scraping ({idx+1}/{len(urls)}): {url}")
        html = safe_fetch(url, timeout=args.timeout)
        if not html:
            print("  (skipped - no html)")
            continue
        text = safe_get_text_from_html(html)
        if not text:
            print("  (skipped - empty text after cleaning)")
            continue
        title = ""
        try:
            bs = BeautifulSoup(html, "html.parser")
            title = bs.title.string.strip() if bs.title and bs.title.string else url
        except Exception:
            title = url
        pages.append({"id": str(len(pages)), "path": url, "text": text})
        pages_meta.append({"id": str(len(pages_meta)), "url": url, "title": title})
        texts.append(text)

    if not texts:
        print("No page texts collected. Exiting.")
        sys.exit(0)

    print("Embedding pages... using GenAI model:", embed_model)
    embeddings = embed_texts_genai(embed_model, texts, client, batch_size=args.batch, sleep_between=args.sleep)

    if embeddings.ndim != 2:
        raise RuntimeError("Embeddings shape unexpected: " + str(embeddings.shape))
    N_emb, D_emb = embeddings.shape
    N_pages = len(texts)
    if N_emb != N_pages:
        raise RuntimeError(f"Embeddings rows ({N_emb}) != pages count ({N_pages})")

    embed_npy = content_dir / "embeddings.npy"
    pages_json = content_dir / "pages.json"
    pages_meta_json = content_dir / "pages_meta.json"
    embed_meta = content_dir / "embeddings_info.json"
    faiss_path = content_dir / "pages.faiss"

    np.save(embed_npy, embeddings.astype(np.float32))
    with open(pages_json, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    with open(pages_meta_json, "w", encoding="utf-8") as f:
        json.dump(pages_meta, f, ensure_ascii=False, indent=2)

    meta_obj = {"model": embed_model, "dim": int(D_emb), "count": int(N_emb)}
    with open(embed_meta, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False, indent=2)

    print("Wrote embeddings_info:", meta_obj)
    print("Saved embeddings.npy ->", embed_npy)
    print("Saved pages.json ->", pages_json)
    print("Saved pages_meta.json ->", pages_meta_json)

    if not args.no_faiss and HAS_FAISS:
        try:
            dim = int(D_emb)
            index = faiss.IndexFlatIP(dim)
            embs_copy = embeddings.copy()
            faiss.normalize_L2(embs_copy)
            index.add(embs_copy)
            faiss.write_index(index, str(faiss_path))
            print("Saved FAISS index to", faiss_path)
        except Exception as e:
            print("Faiss build failed:", repr(e))
    else:
        print("Skipping FAISS build (no-faiss or faiss not installed)")

    print("DONE! Your RAG dataset is ready.")

if __name__ == "__main__":
    main()
