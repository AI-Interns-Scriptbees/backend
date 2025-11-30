#!/usr/bin/env python3
"""
embeddings/embedder.py

One-command tool to produce the files your server needs:
  - content/pages.json   (list of docs)
  - content/embeddings.npy  (float32 matrix: rows == number of pages)

Behavior (in order):
  1) FAST PATH: if `embeddings/embeddings.npy` AND `embeddings/docs.json` exist,
     convert/copy them into `content/embeddings.npy` + `content/pages.json` and exit.
  2) SCAN PATH: scan a directory of raw files (default: ./embeddings/content or ./content),
     build pages list, then:
       a) prefer local sentence-transformers (offline) if installed
       b) else use google-genai (Gemini) if installed and GEMINI_API_KEY set
  3) Save outputs to ./content and optionally build FAISS index if --build-faiss.

Usage (from project root):
  python .\embeddings\embedder.py                # default behavior (fast-path first)
  python .\embeddings\embedder.py --force       # overwrite existing content/* outputs
  python .\embeddings\embedder.py --build-faiss # also create content/pages.faiss

Notes:
 - The script is defensive and prints helpful errors.
 - For local use it's best to have sentence-transformers installed (pip install sentence-transformers faiss-cpu).
"""
import os
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from bs4 import BeautifulSoup

# optional backends
try:
    from sentence_transformers import SentenceTransformer
    HAS_S2 = True
except Exception:
    HAS_S2 = False

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    from google import genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

SUPPORTED_EXTS = {".txt", ".md", ".html", ".htm", ".json"}


def load_text_from_file(p: Path) -> str:
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if p.suffix.lower() in {".html", ".htm"}:
        soup = BeautifulSoup(raw, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        txt = soup.get_text(separator=" ")
    else:
        txt = raw
    txt = " ".join(txt.split())
    return txt


def gather_documents(content_dir: Path) -> List[dict]:
    docs = []
    if not content_dir.exists():
        return docs
    for p in sorted(content_dir.rglob("*")):
        if not p.is_file(): 
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        t = load_text_from_file(p)
        if not t:
            continue
        docs.append({
            "id": str(len(docs)),
            "path": str(p.relative_to(content_dir)),
            "title": p.stem,
            "url": "",
            "text": t,
            "snippet": t[:2000],
        })
    return docs


def try_fastpath_copy(src_dir: Path, out_dir: Path) -> bool:
    """
    If embeddings/embeddings.npy and embeddings/docs.json exist, copy/convert them into out_dir.
    Returns True if copied successfully.
    """
    src_emb = src_dir / "embeddings.npy"
    src_docs = src_dir / "docs.json"
    if not (src_emb.exists() and src_docs.exists()):
        return False
    print("Fast-path: detected precomputed embeddings/docs in", src_dir)
    try:
        with open(src_docs, "r", encoding="utf-8") as f:
            docs_raw = json.load(f)
    except Exception as e:
        print("Failed to read docs.json:", e)
        return False

    try:
        arr = np.load(src_emb, mmap_mode="r")
    except Exception as e:
        print("Failed to load embeddings.npy:", e)
        return False

    # convert docs_raw to pages.json format (robust)
    pages = []
    for i, d in enumerate(docs_raw):
        title = d.get("title") or d.get("name") or Path(d.get("path","")).stem
        path = d.get("path", f"doc_{i}")
        snippet = d.get("snippet") or (d.get("text")[:2000] if d.get("text") else "")
        pages.append({"id": str(i), "path": path, "title": title, "url": d.get("url",""), "snippet": snippet})

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(src_emb, out_dir / "embeddings.npy")
        with open(out_dir / "pages.json", "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)
        print("Copied embeddings and wrote pages.json to", out_dir)
        return True
    except Exception as e:
        print("Failed to copy/write fast-path outputs:", e)
        return False


def embed_with_sentence_transformers(model_name: str, texts: List[str], batch_size: int = 32) -> np.ndarray:
    if not HAS_S2:
        raise RuntimeError("sentence-transformers not installed")
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embs = np.asarray(embs, dtype=np.float32)
    # normalize rows
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return embs


def embed_with_genai(model_name: str, texts: List[str], batch_size: int = 16) -> np.ndarray:
    if not HAS_GENAI:
        raise RuntimeError("google-genai not installed")
    key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=key) if key else genai.Client()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model_name, input=batch)
        # handle response shapes
        if hasattr(resp, "data"):
            for item in resp.data:
                vec = getattr(item, "embedding", None) or item.get("embedding")
                out.append(vec)
        else:
            for item in resp.get("data", []):
                out.append(item.get("embedding"))
        print(f"Computed embeddings {min(i+batch_size, len(texts))}/{len(texts)}")
    arr = np.asarray(out, dtype=np.float32)
    # normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr


def build_faiss(embeddings: np.ndarray, out_dir: Path):
    if not HAS_FAISS:
        print("faiss not installed; skipping FAISS build")
        return
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    embeddings_copy = embeddings.copy()
    faiss.normalize_L2(embeddings_copy)
    index.add(embeddings_copy)
    faiss.write_index(index, str(out_dir / "pages.faiss"))
    print("Saved FAISS index to", out_dir / "pages.faiss")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--content-dir", type=str, default="./content", help="Raw content dir to scan")
    p.add_argument("--raw-dir", type=str, default="./embeddings/content", help="Alternate location of raw files")
    p.add_argument("--out-dir", type=str, default="./content", help="Where to write pages.json + embeddings.npy")
    p.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Local embed model (sentence-transformers) or genai model name")
    p.add_argument("--use-genai", action="store_true", help="Force using google-genai for embeddings")
    p.add_argument("--build-faiss", action="store_true", help="Also build FAISS index (optional)")
    p.add_argument("--force", action="store_true", help="Overwrite outputs if exist")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)
    content_dir = Path(args.content_dir)

    # 1) FAST PATH: copy existing embeddings/docs.json if present
    fast_src = Path("embeddings")
    if try_fastpath_copy(fast_src, out_dir):
        print("Fast-path complete. Done.")
        return

    # 2) Determine where raw files live (prefer embeddings/content, then content/)
    scan_dir = raw_dir if raw_dir.exists() and any(raw_dir.rglob("*")) else content_dir
    if not scan_dir.exists():
        raise SystemExit(f"No raw content found in {raw_dir} or {content_dir}. Put your text/html files there or use fast-path artifacts.")

    print("Scanning raw content in:", scan_dir)
    docs = gather_documents(scan_dir)
    if not docs:
        raise SystemExit("No documents found in content dir. Put .txt/.md/.html files in the folder.")

    texts = [(d.get("title","") + "\n" + d.get("text",""))[:8000] for d in docs]

    # If outputs exist and not forced, abort to avoid accidental overwrite
    emb_path = out_dir / "embeddings.npy"
    pages_path = out_dir / "pages.json"
    if (emb_path.exists() or pages_path.exists()) and not args.force:
        print(f"Output {emb_path} or {pages_path} already exists. Use --force to overwrite.")
        return

    # Choose embedding backend
    embeddings = None
    if args.use_genai:
        if not HAS_GENAI:
            raise SystemExit("google-genai not installed; cannot use --use-genai")
        print("Using google-genai for embeddings (make sure GEMINI_API_KEY is set).")
        embeddings = embed_with_genai(args.model, texts)
    else:
        # prefer local sentence-transformers if available
        if HAS_S2:
            print("Using local sentence-transformers model:", args.model)
            embeddings = embed_with_sentence_transformers(args.model, texts)
        elif HAS_GENAI:
            print("sentence-transformers not found; falling back to google-genai embeddings.")
            embeddings = embed_with_genai(args.model, texts)
        else:
            raise SystemExit("No embedding backend available. Install sentence-transformers for local embeddings or set up google-genai + GEMINI_API_KEY.")

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))
    pages_out = [{"id": d["id"], "path": d["path"], "title": d["title"], "url": d.get("url",""), "snippet": d.get("snippet","")[:2000]} for d in docs]
    with open(out_dir / "pages.json", "w", encoding="utf-8") as f:
        json.dump(pages_out, f, ensure_ascii=False, indent=2)
    print("Wrote:", out_dir / "embeddings.npy", "and", out_dir / "pages.json")

    if args.build_faiss:
        try:
            build_faiss(embeddings, out_dir)
            # also write pages_meta.json for mapping
            with open(out_dir / "pages_meta.json", "w", encoding="utf-8") as f:
                json.dump(pages_out, f, ensure_ascii=False, indent=2)
            print("Saved pages_meta.json")
        except Exception as e:
            print("FAISS build failed:", e)

    print("Done — content generated. Now run your server (uvicorn main:app ...).")


if __name__ == "__main__":
    main()
