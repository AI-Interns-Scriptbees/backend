#!/usr/bin/env python3
"""
embeddings/embedder.py

Precompute embeddings using Google GenAI (Gemini) and save:
 - content/pages.json      (metadata with id,url,title, snippet)
 - content/embeddings.npy  (float32 embeddings, one row per document)
Optionally also build a FAISS index (requires faiss-cpu installed).

Run this on a machine with enough RAM.
Usage:
  python embeddings/embedder.py --input-pages pages_raw.json --out-dir ../content --build-faiss
"""
import os
import json
import argparse
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# GenAI client
try:
    from google import genai
except Exception:
    genai = None

# optional heavy deps (faiss)
try:
    import faiss
except Exception:
    faiss = None


def load_text_from_file(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(content, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        text = soup.get_text(separator=" ")
    else:
        text = content
    text = " ".join(text.split())
    return text


def gather_documents_from_dir(content_dir: Path, exts=None) -> List[dict]:
    if exts is None:
        exts = {".txt", ".md", ".html", ".htm", ".json"}
    docs = []
    for p in sorted(content_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            txt = load_text_from_file(p)
            if not txt:
                continue
            docs.append({
                "id": str(len(docs)),            # numeric id (0..n-1) to match embedding order
                "path": str(p.relative_to(content_dir)),
                "title": p.stem,
                "url": "",                       # optional: fill if you have actual URLs mapping
                "text": txt,
                "snippet": txt[:2000]
            })
    return docs


def compute_embeddings_genai(client, model_name: str, texts: List[str], batch_size: int = 16):
    """
    Use google-genai client to create embeddings for texts.
    Returns numpy array shape (n, dim) dtype float32.
    """
    out = []
    i = 0
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i:i+batch_size]
        # The genai embeddings API expects input as a list
        resp = client.embeddings.create(model=model_name, input=batch)
        # The response format: resp.data -> list of {embedding: [...]}
        # Support both attribute and dict access.
        if hasattr(resp, "data"):
            for item in resp.data:
                vec = item.embedding if hasattr(item, "embedding") else item.get("embedding")
                out.append(vec)
        else:
            for item in resp.get("data", []):
                out.append(item.get("embedding"))
        print(f"Computed embeddings {min(i+batch_size, n)}/{n}")
    arr = np.asarray(out, dtype=np.float32)
    return arr


def build_faiss_index_and_save(embeddings: np.ndarray, out_dir: Path, normalize=True):
    if faiss is None:
        raise RuntimeError("faiss is not installed. Install faiss-cpu on the precompute machine.")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (use L2-normalized vectors for cosine)
    if normalize:
        faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, str(out_dir / "pages.faiss"))
    print("Saved FAISS index to", out_dir / "pages.faiss")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-dir", type=str, default="./content_raw", help="Directory with raw content files")
    parser.add_argument("--input-pages", type=str, default="", help="Alternatively: prebuilt pages_raw.json")
    parser.add_argument("--out-dir", type=str, default="./content", help="Output directory for pages.json and embeddings.npy")
    parser.add_argument("--embed-model", type=str, default="embed-gecko-001", help="Gemini embedding model")
    parser.add_argument("--build-faiss", action="store_true", help="Also build pages.faiss (requires faiss-cpu)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding API calls")
    args = parser.parse_args()

    load_dotenv()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pages list
    if args.input_pages:
        with open(args.input_pages, "r", encoding="utf-8") as f:
            pages_raw = json.load(f)
        # Expect pages_raw to be list of dicts with 'text' and optional 'title','url'
        docs = []
        for i, p in enumerate(pages_raw):
            text = p.get("text") or p.get("content") or ""
            if not text:
                continue
            docs.append({
                "id": str(i),
                "path": p.get("path", f"doc_{i}"),
                "title": p.get("title", "")[:200],
                "url": p.get("url", ""),
                "text": text,
                "snippet": text[:2000]
            })
    else:
        content_dir = Path(args.content_dir)
        if not content_dir.exists():
            raise SystemExit(f"Content directory not found: {content_dir}")
        docs = gather_documents_from_dir(content_dir)

    if not docs:
        raise SystemExit("No documents found to embed.")

    # Prepare texts for embedding
    texts = [(d.get("title","") + "\n" + d.get("text",""))[:8000] for d in docs]

    # Create genai client
    if genai is None:
        raise SystemExit("google-genai is not installed. pip install google-genai")
    # Use API key env or ADC
    gemini_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_key) if gemini_key else genai.Client()

    print("Computing embeddings with model:", args.embed_model)
    embeddings = compute_embeddings_genai(client, args.embed_model, texts, batch_size=args.batch_size)

    # Normalize and save numpy memmap (unit vectors recommended)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # Save embeddings as .npy (float32)
    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))
    print("Wrote", out_dir / "embeddings.npy")

    # Save pages.json containing minimal metadata (id order must match embeddings rows)
    pages_out = []
    for d in docs:
        pages_out.append({
            "id": d["id"],
            "path": d["path"],
            "title": d["title"],
            "url": d.get("url", ""),
            "snippet": d.get("snippet", "")[:2000]
        })
    with open(out_dir / "pages.json", "w", encoding="utf-8") as f:
        json.dump(pages_out, f, ensure_ascii=False, indent=2)
    print("Wrote", out_dir / "pages.json")

    # Optionally build FAISS index (for local experiments)
    if args.build_faiss:
        if faiss is None:
            print("faiss not available -- skipping FAISS build. Install faiss-cpu to enable.")
        else:
            build_faiss_index_and_save(embeddings.copy(), out_dir, normalize=False)
            # save pages_meta.json for mapping index -> doc
            with open(out_dir / "pages_meta.json", "w", encoding="utf-8") as f:
                json.dump(pages_out, f, ensure_ascii=False, indent=2)
            print("Saved pages_meta.json and pages.faiss in", out_dir)

    print("Embedding precompute complete. Copy content/ to your Render repo (or upload to storage).")


if __name__ == "__main__":
    main()
