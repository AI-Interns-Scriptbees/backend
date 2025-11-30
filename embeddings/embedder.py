#!/usr/bin/env python3
"""
embeddings/embedder.py

Precompute embeddings and produce:
 - <out_dir>/pages.json      (metadata: id, path, title, url, snippet)
 - <out_dir>/embeddings.npy  (float32, one row per document)

Behavior:
 - If an existing pair embeddings/embeddings.npy + embeddings/docs.json is detected,
   the script will auto-copy/convert those into the out-dir and exit (fast path).
 - Otherwise, the script will scan --content-dir (default ./content) and compute
   embeddings using google-genai (requires GEMINI_API_KEY / google-genai installed).
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional
import shutil

import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# optional google-genai client (for Gemini embeddings)
try:
    from google import genai
except Exception:
    genai = None

# optional faiss (only used if --build-faiss)
try:
    import faiss
except Exception:
    faiss = None

SUPPORTED_EXTS = {".txt", ".md", ".html", ".htm", ".json"}


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


def gather_documents_from_dir(content_dir: Path, exts: Optional[set] = None) -> List[dict]:
    if exts is None:
        exts = SUPPORTED_EXTS
    docs = []
    for p in sorted(content_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            txt = load_text_from_file(p)
            if not txt:
                continue
            docs.append({
                "id": str(len(docs)),
                "path": str(p.relative_to(content_dir)),
                "title": p.stem,
                "url": "",
                "text": txt,
                "snippet": txt[:2000]
            })
    return docs


def compute_embeddings_genai(client, model_name: str, texts: List[str], batch_size: int = 16):
    out = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model_name, input=batch)
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
    index = faiss.IndexFlatIP(dim)
    if normalize:
        faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, str(out_dir / "pages.faiss"))
    print("Saved FAISS index to", out_dir / "pages.faiss")


def try_fastpath_copy(existing_dir: Path, out_dir: Path) -> bool:
    """
    If embeddings/embeddings.npy and embeddings/docs.json exist, copy/convert them into out_dir.
    Returns True if fast-path succeeded and script can exit.
    """
    src_emb = existing_dir / "embeddings.npy"
    src_docs = existing_dir / "docs.json"
    if not src_emb.exists() or not src_docs.exists():
        return False

    print("Detected existing embeddings at:", existing_dir)
    print("Using fast-path: copying existing embeddings/docs into out-dir.")

    # load docs.json and convert to pages.json format
    try:
        with open(src_docs, "r", encoding="utf-8") as f:
            docs_raw = json.load(f)
    except Exception as e:
        print("Failed to read docs.json:", e)
        return False

    # load embeddings to validate shape (without loading everything into RAM - small files okay)
    try:
        emb = np.load(src_emb, mmap_mode="r")
    except Exception as e:
        print("Failed to load embeddings.npy for validation:", e)
        return False

    n_docs = len(docs_raw) if isinstance(docs_raw, list) else 0
    n_emb_rows = emb.shape[0] if emb.ndim == 2 else (emb.size // emb.shape[0] if emb.size else 0)

    if n_docs != n_emb_rows:
        print(f"Warning: docs.json length ({n_docs}) != embeddings rows ({n_emb_rows}). Proceeding but check order.")
    # build pages list from docs_raw trying to be robust
    pages_out = []
    for i, d in enumerate(docs_raw):
        # try multiple possible keys to populate snippet/text/title/url
        snippet = d.get("snippet") or (d.get("text")[:2000] if d.get("text") else "") or d.get("content","")[:2000] if d.get("content") else ""
        title = d.get("title") or d.get("name") or Path(d.get("path","")).stem if d.get("path") else f"doc_{i}"
        path = d.get("path", f"doc_{i}")
        url = d.get("url", "")
        pages_out.append({
            "id": str(i),
            "path": path,
            "title": title,
            "url": url,
            "snippet": snippet
        })

    # ensure out_dir exists
    out_dir.mkdir(parents=True, exist_ok=True)
    # copy embeddings file
    dst_emb = out_dir / "embeddings.npy"
    try:
        shutil.copyfile(src_emb, dst_emb)
        print("Copied embeddings:", dst_emb)
    except Exception as e:
        print("Failed to copy embeddings.npy:", e)
        return False

    # write pages.json
    dst_pages = out_dir / "pages.json"
    try:
        with open(dst_pages, "w", encoding="utf-8") as f:
            json.dump(pages_out, f, ensure_ascii=False, indent=2)
        print("Wrote pages.json:", dst_pages)
    except Exception as e:
        print("Failed to write pages.json:", e)
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-dir", type=str, default="./content",
                        help="Directory with raw content files (default: ./content)")
    parser.add_argument("--input-pages", type=str, default="",
                        help="Path to a JSON file with prebuilt pages list (overrides content-dir)")
    parser.add_argument("--out-dir", type=str, default="./content",
                        help="Directory to write pages.json and embeddings.npy (default: ./content)")
    parser.add_argument("--embed-model", type=str, default="embed-gecko-001",
                        help="Gemini embedding model (e.g. embed-gecko-001)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding API")
    parser.add_argument("--build-faiss", action="store_true", help="Also build FAISS index (requires faiss-cpu)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs if present")
    args = parser.parse_args()

    load_dotenv()

    content_dir = Path(args.content_dir)
    out_dir = Path(args.out_dir)
    # fastpath source folder (where older artifacts may live)
    existing_dir = Path("embeddings")

    # If existing artifacts are present, copy/convert them and exit (fast-path)
    fast_ok = try_fastpath_copy(existing_dir, out_dir)
    if fast_ok:
        print("Fast-path copy complete. Exiting.")
        return

    # Otherwise proceed with normal precompute
    out_dir.mkdir(parents=True, exist_ok=True)

    # load pages either from input JSON or by scanning directory
    if args.input_pages:
        input_path = Path(args.input_pages)
        if not input_path.exists():
            raise SystemExit(f"Input pages JSON not found: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            pages_raw = json.load(f)
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
        if not content_dir.exists():
            raise SystemExit(f"Content directory not found: {content_dir}")
        docs = gather_documents_from_dir(content_dir)

    if not docs:
        raise SystemExit("No documents found to embed. Ensure files exist with extensions: .txt .md .html .htm .json")

    texts = [(d.get("title", "") + "\n" + d.get("text", ""))[:8000] for d in docs]

    embeddings_path = out_dir / "embeddings.npy"
    pages_out_path = out_dir / "pages.json"

    if embeddings_path.exists() or pages_out_path.exists():
        if not args.force:
            print(f"Warning: {embeddings_path} or {pages_out_path} already exist.")
            print("Run again with --force to overwrite, or move/delete the existing files.")
            return

    # gemini genai client (requires google-genai)
    if genai is None:
        raise SystemExit("google-genai is not installed. pip install google-genai")

    gemini_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_key) if gemini_key else genai.Client()

    print("Computing embeddings with model:", args.embed_model)
    embeddings = compute_embeddings_genai(client, args.embed_model, texts, batch_size=args.batch_size)

    # normalize rows (unit vectors)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # Save embeddings and pages metadata
    np.save(embeddings_path, embeddings.astype(np.float32))
    print("Wrote", embeddings_path)

    pages_out = []
    for d in docs:
        pages_out.append({
            "id": d["id"],
            "path": d["path"],
            "title": d["title"],
            "url": d.get("url", ""),
            "snippet": d.get("snippet", "")[:2000]
        })
    with open(pages_out_path, "w", encoding="utf-8") as f:
        json.dump(pages_out, f, ensure_ascii=False, indent=2)
    print("Wrote", pages_out_path)

    if args.build_faiss:
        try:
            build_faiss_index_and_save(embeddings.copy(), out_dir, normalize=False)
            with open(out_dir / "pages_meta.json", "w", encoding="utf-8") as f:
                json.dump(pages_out, f, ensure_ascii=False, indent=2)
            print("Saved pages_meta.json and pages.faiss in", out_dir)
        except Exception as e:
            print("Failed to build FAISS index:", e)

    print("Embedding precompute complete. Ensure the out-dir (content) is available to your app.")


if __name__ == "__main__":
    main()
