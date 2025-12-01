#!/usr/bin/env python3
"""
embeddings/embedder.py - robust single-command generator for content pages + embeddings.

Behavior:
  1) FAST PATH: if `embeddings/embeddings.npy` AND `embeddings/docs.json` (or pages.json)
     exist, convert/copy them into `out_dir/embeddings.npy` + `out_dir/pages.json` (validates counts).
  2) Otherwise scan `content` (or provided raw dir), embed pages using either local
     sentence-transformers or google-genai (Genie/Gemini via google-genai) and write outputs.
  3) Optionally build FAISS index with --build-faiss.

Usage (single-line, Windows-friendly):
  python .\embeddings\embedder.py --out-dir .\content --build-faiss --force
"""
import os
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Any

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


def try_fastpath_copy(src_dir: Path, out_dir: Path, force: bool = False) -> bool:
    """
    Accept either:
      - embeddings/embeddings.npy + embeddings/docs.json
      - embeddings/embeddings.npy + embeddings/pages.json
    Validate row counts (embeddings rows == number of pages).
    If mismatch and not force -> return False and print helpful message.
    """
    src_emb = src_dir / "embeddings.npy"
    src_docs1 = src_dir / "docs.json"
    src_docs2 = src_dir / "pages.json"
    docs_file = None
    if not src_emb.exists():
        return False
    if src_docs1.exists():
        docs_file = src_docs1
    elif src_docs2.exists():
        docs_file = src_docs2
    else:
        # no docs file to map to embedding rows
        return False

    print(f"Fast-path: detected precomputed embeddings in {src_dir} and docs at {docs_file.name}")
    try:
        with open(docs_file, "r", encoding="utf-8") as f:
            docs_raw = json.load(f)
    except Exception as e:
        print("Failed to read docs file:", e)
        return False

    try:
        arr = np.load(src_emb, mmap_mode="r")
    except Exception as e:
        print("Failed to load embeddings.npy:", e)
        return False

    num_rows = int(arr.shape[0]) if arr.ndim >= 2 else None
    num_docs = len(docs_raw)
    if num_rows is None:
        print("Embeddings file shape is unexpected:", arr.shape)
        if not force:
            print("Use --force to overwrite outputs instead of fast-path.")
            return False
    else:
        if num_rows != num_docs:
            msg = f"Embeddings rows ({num_rows}) != docs count ({num_docs})."
            if not force:
                print(msg, "Fast-path aborted. Use --force to overwrite outputs or regenerate embeddings.")
                return False
            else:
                print(msg, "But --force given: fast-path will proceed to copy (be careful).")

    # convert docs_raw to pages.json format (robust)
    pages = []
    for i, d in enumerate(docs_raw):
        title = d.get("title") or d.get("name") or Path(d.get("path", "")).stem
        path = d.get("path", f"doc_{i}")
        snippet = d.get("snippet") or (d.get("text")[:2000] if d.get("text") else "")
        pages.append({"id": str(i), "path": path, "title": title, "url": d.get("url", ""), "snippet": snippet})

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
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=batch_size)
    embs = np.asarray(embs, dtype=np.float32)
    # normalize rows
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return embs


def _extract_numeric_from_genai_resp(item: Any) -> List[float]:
    """
    Aggressive extractor for numeric embedding from various google-genai SDK shapes.
    Returns a flat list of floats or raises RuntimeError with a helpful repr.
    """
    import re

    # 1) None
    if item is None:
        raise RuntimeError("Empty item")

    # 2) dict-like: common keys
    if isinstance(item, dict):
        for key in ("embedding", "embedding_vector", "values", "value", "vector", "data"):
            if key in item and item[key] is not None:
                cand = item[key]
                if hasattr(cand, "__iter__") and not isinstance(cand, (str, bytes)):
                    return [float(x) for x in list(cand)]
        # nested lists under known keys
        for key in ("data", "results", "candidates"):
            if key in item and isinstance(item[key], (list, tuple)) and len(item[key]) > 0:
                return _extract_numeric_from_genai_resp(item[key][0])

    # 3) wrapper objects with .embedding
    try:
        if hasattr(item, "embedding"):
            val = getattr(item, "embedding")
            # val might be list-like or another wrapper
            if val is None:
                pass
            elif hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                return [float(x) for x in list(val)]
            else:
                return _extract_numeric_from_genai_resp(val)
    except Exception:
        pass

    # 4) wrapper objects with .data (some SDKs)
    try:
        if hasattr(item, "data") and item.data:
            first = item.data[0]
            if hasattr(first, "embedding"):
                emb = getattr(first, "embedding")
                if hasattr(emb, "__iter__"):
                    return [float(x) for x in list(emb)]
            if isinstance(first, dict) and "embedding" in first:
                return [float(x) for x in first["embedding"]]
            return _extract_numeric_from_genai_resp(first)
    except Exception:
        pass

    # 5) plain list/tuple/ndarray
    if isinstance(item, (list, tuple, np.ndarray)):
        flat = []
        for el in item:
            if isinstance(el, (list, tuple, np.ndarray)):
                flat.extend([float(x) for x in list(el)])
            else:
                try:
                    flat.append(float(el))
                except Exception:
                    sub = None
                    try:
                        sub = _extract_numeric_from_genai_resp(el)
                    except Exception:
                        sub = None
                    if sub is None:
                        raise RuntimeError("Cannot convert nested element to float")
                    flat.extend(sub)
        if flat:
            return flat

    # 6) try .tolist(), .to_list(), .numpy()
    try:
        if hasattr(item, "tolist"):
            arr = item.tolist()
            if hasattr(arr, "__iter__"):
                return [float(x) for x in list(arr)]
    except Exception:
        pass
    try:
        if hasattr(item, "to_list"):
            arr = item.to_list()
            if hasattr(arr, "__iter__"):
                return [float(x) for x in list(arr)]
    except Exception:
        pass
    try:
        if hasattr(item, "numpy"):
            arr = item.numpy()
            if hasattr(arr, "__iter__"):
                return [float(x) for x in list(arr)]
    except Exception:
        pass

    # 7) last-resort: scan repr for long numeric list
    try:
        r = repr(item)
        m = re.search(r"\[[-0-9eE\.,\s]{50,}\]", r)
        if m:
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", m.group(0))]
            if nums:
                return nums
    except Exception:
        pass

    # failed
    raise RuntimeError("Unable to extract numeric embedding from GenAI response item. repr(item)[:400]=" + repr(item)[:400])


def embed_with_genai(model_name: str, texts: List[str], batch_size: int = 16) -> np.ndarray:
    """
    Use google-genai's modern SDK shape `models.embed_content(...)` when available.
    Returns normalized float32 array (N, D).
    """
    if not HAS_GENAI:
        raise RuntimeError("google-genai not installed")

    key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=key) if key else genai.Client()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # prefer models.embed_content
        try:
            resp = client.models.embed_content(model=model_name, contents=batch)
        except Exception as e:
            # some older SDKs might use client.embeddings.create(...)
            try:
                resp = client.embeddings.create(model=model_name, input=batch)
            except Exception as e2:
                raise RuntimeError(f"Embed API call failed for batch starting at {i}: {e}; fallback also failed: {e2}")

        # unify plausible shapes
        results = None
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

        # extract each
        for item in results:
            vec = _extract_numeric_from_genai_resp(item)
            out.append(vec)
        print(f"Computed embeddings {min(i + batch_size, len(texts))}/{len(texts)}")

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
    p.add_argument("--force", action="store_true", help="Overwrite outputs if exist / override fast-path checks")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)
    content_dir = Path(args.content_dir)

    # 1) FAST PATH: try copying precomputed embeddings/docs from 'embeddings'
    fast_src = Path("embeddings")
    if try_fastpath_copy(fast_src, out_dir, force=args.force):
        print("Fast-path complete. Done.")
        return

    # 2) Decide where to scan raw content
    scan_dir = raw_dir if raw_dir.exists() and any(raw_dir.rglob("*")) else content_dir
    if not scan_dir.exists():
        raise SystemExit(f"No raw content found in {raw_dir} or {content_dir}. Put your files there or use artifacts in embeddings/")

    print("Scanning raw content in:", scan_dir)
    docs = gather_documents(scan_dir)
    if not docs:
        raise SystemExit("No documents found in content dir. Put .txt/.md/.html files in the folder.")

    texts = [(d.get("title", "") + "\n" + d.get("text", ""))[:8000] for d in docs]

    # avoid accidental overwrite
    emb_path = out_dir / "embeddings.npy"
    pages_path = out_dir / "pages.json"
    if (emb_path.exists() or pages_path.exists()) and not args.force:
        print(f"Output {emb_path} or {pages_path} already exists. Use --force to overwrite.")
        return

    # choose backend and embed
    embeddings = None
    if args.use_genai:
        if not HAS_GENAI:
            raise SystemExit("google-genai not installed; cannot use --use-genai")
        print("Using google-genai for embeddings (ensure GEMINI_API_KEY set).")
        embeddings = embed_with_genai(args.model, texts)
    else:
        if HAS_S2:
            print("Using local sentence-transformers model:", args.model)
            embeddings = embed_with_sentence_transformers(args.model, texts)
        elif HAS_GENAI:
            print("sentence-transformers not found; falling back to google-genai.")
            embeddings = embed_with_genai(args.model, texts)
        else:
            raise SystemExit("No embedding backend available. Install sentence-transformers or configure google-genai+GEMINI_API_KEY.")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))
    pages_out = [{"id": d["id"], "path": d["path"], "title": d["title"], "url": d.get("url", ""), "snippet": d.get("snippet", "")[:2000]} for d in docs]
    with open(out_dir / "pages.json", "w", encoding="utf-8") as f:
        json.dump(pages_out, f, ensure_ascii=False, indent=2)
    print("Wrote:", out_dir / "embeddings.npy", "and", out_dir / "pages.json")

    # optional FAISS
    if args.build_faiss:
        try:
            build_faiss(embeddings, out_dir)
            # also write pages_meta.json for mapping
            with open(out_dir / "pages_meta.json", "w", encoding="utf-8") as f:
                json.dump(pages_out, f, ensure_ascii=False, indent=2)
            print("Saved pages_meta.json")
        except Exception as e:
            print("FAISS build failed:", e)

    print("Done — content generated. Now run your server (uvicorn main:app --host 0.0.0.0 --port $PORT).")


if __name__ == "__main__":
    main()
