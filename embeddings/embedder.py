"""
embeddings/embedder.py

Simple, robust embedding builder for your RAG project.

Features:
- Reads text/html files from a content directory (default: ./content)
- Converts them to embeddings using either:
  * sentence-transformers (default, local/offline-capable)
  * OpenAI embeddings (if OPENAI_API_KEY is set and --model is an OpenAI text-embedding model)
- Builds a FAISS index and saves it to disk under ./embeddings/
- Saves metadata (docs list) and raw embeddings for later use

Usage:
    python embeddings/embedder.py --content-dir ./content --persist-dir ./embeddings --model all-MiniLM-L6-v2

Dependencies (add to requirements.txt):
    sentence-transformers
    faiss-cpu
    numpy
    tqdm
    python-dotenv
    beautifulsoup4
    openai  # only if you want to use OpenAI embeddings

Outputs (by default written to persist-dir):
    index.faiss        # FAISS index file
    docs.json          # metadata for each vector (id, path, text snippet)
    embeddings.npy     # raw numpy array of embeddings (optional but useful)

Notes:
- The script is idempotent: running again overwrites the saved index and metadata.
- If your content contains HTML, the script extracts visible text using BeautifulSoup.
- For cosine similarity, vectors are normalized before adding to the FAISS IndexFlatIP index.

"""

import os
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Try imports that may not exist in every environment
try:
    from sentence_transformers import SentenceTransformer
    has_s2 = True
except Exception:
    has_s2 = False

try:
    import faiss
    has_faiss = True
except Exception:
    has_faiss = False

try:
    import openai
    has_openai = True
except Exception:
    has_openai = False


def load_text_from_file(path: Path) -> str:
    text = ""
    suffix = path.suffix.lower()
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(content, "html.parser")
        # remove scripts/styles
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        text = soup.get_text(separator=" ")
    else:
        text = content

    # collapse whitespace
    text = " ".join(text.split())
    return text


def gather_documents(content_dir: Path, exts=None) -> List[dict]:
    if exts is None:
        exts = {".txt", ".md", ".html", ".htm", ".json"}

    docs = []
    for p in sorted(content_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            txt = load_text_from_file(p)
            if not txt:
                continue
            docs.append({
                "id": str(p.resolve()),
                "path": str(p.relative_to(content_dir.parent)) if content_dir.parent in p.parents else str(p),
                "text": txt[:2000],  # store up to 2k chars as preview
            })
    return docs


def embed_with_sentence_transformers(texts: List[str], model_name: str):
    if not has_s2:
        raise RuntimeError("sentence-transformers not installed. Install with: pip install sentence-transformers")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    return embeddings


def embed_with_openai(texts: List[str], model_name: str):
    if not has_openai:
        raise RuntimeError("openai package is not installed. Install with: pip install openai")
    # openai expects per-input calls for many SDKs; batching depends on model
    out = []
    for t in tqdm(texts, desc="OpenAI embeddings"):
        resp = openai.Embedding.create(input=t, model=model_name)
        vec = resp["data"][0]["embedding"]
        out.append(vec)
    return np.asarray(out, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray):
    if not has_faiss:
        raise RuntimeError("faiss not installed. Install with: pip install faiss-cpu")

    dim = embeddings.shape[1]
    # Use inner-product on normalized vectors for cosine similarity
    index = faiss.IndexFlatIP(dim)
    # normalize
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def save_index(index, persist_dir: Path):
    idx_path = persist_dir / "index.faiss"
    faiss.write_index(index, str(idx_path))
    print(f"Saved FAISS index to {idx_path}")


def main():
    parser = argparse.ArgumentParser(description="Build embeddings and FAISS index for project content.")
    parser.add_argument("--content-dir", type=str, default="./content", help="Path to content directory")
    parser.add_argument("--persist-dir", type=str, default="./embeddings", help="Where to save index/metadata")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Embedding model name (sentence-transformers or openai model)")
    parser.add_argument("--use-openai", action="store_true", help="Force using OpenAI embeddings (requires OPENAI_API_KEY in env)")

    args = parser.parse_args()

    load_dotenv()

    content_dir = Path(args.content_dir)
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    if not content_dir.exists():
        raise SystemExit(f"Content directory not found: {content_dir}")

    docs = gather_documents(content_dir)
    if not docs:
        raise SystemExit("No documents found in content directory.")

    texts = [d["text"] for d in docs]

    embeddings = None

    # Decide embedding backend
    use_openai = args.use_openai or (has_openai and os.getenv("OPENAI_API_KEY") is not None and args.model.startswith("text-"))

    if use_openai:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        print("Using OpenAI embeddings (this will use your API key)")
        embeddings = embed_with_openai(texts, args.model)
    else:
        # default to sentence-transformers
        print("Using sentence-transformers (local model)")
        try:
            embeddings = embed_with_sentence_transformers(texts, args.model)
        except Exception as e:
            raise RuntimeError("Failed to embed with sentence-transformers: " + str(e))

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Build FAISS index
    index = build_faiss_index(embeddings.copy())

    # Save index and metadata
    # Save raw embeddings too (optional)
    np.save(persist_dir / "embeddings.npy", embeddings)
    with open(persist_dir / "docs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    save_index(index, persist_dir)

    print("Done. You can now load 'embeddings/index.faiss' and 'embeddings/docs.json' in your app.")


if __name__ == "__main__":
    main()
