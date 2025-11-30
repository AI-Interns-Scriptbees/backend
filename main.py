#!/usr/bin/env python3
"""
main.py - Lightweight ScriptBees Gemini memmap server suitable for 512MB Render.

Requirements (on Render):
  - google-genai
  - fastapi
  - uvicorn
  - numpy
  - python-dotenv
  - httpx (optional)
  - pydantic
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import json
import hashlib
import logging
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Security, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# GenAI client
try:
    from google import genai
except Exception:
    genai = None

# Load .env if present (local dev)
def find_env():
    cur = Path(__file__).resolve().parent
    for _ in range(5):
        if (cur / ".env").exists():
            return cur / ".env"
        cur = cur.parent
    return None

env = find_env()
if env:
    load_dotenv(env)

# Config from env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_KEY = os.getenv("RAG_APIKey".upper(), os.getenv("RAG_API_KEY", "change-me"))  # support both RAG_API_KEY or rag_api_key
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "*")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embed-gecko-001")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-mini")
TOP_K = int(os.getenv("TOP_K", "1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

CONTENT_DIR = Path("content")
PAGES_FILE = CONTENT_DIR / "pages.json"
EMBED_FILE = CONTENT_DIR / "embeddings.npy"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scriptbees-memmap")

# FastAPI app & CORS
app = FastAPI(title="ScriptBees — small memmap Gemini")
_allow_credentials = False if FRONTEND_ORIGINS.strip() == "*" else True
origins = [o.strip() for o in FRONTEND_ORIGINS.split(",")] if FRONTEND_ORIGINS != "*" else ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=_allow_credentials, allow_methods=["*"], allow_headers=["*"])

# Models
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)

class Source(BaseModel):
    url: str
    title: str
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved: List[Source]
    cached: bool
    response_time_seconds: float

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
def verify_api_key(req: Request, key: str = Security(api_key_header)):
    incoming = key or ""
    if not incoming:
        auth = req.headers.get("authorization", "")
        if auth and auth.lower().startswith("bearer "):
            incoming = auth.split(" ", 1)[1]
    if incoming != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return incoming

# Cache util
def ckey(q): return hashlib.md5(q.lower().encode()).hexdigest()
cache = {}

# Load pages metadata (small)
if not PAGES_FILE.exists() or not EMBED_FILE.exists():
    logger.error("Missing content files. Ensure content/pages.json and content/embeddings.npy exist.")
    raise SystemExit("Missing content files. Run precompute/embedder script first.")

with open(PAGES_FILE, "r", encoding="utf-8") as f:
    pages = json.load(f)
num_pages = len(pages)

# Memmap loader robustly determines shape
def load_memmap_embeddings(path: Path, n_rows: int):
    # Try to load with -1 dim; numpy memmap requires known shape, so try to infer dim
    # Load header-less to infer size
    # Approach: load raw file size and deduce dimension
    import os as _os
    size_bytes = _os.path.getsize(str(path))
    # float32 => 4 bytes
    total_floats = size_bytes // 4
    if total_floats % n_rows != 0:
        raise RuntimeError("Embeddings file size incompatible with number of pages.")
    dim = total_floats // n_rows
    mm = np.memmap(str(path), dtype=np.float32, mode="r", shape=(n_rows, dim))
    return mm

embeddings = load_memmap_embeddings(EMBED_FILE, num_pages)
emb_dim = embeddings.shape[1]
logger.info(f"Loaded {num_pages} pages, embedding dim {emb_dim} (memmap)")

# Create genai client
if genai is None:
    raise SystemExit("google-genai client not installed. pip install google-genai")

if GEMINI_API_KEY:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    genai_client = genai.Client()

# Helpers
def get_embedding_for_text(text: str):
    resp = genai_client.embeddings.create(model=EMBED_MODEL, input=text)
    # parse robustly
    if hasattr(resp, "data"):
        vec = resp.data[0].embedding
    else:
        vec = resp.get("data", [])[0].get("embedding")
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr

def top_k_search(query_vec: np.ndarray, k: int = 1):
    q = query_vec.reshape(-1)
    best_scores = np.full(k, -np.inf, dtype=np.float32)
    best_idxs = np.full(k, -1, dtype=np.int32)
    # chunk to avoid large RAM spikes
    chunk = 2048
    for i in range(0, num_pages, chunk):
        block = embeddings[i: i+chunk]   # memmap slice
        # block assumed normalized on precompute; otherwise normalize here
        dots = block.dot(q)
        # update best
        for j, score in enumerate(dots):
            if score > best_scores.min():
                pos = int(best_scores.argmin())
                best_scores[pos] = float(score)
                best_idxs[pos] = int(i + j)
    order = np.argsort(-best_scores)
    results = []
    for pos in order:
        idx = int(best_idxs[pos])
        if idx == -1:
            continue
        results.append((idx, float(best_scores[pos])))
    return results

def generate_answer_with_context(question: str, context_snippets: List[str]):
    # Combine snippets but cap total length to avoid giant prompts
    context = "\n\n".join(context_snippets)[:8000]
    prompt = f"""You are ScriptBees Assistant. Use ONLY the content provided to answer the question.

Context:
{context}

Question: {question}

Give a short, factual answer based ONLY on the context above."""
    try:
        resp = genai_client.models.generate_content(
            model=GEN_MODEL,
            contents=[prompt],
            max_output_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        # robust extraction
        text = getattr(resp, "text", None)
        if not text:
            # may be dict-like
            candidates = getattr(resp, "candidates", None) or (resp.get("candidates") if isinstance(resp, dict) else None)
            if candidates and len(candidates) > 0:
                # some SDKs return 'content' inside candidate
                text = candidates[0].get("content") if isinstance(candidates[0], dict) else getattr(candidates[0], "content", None)
        if isinstance(text, dict):
            text = text.get("text", "") or str(text)
        if not text:
            text = "No answer returned from upstream."
    except Exception as e:
        logger.exception("Gemini generation failed: %s", e)
        raise HTTPException(status_code=502, detail="Upstream Gemini error")
    return str(text).strip()

# Routes
@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, key: str = Depends(verify_api_key)):
    start = time.time()
    h = ckey(req.question)
    if h in cache:
        r = cache[h]
        r["cached"] = True
        r["response_time_seconds"] = time.time() - start
        return r

    q_emb = get_embedding_for_text(req.question)
    # ensure normalized
    q_norm = np.linalg.norm(q_emb)
    if q_norm > 0:
        q_emb = q_emb / q_norm

    results = top_k_search(q_emb, k=TOP_K)
    if not results:
        return AskResponse(answer="No matching information found.", sources=[], retrieved=[], cached=False, response_time_seconds=time.time()-start)

    # collect top snippets (may be multiple if TOP_K > 1)
    snippets = []
    retrieved_objs = []
    for idx, score in results[:TOP_K]:
        p = pages[idx]
        snippets.append(p.get("snippet", "")[:2000])
        retrieved_objs.append(Source(url=p.get("url",""), title=p.get("title",""), score=float(score)))

    answer = generate_answer_with_context(req.question, snippets)

    resp = {
        "answer": answer,
        "sources": [retrieved_objs[0].url] if retrieved_objs else [],
        "retrieved": retrieved_objs,
        "cached": False,
        "response_time_seconds": time.time() - start
    }
    cache[h] = resp
    return resp

@app.get("/")
async def health():
    return {"status": "online", "mode": "gemini-memmap"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
