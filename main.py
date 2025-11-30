#!/usr/bin/env python3
"""
main.py - Robust memmap Gemini server for ScriptBees (REPAIR + tolerant mode)

Behavior improvements vs prior:
- Attempts to auto-repair embeddings.npy if its byte size can be reshaped to (n_pages, dim).
- If repair impossible, server starts with retriever disabled and /api/ask returns 503 with a message.
- All original behavior otherwise preserved.

Requirements (same as before):
- google-genai
- fastapi
- uvicorn
- numpy
- python-dotenv
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
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Security, Request, Response, status
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

# Config from env (defaults)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_KEY = os.getenv("RAG_API_KEY", "change-me")
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
logger = logging.getLogger("scriptbees-memmap-robust")

# FastAPI app & CORS
app = FastAPI(title="ScriptBees — Gemini Memmap (Robust)")
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
def verify_api_key(req: Request, key: str = Depends(api_key_header)):
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

# Globals for retriever
pages = []
embeddings = None   # numpy memmap or ndarray
emb_dim = 0
num_pages = 0
retriever_enabled = False

# Utility: attempt to load pages.json
def load_pages():
    global pages, num_pages
    if not PAGES_FILE.exists():
        logger.error("Missing pages.json at %s", PAGES_FILE)
        return False
    with open(PAGES_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)
    num_pages = len(pages)
    logger.info("Loaded %d pages from %s", num_pages, PAGES_FILE)
    return True

# Utility: try robustly load embeddings memmap or repair if possible
def try_load_or_repair_embeddings():
    global embeddings, emb_dim, retriever_enabled
    if not EMBED_FILE.exists():
        logger.error("Missing embeddings.npy at %s", EMBED_FILE)
        return False

    # Quick checks by file size
    size_bytes = EMBED_FILE.stat().st_size
    if size_bytes % 4 != 0:
        logger.error("Embeddings file size (%d) not divisible by 4 -> not float32", size_bytes)
        return False
    total_floats = size_bytes // 4
    if num_pages <= 0:
        logger.error("num_pages is zero; cannot shape embeddings")
        return False

    if total_floats % num_pages == 0:
        dim = total_floats // num_pages
        logger.info("Inferred embeddings shape: (%d, %d)", num_pages, dim)
        # Try memmap load as (num_pages, dim)
        try:
            mm = np.memmap(str(EMBED_FILE), dtype=np.float32, mode="r", shape=(num_pages, dim))
            # basic sanity: check NaNs
            sample = mm[0]
            if np.isnan(sample).any():
                logger.warning("Embeddings contain NaNs in sample row.")
            embeddings = mm
            emb_dim = dim
            retriever_enabled = True
            logger.info("Loaded embeddings memmap with shape (%d,%d)", num_pages, emb_dim)
            return True
        except Exception as e:
            logger.warning("Memmap load failed for shape (%d,%d): %s", num_pages, dim, e)
            # try fallback: load full array and reshape
            try:
                arr = np.load(str(EMBED_FILE), mmap_mode="r")
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape((num_pages, dim))
                elif arr.ndim == 2 and arr.shape[0] == num_pages and arr.shape[1] == dim:
                    pass
                else:
                    logger.warning("Loaded array shape %s not matching expected (%d,%d)", arr.shape, num_pages, dim)
                    return False
                # save a memmap-style .npy (overwrite safe path embeddings_repaired.npy then rename)
                repaired = str(EMBED_FILE.parent / "embeddings_repaired.npy")
                np.save(repaired, arr.astype(np.float32))
                # replace original
                os.replace(repaired, str(EMBED_FILE))
                # load memmap fresh
                mm2 = np.memmap(str(EMBED_FILE), dtype=np.float32, mode="r", shape=(num_pages, dim))
                embeddings = mm2
                emb_dim = dim
                retriever_enabled = True
                logger.info("Repaired and loaded embeddings memmap (%d,%d)", num_pages, emb_dim)
                return True
            except Exception as e2:
                logger.exception("Failed fallback reshape & load: %s", e2)
                return False
    else:
        # cannot reshape cleanly
        logger.error("Total floats (%d) not divisible by num_pages (%d) -> cannot reshape", total_floats, num_pages)
        return False

# Startup: load pages + embeddings (try repair)
logger.info("Starting ScriptBees memmap server (robust startup)...")
_pages_ok = load_pages()
if _pages_ok:
    ok = try_load_or_repair_embeddings()
    if ok:
        logger.info("Retriever enabled.")
    else:
        retriever_enabled = False
        logger.warning("Retriever disabled due to embeddings/pages mismatch. App will start but /api/ask returns 503 until fixed.")
else:
    retriever_enabled = False
    logger.warning("Pages not loaded. Retriever disabled. Ensure content/pages.json exists.")

# Create genai client (even if retriever disabled we keep client init)
if genai is None:
    logger.warning("google-genai client not installed. Gemini features will fail if invoked.")
    genai_client = None
else:
    if GEMINI_API_KEY:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        genai_client = genai.Client()

# Helper: embed question using Gemini
def get_embedding_for_text(text: str):
    if genai_client is None:
        raise HTTPException(status_code=502, detail="Embedding client not available")
    resp = genai_client.embeddings.create(model=EMBED_MODEL, input=text)
    if hasattr(resp, "data"):
        vec = resp.data[0].embedding
    else:
        vec = resp.get("data", [])[0].get("embedding")
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr

# Simple top-k search against memmap embeddings
def top_k_search(query_vec: np.ndarray, k: int = 1):
    if embeddings is None:
        return []
    q = query_vec.reshape(-1)
    best_scores = np.full(k, -np.inf, dtype=np.float32)
    best_idxs = np.full(k, -1, dtype=np.int32)
    chunk = 2048
    for i in range(0, num_pages, chunk):
        block = embeddings[i: i+chunk]
        dots = block.dot(q)
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

# Gemini generation using context snippets
def generate_answer_with_context(question: str, context_snippets: List[str]):
    if genai_client is None:
        raise HTTPException(status_code=502, detail="Generation client not available")
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
        text = getattr(resp, "text", None)
        if not text:
            candidates = getattr(resp, "candidates", None) or (resp.get("candidates") if isinstance(resp, dict) else None)
            if candidates and len(candidates) > 0:
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
    if not retriever_enabled:
        # give a helpful error with guidance
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retriever not available. Ensure content/pages.json and content/embeddings.npy exist and match. See logs."
        )

    h = ckey(req.question)
    if h in cache:
        r = cache[h]
        r["cached"] = True
        r["response_time_seconds"] = time.time() - start
        return r

    # embed question and search
    q_emb = get_embedding_for_text(req.question)
    q_norm = np.linalg.norm(q_emb)
    if q_norm > 0:
        q_emb = q_emb / q_norm

    results = top_k_search(q_emb, k=TOP_K)
    if not results:
        return AskResponse(answer="No matching information found.", sources=[], retrieved=[], cached=False, response_time_seconds=time.time()-start)

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
async def home():
    status_text = {"status": "online", "retriever": bool(retriever_enabled)}
    return status_text

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
