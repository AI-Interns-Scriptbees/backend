#!/usr/bin/env python3
"""
main.py - Robust memmap Gemini server for ScriptBees (UPDATED, full)

This file:
- Uses genai_client.models.embed_content(...) (robust across SDKs)
- Extracts numeric embeddings from SDK wrapper objects (ContentEmbedding etc.)
- Auto-discovers a working embedding model at startup if EMBED_MODEL fails
- Loads/repairs content/pages.json and content/embeddings.npy (memmap)
- Provides / endpoint showing selected embed model
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Try importing genai; keep None if unavailable
try:
    from google import genai
except Exception:
    genai = None

# -------- Load .env (if present) --------
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

# -------- Config from env (defaults) --------
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

# -------- Logging --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scriptbees-memmap-robust")

# -------- FastAPI app & CORS --------
app = FastAPI(title="ScriptBees — Gemini Memmap (Robust)")
_allow_credentials = False if FRONTEND_ORIGINS.strip() == "*" else True
origins = [o.strip() for o in FRONTEND_ORIGINS.split(",")] if FRONTEND_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Models --------
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

# -------- API Key security --------
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

# -------- Cache util --------
def ckey(q: str) -> str:
    return hashlib.md5(q.lower().encode()).hexdigest()

cache = {}

# -------- Globals for retriever --------
pages: List[dict] = []
embeddings: Optional[np.memmap] = None   # numpy memmap or ndarray
emb_dim = 0
num_pages = 0
retriever_enabled = False

# -------- Utilities: load pages + embeddings --------
def load_pages() -> bool:
    global pages, num_pages
    if not PAGES_FILE.exists():
        logger.error("Missing pages.json at %s", PAGES_FILE)
        return False
    try:
        with open(PAGES_FILE, "r", encoding="utf-8") as f:
            pages = json.load(f)
        num_pages = len(pages)
        logger.info("Loaded %d pages from %s", num_pages, PAGES_FILE)
        return True
    except Exception as e:
        logger.exception("Failed to load pages.json: %s", e)
        return False

def try_load_or_repair_embeddings() -> bool:
    global embeddings, emb_dim, retriever_enabled
    if not EMBED_FILE.exists():
        logger.error("Missing embeddings.npy at %s", EMBED_FILE)
        return False

    try:
        size_bytes = EMBED_FILE.stat().st_size
        if size_bytes % 4 != 0:
            logger.error("Embeddings file size (%d) not divisible by 4 -> not float32", size_bytes)
            return False
        total_floats = size_bytes // 4
        if num_pages <= 0:
            logger.error("num_pages is zero; cannot shape embeddings")
            return False
        if total_floats % num_pages != 0:
            logger.error("Total floats (%d) not divisible by num_pages (%d) -> cannot reshape", total_floats, num_pages)
            return False

        dim = total_floats // num_pages
        logger.info("Inferred embeddings shape: (%d, %d)", num_pages, dim)

        # Try memmap load
        try:
            mm = np.memmap(str(EMBED_FILE), dtype=np.float32, mode="r", shape=(num_pages, dim))
            # quick sanity
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
            # fallback: load and reshape
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
                repaired = str(EMBED_FILE.parent / "embeddings_repaired.npy")
                np.save(repaired, arr.astype(np.float32))
                os.replace(repaired, str(EMBED_FILE))
                mm2 = np.memmap(str(EMBED_FILE), dtype=np.float32, mode="r", shape=(num_pages, dim))
                embeddings = mm2
                emb_dim = dim
                retriever_enabled = True
                logger.info("Repaired and loaded embeddings memmap (%d,%d)", num_pages, emb_dim)
                return True
            except Exception as e2:
                logger.exception("Failed fallback reshape & load: %s", e2)
                return False
    except Exception as outer_e:
        logger.exception("Unexpected error when loading embeddings: %s", outer_e)
        return False

# Startup: load pages + embeddings
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

# -------- Create genai client --------
if genai is None:
    logger.warning("google-genai client not installed. Gemini features will fail if invoked.")
    genai_client = None
else:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
    except Exception as e:
        logger.exception("Failed to create genai client: %s", e)
        genai_client = None

# -------- Auto-discover embedding model (if configured model fails) --------
def _auto_discover_embedding_model(client: Any, configured_model: str) -> str:
    global EMBED_MODEL
    if client is None:
        logger.warning("No genai client available for auto-discovery.")
        return configured_model

    # Try configured model first
    if configured_model:
        try:
            logger.info("Testing configured EMBED_MODEL='%s' ...", configured_model)
            if hasattr(client, "models") and hasattr(client.models, "embed_content"):
                client.models.embed_content(model=configured_model, contents=["test"])
            elif hasattr(client, "embed") and hasattr(client.embed, "create"):
                client.embed.create(model=configured_model, input="test")
            else:
                raise RuntimeError("No embed method available on client to test configured model.")
            logger.info("Configured EMBED_MODEL '%s' appears to work.", configured_model)
            return configured_model
        except Exception as e:
            logger.warning("Configured EMBED_MODEL '%s' failed test: %s", configured_model, getattr(e, "message", str(e)))

    # Try to list models
    raw = None
    try:
        logger.info("Listing models to discover embedding-capable model...")
        if hasattr(client, "list_models"):
            raw = client.list_models()
            logger.info("Used client.list_models()")
        elif hasattr(client, "models") and hasattr(client.models, "list"):
            raw = client.models.list()
            logger.info("Used client.models.list()")
        else:
            logger.warning("Client has no models.list/list_models API; skipping discovery.")
            raw = None
    except Exception as e:
        logger.warning("List models call failed: %s", e)
        raw = None

    # Normalize into iterable items
    items = []
    try:
        if raw is None:
            items = []
        elif isinstance(raw, dict) and "models" in raw:
            items = raw["models"]
        elif hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
            items = list(raw)
        else:
            items = []
    except Exception as e:
        logger.warning("Failed to normalize models list: %s", e)
        items = []

    logger.info("Discovered %d model entries to inspect.", len(items))

    # Heuristic: prefer model ids containing 'embed' or 'embedding'
    candidates = []
    for m in items:
        try:
            model_id = None
            if isinstance(m, dict):
                model_id = m.get("name") or m.get("id") or m.get("model")
                supported = m.get("supported_methods") or m.get("capabilities") or m.get("methods") or []
            else:
                model_id = getattr(m, "name", None) or getattr(m, "id", None) or getattr(m, "model", None)
                supported = getattr(m, "supported_methods", None) or getattr(m, "capabilities", None) or getattr(m, "methods", None) or []
            supports_embed = False
            if supported:
                try:
                    supports_embed = any("embed" in str(x).lower() for x in supported)
                except Exception:
                    supports_embed = False
            if not supports_embed and model_id:
                supports_embed = "embed" in model_id.lower() or "embedding" in model_id.lower()
            if model_id:
                candidates.append((model_id, supports_embed))
        except Exception:
            continue

    # Order: explicit embed-capable first
    ordered = [m for m, ok in candidates if ok] + [m for m, ok in candidates if not ok]

    for candidate in ordered:
        try:
            logger.info("Trying candidate embedding model: %s", candidate)
            if hasattr(client, "models") and hasattr(client.models, "embed_content"):
                client.models.embed_content(model=candidate, contents=["hello world"])
            elif hasattr(client, "embed") and hasattr(client.embed, "create"):
                client.embed.create(model=candidate, input="hello world")
            else:
                logger.warning("Client lacks a known embed API to test candidates.")
                break
            EMBED_MODEL = candidate
            logger.info("Auto-discovery selected EMBED_MODEL='%s'", candidate)
            return candidate
        except Exception as e:
            logger.info("Candidate %s failed: %s", candidate, getattr(e, "message", str(e)))
            continue

    logger.warning("Auto-discovery failed to find a working embedding model; keeping EMBED_MODEL='%s'", configured_model)
    return configured_model

# Run auto-discovery at startup (updates in-memory EMBED_MODEL if successful)
try:
    resolved = _auto_discover_embedding_model(genai_client, EMBED_MODEL)
    if resolved != EMBED_MODEL:
        logger.info("Resolved EMBED_MODEL -> %s", resolved)
    else:
        logger.info("EMBED_MODEL remains -> %s", EMBED_MODEL)
except Exception as e:
    logger.warning("Auto-discovery encountered an exception: %s", e)

# -------- Robust embedding extraction --------
def _extract_numeric_from_obj(obj: Any) -> Optional[List[float]]:
    """
    Try many strategies to pull a plain list of numbers from an SDK wrapper object.
    Returns a Python list of floats or None.
    """
    # 1) If it's a dict with common keys
    if isinstance(obj, dict):
        for k in ("embedding", "value", "values", "vector", "data", "content"):
            if k in obj:
                res = _extract_numeric_from_obj(obj[k])
                if res is not None:
                    return res

    # 2) Try common attributes
    for attr in ("embedding", "value", "values", "vector", "content", "content_embedding"):
        try:
            if hasattr(obj, attr):
                cand = getattr(obj, attr)
                res = _extract_numeric_from_obj(cand)
                if res is not None:
                    return res
        except Exception:
            pass

    # 3) If it exposes numpy(), tolist(), list-like conversions
    try:
        if hasattr(obj, "numpy"):
            arr = obj.numpy()
            if hasattr(arr, "__iter__"):
                return [float(x) for x in arr]
    except Exception:
        pass

    try:
        if hasattr(obj, "to_list"):
            arr = obj.to_list()
            if hasattr(arr, "__iter__"):
                return [float(x) for x in arr]
    except Exception:
        pass

    try:
        if hasattr(obj, "tolist"):
            arr = obj.tolist()
            if hasattr(arr, "__iter__"):
                return [float(x) for x in arr]
    except Exception:
        pass

    # 4) If it's already a list/tuple/ndarray
    if isinstance(obj, (list, tuple, np.ndarray)):
        try:
            return [float(x) for x in obj]
        except Exception:
            # elements might be wrapped; try flattening via recursion
            out = []
            for el in obj:
                el_res = _extract_numeric_from_obj(el)
                if el_res is None:
                    return None
                out.extend(el_res)
            return out if out else None

    # 5) Iterable fallback (but not string/bytes)
    try:
        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
            lst = list(obj)
            # if the iterable yields numbers directly
            try:
                return [float(x) for x in lst]
            except Exception:
                # else try recursive extraction
                out = []
                for el in lst:
                    er = _extract_numeric_from_obj(el)
                    if er is None:
                        return None
                    out.extend(er)
                return out if out else None
    except Exception:
        pass

    return None

def get_embedding_for_text(text: str) -> np.ndarray:
    """
    Request an embedding and return a normalized float32 numpy vector.
    """
    if genai_client is None:
        logger.error("Embedding client not available (genai_client is None)")
        raise HTTPException(status_code=502, detail="Embedding client not available")

    try:
        resp = genai_client.models.embed_content(model=EMBED_MODEL, contents=[text])
    except Exception as e:
        logger.exception("Embedding request failed: %s", e)
        raise HTTPException(status_code=502, detail="Upstream embedding error")

    # Find candidate wrapper that likely contains numbers
    candidate = None

    # 1) resp.embeddings (SDK object)
    try:
        if hasattr(resp, "embeddings"):
            emblist = resp.embeddings
            if emblist and len(emblist) > 0:
                candidate = emblist[0]
    except Exception:
        candidate = None

    # 2) dict-like with 'embeddings' or 'data'
    if candidate is None and isinstance(resp, dict):
        for key in ("embeddings", "data", "results"):
            if key in resp and isinstance(resp[key], (list, tuple)) and len(resp[key]) > 0:
                candidate = resp[key][0]
                break

    # 3) 'candidates' list (some SDK variants)
    if candidate is None and isinstance(resp, dict) and "candidates" in resp:
        cand = resp["candidates"]
        if isinstance(cand, (list, tuple)) and len(cand) > 0:
            if isinstance(cand[0], dict) and "embedding" in cand[0]:
                candidate = cand[0]["embedding"]
            else:
                candidate = cand[0]

    # 4) resp.data[0].embedding or resp.data
    if candidate is None:
        try:
            data_attr = getattr(resp, "data", None)
            if data_attr and len(data_attr) > 0:
                item0 = data_attr[0]
                if hasattr(item0, "embedding"):
                    candidate = item0.embedding
                elif isinstance(item0, dict) and "embedding" in item0:
                    candidate = item0["embedding"]
                else:
                    candidate = item0
        except Exception:
            pass

    # 5) fallback to resp itself
    if candidate is None:
        candidate = resp

    numeric = _extract_numeric_from_obj(candidate)
    if numeric is None:
        logger.error("Unable to parse numeric embedding. Response repr: %s", repr(resp)[:2000])
        raise HTTPException(status_code=502, detail="Unexpected embedding response structure from upstream")

    # Convert to numpy array and normalize
    try:
        vec = np.asarray(numeric, dtype=np.float32).reshape(-1)
    except Exception as e:
        logger.exception("Failed to convert embedding to numpy array: %s", e)
        raise HTTPException(status_code=502, detail="Invalid numeric embedding form")

    if vec.size == 0:
        logger.error("Received empty embedding vector")
        raise HTTPException(status_code=502, detail="Empty embedding returned by upstream")

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec

# -------- Simple top-k search against memmap embeddings --------
def top_k_search(query_vec: np.ndarray, k: int = 1):
    if embeddings is None:
        return []
    q = query_vec.reshape(-1).astype(np.float32)
    best_scores = np.full(k, -np.inf, dtype=np.float32)
    best_idxs = np.full(k, -1, dtype=np.int32)
    chunk = 2048
    for i in range(0, num_pages, chunk):
        block = embeddings[i: i+chunk]
        if block.shape[1] != q.shape[0]:
            logger.error("Embedding dim mismatch: block %s vs query %s", block.shape, q.shape)
            break
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

# -------- Gemini generation using context snippets --------
def generate_answer_with_context(question: str, context_snippets: List[str]):
    if genai_client is None:
        logger.error("Generation client not available (genai_client is None)")
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
        # Try to extract text from responses returned by different SDK variants
        text = None
        if hasattr(resp, "text") and resp.text:
            text = resp.text
        elif hasattr(resp, "candidates") and getattr(resp, "candidates"):
            c = resp.candidates[0]
            text = getattr(c, "content", None) or getattr(c, "text", None)
        elif isinstance(resp, dict):
            if "candidates" in resp and isinstance(resp["candidates"], (list, tuple)) and len(resp["candidates"])>0:
                cand0 = resp["candidates"][0]
                if isinstance(cand0, dict) and "content" in cand0:
                    text = cand0["content"]
                elif isinstance(cand0, dict) and "text" in cand0:
                    text = cand0["text"]
            elif "text" in resp:
                text = resp["text"]
        if not text:
            text = "No answer returned from upstream."
    except Exception as e:
        logger.exception("Gemini generation failed: %s", e)
        raise HTTPException(status_code=502, detail="Upstream Gemini error")
    return str(text).strip()

# -------- Routes --------
@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, key: str = Depends(verify_api_key)):
    start = time.time()
    if not retriever_enabled:
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
    retrieved_objs: List[Source] = []
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
    return {"status": "online", "retriever": bool(retriever_enabled), "embed_model": EMBED_MODEL}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# -------- Run --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
