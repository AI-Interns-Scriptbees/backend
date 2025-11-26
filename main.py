"""
SCRIPTBEES ASSISTANT - ULTRA FAST (OPENAI VERSION)
Replaces slow GPT4All with GPT-4o-mini (super fast, accurate)

NO CHANGES REQUIRED IN FRONTEND
Your UI + API key system continues same.
"""

import os
import json
import time
import logging
import hashlib
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ======================================================================
# ENV LOADING
# ======================================================================

def find_env():
    cur = Path(__file__).resolve().parent
    for _ in range(10):
        if (cur / ".env").exists():
            return cur / ".env"
        cur = cur.parent
    return None

env_path = find_env()
if env_path:
    load_dotenv(env_path)

# ======================================================================
# LOGGING
# ======================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("scriptbees")

# ======================================================================
# CONFIG
# ======================================================================

API_KEY = os.getenv("RAG_API_KEY", "change-me")  # your bot API key  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise Exception("❌ Missing OPENAI_API_KEY in .env file")

CONTENT_DIR = "content"
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

TOP_K = 1                       # FASTEST retrieval  
MAX_TOKENS = 150                # Enough for complete answers  
TEMPERATURE = 0.2               # Stable + fast  
INDEX_PATH = f"{CONTENT_DIR}/pages.faiss"
PAGES_PATH = f"{CONTENT_DIR}/pages.json"
META_PATH = f"{CONTENT_DIR}/pages_meta.json"

# ======================================================================
# API REQUEST MODELS
# ======================================================================

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
    cached: bool = False
    response_time_seconds: float

# ======================================================================
# API KEY SECURITY
# ======================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    incoming = api_key or ""
    if not incoming:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            incoming = auth.split(" ", 1)[1].strip()

    if incoming != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

    return incoming

# ======================================================================
# RESPONSE CACHE
# ======================================================================

response_cache = {}
MAX_CACHE = 300

def cache_key(q): return hashlib.md5(q.strip().lower().encode()).hexdigest()

def cache_get(q): return response_cache.get(cache_key(q))

def cache_set(q, data):
    if len(response_cache) >= MAX_CACHE:
        response_cache.pop(next(iter(response_cache)))
    response_cache[cache_key(q)] = data

# ======================================================================
# FASTAPI APP
# ======================================================================

app = FastAPI(title="ScriptBees Assistant (OpenAI)", version="6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = None
generator = None

# ======================================================================
# STARTUP
# ======================================================================

@app.on_event("startup")
async def startup():
    global retriever, generator

    logger.info("🚀 Starting ScriptBees Assistant (OpenAI Ultra-Fast)")

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        from openai import OpenAI

        # ============== RETRIEVER CLASS ==============
        class Retriever:
            def __init__(self):
                logger.info("📦 Loading FAISS retriever...")
                self.model = SentenceTransformer(MODEL_NAME)
                self.index = faiss.read_index(INDEX_PATH)

                with open(META_PATH, "r", encoding="utf8") as f:
                    self.meta = json.load(f)

                with open(PAGES_PATH, "r", encoding="utf8") as f:
                    pages = json.load(f)

                self.pages = {p["id"]: p for p in pages}

                logger.info(f"✓ Loaded {self.index.ntotal} embedded documents")

            def retrieve(self, query, k=TOP_K):
                vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
                scores, indices = self.index.search(vec, k)

                results = []
                for s, idx in zip(scores[0], indices[0]):
                    if idx == -1:
                        continue
                    meta = self.meta[idx]
                    page = self.pages.get(meta["id"])
                    results.append({
                        "url": meta["url"],
                        "title": meta["title"],
                        "score": float(s),
                        "text": page["text"][:500]
                    })
                return results

        # ============== GENERATOR CLASS (OPENAI) ==============
        class LLMGenerator:
            def __init__(self):
                logger.info("🤖 Using OpenAI GPT-4o-mini")
                self.client = OpenAI(api_key=OPENAI_API_KEY)

            def generate(self, question, docs):
                context = docs[0]["text"]

                prompt = f"""
You are ScriptBees AI Assistant. Answer ONLY using this info:

Context:
{context}

Question: {question}

Give a short, accurate answer.
"""

                res = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )

                answer = res.choices[0].message.content.strip()
                answer += f"\n\n[Source: {docs[0]['url']}]"

                return answer

        retriever = Retriever()
        generator = LLMGenerator()

        logger.info("✅ ScriptBees Assistant READY (OpenAI Turbo Mode)")

    except Exception as e:
        logger.error("❌ Startup failed", exc_info=True)
        raise

# ======================================================================
# ROUTES
# ======================================================================

@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, api_key: str = Depends(verify_api_key)):
    start = time.time()

    # --- cache check ---
    cached = cache_get(req.question)
    if cached:
        cached["cached"] = True
        cached["response_time_seconds"] = time.time() - start
        return cached

    # --- retrieval ---
    docs = retriever.retrieve(req.question)
    if not docs:
        return AskResponse(
            answer="No relevant information found.",
            sources=[],
            retrieved=[],
            response_time_seconds=time.time() - start
        )

    # --- generation ---
    answer = generator.generate(req.question, docs)
    sources = [d["url"] for d in docs]

    resp = {
        "answer": answer,
        "sources": sources,
        "retrieved": [
            Source(url=d["url"], title=d["title"], score=d["score"]) for d in docs
        ],
        "response_time_seconds": time.time() - start,
        "cached": False
    }

    cache_set(req.question, resp)
    return resp


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "documents": retriever.index.ntotal if retriever else 0,
        "model": "gpt-4o-mini",
        "mode": "openai-ultra-fast"
    }


@app.get("/")
async def home():
    return {"service": "ScriptBees Assistant", "mode": "openai", "status": "online"}

# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
