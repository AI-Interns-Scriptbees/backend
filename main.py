"""
SCRIPTBEES ASSISTANT - HEAVY VERSION (FAISS + TORCH + SENTENCE-TRANSFORMERS)
Fully Render-Compatible — Fixes OpenAI Client.proxy error
"""

import os
import json
import time
import logging
import hashlib
from typing import List
from pathlib import Path

# ======================================================================
# REMOVE ALL PROXY ENV VARIABLES (Render injects them → OpenAI crashes)
# ======================================================================
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import httpx
from openai import OpenAI

# -> Create a proxy-free HTTP client for OpenAI
safe_http_client = httpx.Client(
    proxies=None,
    timeout=30,
    verify=True,
)

# ======================================================================
# ENV LOADING
# ======================================================================
from dotenv import load_dotenv

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("RAG_API_KEY", "change-me")

if not OPENAI_API_KEY:
    raise Exception("❌ Missing OPENAI_API_KEY in .env")

# ======================================================================
# LOGGING
# ======================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scriptbees")

# ======================================================================
# CONFIG
# ======================================================================
CONTENT_DIR = "content"
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 1
MAX_TOKENS = 150
TEMPERATURE = 0.2

INDEX_PATH = f"{CONTENT_DIR}/pages.faiss"
PAGES_PATH = f"{CONTENT_DIR}/pages.json"
META_PATH = f"{CONTENT_DIR}/pages_meta.json"

# ======================================================================
# FASTAPI SETUP
# ======================================================================
from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

app = FastAPI(title="ScriptBees Assistant (Heavy Version)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ======================================================================
# MODELS
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
    cached: bool
    response_time_seconds: float

# ======================================================================
# API KEY SECURITY
# ======================================================================
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(req: Request, key: str = Security(api_key_header)):
    incoming = key or ""

    if not incoming:
        auth = req.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            incoming = auth.split(" ", 1)[1]

    if incoming != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    return incoming

# ======================================================================
# CACHE
# ======================================================================
cache = {}
def ckey(q): return hashlib.md5(q.lower().encode()).hexdigest()

# ======================================================================
# STARTUP (HEAVY LOAD)
# ======================================================================
retriever = None
generator = None

@app.on_event("startup")
async def startup():
    global retriever, generator

    logger.info("🚀 Starting ScriptBees Assistant (Heavy Version)")

    import faiss
    from sentence_transformers import SentenceTransformer

    # ------------------ RETRIEVER ----------------------
    class Retriever:
        def __init__(self):
            logger.info("📦 Loading FAISS index + transformer encoder...")

            self.model = SentenceTransformer(MODEL_NAME)

            self.index = faiss.read_index(INDEX_PATH)

            with open(META_PATH, "r") as f:
                self.meta = json.load(f)

            with open(PAGES_PATH, "r") as f:
                pages = json.load(f)

            self.pages = {p["id"]: p for p in pages}

            logger.info(f"✓ Loaded {self.index.ntotal} documents")

        def retrieve(self, q):
            vec = self.model.encode([q], normalize_embeddings=True).astype("float32")
            scores, idxs = self.index.search(vec, TOP_K)

            results = []
            for s, idx in zip(scores[0], idxs[0]):
                if idx == -1:
                    continue
                meta = self.meta[idx]
                page = self.pages[meta["id"]]

                results.append({
                    "score": float(s),
                    "url": meta["url"],
                    "title": meta["title"],
                    "text": page["text"][:500]
                })
            return results

    # ------------------ GENERATOR (OpenAI) ----------------------
    class LLMGenerator:
        def __init__(self):
            logger.info("🤖 Using GPT-4o-mini (safe, proxy-free)")

            self.client = OpenAI(
                api_key=OPENAI_API_KEY,
                http_client=safe_http_client    # <--- IMPORTANT FIX
            )

        def generate(self, question, docs):
            context = docs[0]["text"]

            prompt = f"""
Use ONLY this context to answer:

{context}

Question: {question}

Short and accurate answer:
"""

            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )

            answer = resp.choices[0].message.content.strip()

            return answer + f"\n\n[Source: {docs[0]['url']}]"

    retriever = Retriever()
    generator = LLMGenerator()

    logger.info("✅ Assistant ready.")

# ======================================================================
# ROUTES
# ======================================================================
@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, key: str = Depends(verify_api_key)):
    start = time.time()

    # cache check
    if ckey(req.question) in cache:
        r = cache[ckey(req.question)]
        r["cached"] = True
        r["response_time_seconds"] = time.time() - start
        return r

    docs = retriever.retrieve(req.question)
    if not docs:
        return AskResponse(
            answer="No information found.",
            sources=[],
            retrieved=[],
            cached=False,
            response_time_seconds=time.time() - start
        )

    answer = generator.generate(req.question, docs)

    resp = {
        "answer": answer,
        "sources": [d["url"] for d in docs],
        "retrieved": [Source(**d) for d in docs],
        "cached": False,
        "response_time_seconds": time.time() - start
    }

    cache[ckey(req.question)] = resp
    return resp

@app.get("/")
async def home():
    return {"status": "online", "version": "heavy"}

# ======================================================================
# RUN (LOCAL)
# ======================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
