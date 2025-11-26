"""
SCRIPTBEES ASSISTANT - ULTRA FAST (OPENAI VERSION)
CLEAN RENDER-SAFE VERSION — NO PROXY, NO HTTPX OVERRIDE
"""

import os
import json
import time
import logging
import hashlib
from typing import List, Optional
from pathlib import Path

# ----------------------------------------------------------------------
# Remove proxy variables (Render Fix)
# ----------------------------------------------------------------------
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

# ----------------------------------------------------------------------
# Load Environment
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scriptbees")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
API_KEY = os.getenv("RAG_API_KEY", "change-me")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise Exception("❌ Missing OPENAI_API_KEY in .env")

CONTENT_DIR = "content"
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 1
MAX_TOKENS = 150
TEMPERATURE = 0.2

INDEX_PATH = f"{CONTENT_DIR}/pages.faiss"
PAGES_PATH = f"{CONTENT_DIR}/pages.json"
META_PATH = f"{CONTENT_DIR}/pages_meta.json"

# ----------------------------------------------------------------------
# FastAPI App
# ----------------------------------------------------------------------
from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

app = FastAPI(title="ScriptBees Assistant (OpenAI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Request/Response Models
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# API Key Security
# ----------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    incoming = api_key or ""

    # Support "Authorization: Bearer"
    if not incoming:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            incoming = auth.split(" ", 1)[1].strip()

    if incoming != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

    return incoming

# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------
response_cache = {}

def cache_key(q): return hashlib.md5(q.lower().encode()).hexdigest()

# ----------------------------------------------------------------------
# Startup — load FAISS + OpenAI
# ----------------------------------------------------------------------
from openai import OpenAI

retriever = None
generator = None

@app.on_event("startup")
async def startup():
    global retriever, generator

    logger.info("🚀 Starting ScriptBees Assistant (OpenAI Ultra-Fast)")

    import faiss
    from sentence_transformers import SentenceTransformer

    # ------------------------ RETRIEVER ------------------------
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

        def retrieve(self, question: str):
            vec = self.model.encode([question], normalize_embeddings=True).astype("float32")
            scores, indices = self.index.search(vec, TOP_K)

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

    # ------------------------ LLM GENERATOR ------------------------
    class LLMGenerator:
        def __init__(self):
            logger.info("🤖 Using OpenAI GPT-4o-mini")
            self.client = OpenAI(api_key=OPENAI_API_KEY)

        def generate(self, question, docs):
            context = docs[0]["text"]

            prompt = f"""
You are ScriptBees AI Assistant.
Answer ONLY using this context:

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
            return answer + f"\n\n[Source: {docs[0]['url']}]"

    retriever = Retriever()
    generator = LLMGenerator()

    logger.info("✅ ScriptBees Assistant READY")

# ----------------------------------------------------------------------
# Ask Endpoint
# ----------------------------------------------------------------------
@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, api_key: str = Depends(verify_api_key)):
    start = time.time()

    key = cache_key(req.question)
    if key in response_cache:
        cached = response_cache[key]
        cached["cached"] = True
        cached["response_time_seconds"] = time.time() - start
        return cached

    docs = retriever.retrieve(req.question)
    if not docs:
        return AskResponse(
            answer="No relevant information found.",
            sources=[],
            retrieved=[],
            cached=False,
            response_time_seconds=time.time() - start
        )

    answer = generator.generate(req.question, docs)

    response = {
        "answer": answer,
        "sources": [d["url"] for d in docs],
        "retrieved": [Source(**d) for d in docs],
        "cached": False,
        "response_time_seconds": time.time() - start
    }

    response_cache[key] = response
    return response

# ----------------------------------------------------------------------
# Health & Root
# ----------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "documents": retriever.index.ntotal if retriever else 0,
        "model": "gpt-4o-mini"
    }

@app.get("/")
async def home():
    return {"service": "ScriptBees Assistant", "mode": "openai", "status": "online"}

# ----------------------------------------------------------------------
# RUN LOCAL
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
