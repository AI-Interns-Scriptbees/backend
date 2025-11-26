"""
SCRIPTBEES ASSISTANT - ULTRA FAST (OPENAI VERSION)
"""

import os
import json
import time
import logging
import hashlib
from typing import List
from pathlib import Path

# ----------------------------------------------------------------------
# ABSOLUTE FIX FOR RENDER + OPENAI PROXY BUG
# ----------------------------------------------------------------------
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None")

import httpx
from openai import OpenAI

# Create Safe HTTP client (NO PROXIES)
safe_http_client = httpx.Client(
    timeout=25,
    proxies=None,   # <--- CRITICAL
    verify=True
)

# ----------------------------------------------------------------------
# Load env
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
# Config
# ----------------------------------------------------------------------
API_KEY = os.getenv("RAG_API_KEY", "change-me")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise Exception("Missing OPENAI_API_KEY in .env")

CONTENT_DIR = "content"
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 1
MAX_TOKENS = 150
TEMPERATURE = 0.2

INDEX_PATH = f"{CONTENT_DIR}/pages.faiss"
PAGES_PATH = f"{CONTENT_DIR}/pages.json"
META_PATH = f"{CONTENT_DIR}/pages_meta.json"

# ----------------------------------------------------------------------
# FastAPI
# ----------------------------------------------------------------------
from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

app = FastAPI(title="ScriptBees Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------------------------------------------------------------
# API Request/Response Models
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
# Security
# ----------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(req: Request, api_key: str = Security(api_key_header)):
    incoming = api_key or ""
    if not incoming:
        auth = req.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            incoming = auth.split(" ", 1)[1]
    if incoming != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return incoming

# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------
cache = {}
def ckey(q): return hashlib.md5(q.lower().encode()).hexdigest()

# ----------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------
retriever = None
generator = None

@app.on_event("startup")
async def startup():
    global retriever, generator

    logger.info("🚀 Starting ScriptBees Assistant")

    import faiss
    from sentence_transformers import SentenceTransformer

    # --------------------- Retriever ---------------------
    class Retriever:
        def __init__(self):
            logger.info("📦 Loading FAISS...")
            self.model = SentenceTransformer(MODEL_NAME)
            self.index = faiss.read_index(INDEX_PATH)

            with open(META_PATH) as f:
                self.meta = json.load(f)

            with open(PAGES_PATH) as f:
                pages = json.load(f)

            self.pages = {p["id"]: p for p in pages}

        def retrieve(self, question: str):
            v = self.model.encode([question], normalize_embeddings=True).astype("float32")
            scores, indices = self.index.search(v, TOP_K)

            results = []
            for s, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                meta = self.meta[idx]
                page = self.pages[meta["id"]]
                results.append({
                    "url": meta["url"],
                    "title": meta["title"],
                    "score": float(s),
                    "text": page["text"][:500]
                })
            return results

    # ----------------------- LLM -------------------------
    class LLMGenerator:
        def __init__(self):
            logger.info("🤖 Using GPT-4o-mini (SAFE MODE)")
            self.client = OpenAI(
                api_key=OPENAI_API_KEY,
                http_client=safe_http_client  # <--- FIX
            )

        def generate(self, question, docs):
            ctx = docs[0]["text"]

            prompt = f"""
You are ScriptBees Assistant.
Answer ONLY from this context:

{ctx}

Question: {question}

Short and accurate answer:
"""

            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )

            text = res.choices[0].message.content.strip()
            return text + f"\n\n[Source: {docs[0]['url']}]"

    retriever = Retriever()
    generator = LLMGenerator()

    logger.info("✅ Assistant Ready")

# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, api_key: str = Depends(verify_api_key)):
    start = time.time()

    if ckey(req.question) in cache:
        r = cache[ckey(req.question)]
        r["cached"] = True
        r["response_time_seconds"] = time.time() - start
        return r

    docs = retriever.retrieve(req.question)
    if not docs:
        return AskResponse(answer="No info found.", sources=[], retrieved=[], cached=False, response_time_seconds=0)

    answer = generator.generate(req.question, docs)
    sources = [d["url"] for d in docs]

    resp = {
        "answer": answer,
        "sources": sources,
        "retrieved": [Source(**d) for d in docs],
        "cached": False,
        "response_time_seconds": time.time() - start
    }

    cache[ckey(req.question)] = resp
    return resp

@app.get("/")
async def root():
    return {"service": "ScriptBees RAG", "status": "online"}

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)