# scriptbees_gemini.py
"""
SCRIPTBEES ASSISTANT - Gemini (Google GenAI) version
Replaces OpenAI client with google.genai client.
Keep environment variable GEMINI_API_KEY (or use ADC).
"""
import os
import json
import time
import logging
import hashlib
from typing import List
from pathlib import Path

# ------------------------------
# Remove proxy variables (Render injects these)
# ------------------------------
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# ------------------------------
# Load env (optionally from .env for local dev)
# ------------------------------
from dotenv import load_dotenv

def find_env():
    cur = Path(__file__).resolve().parent
    for _ in range(10):
        if (cur / ".env").exists():
            return cur / ".env"
        cur = cur.parent
    return None

env = find_env()
if env:
    load_dotenv(env)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # preferred env var for Google GenAI API key
API_KEY = os.getenv("RAG_API_KEY", "change-me")

# FRONTEND_ORIGINS logic (unchanged)
_frontend_env = os.getenv("FRONTEND_ORIGINS", "*").strip()
if _frontend_env == "":
    FRONTEND_ORIGINS = ["*"]
elif _frontend_env == "*":
    FRONTEND_ORIGINS = ["*"]
else:
    FRONTEND_ORIGINS = [o.strip() for o in _frontend_env.split(",") if o.strip()]

# If you require an API key, enforce it (optional)
# We allow running without GEMINI_API_KEY if using ADC; but if you want to require it, uncomment:
# if not GEMINI_API_KEY:
#     raise SystemExit("Missing GEMINI_API_KEY environment variable. Set GEMINI_API_KEY or configure ADC and redeploy.")

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scriptbees")

# ------------------------------
# Config
# ------------------------------
CONTENT_DIR = "content"
MODEL_NAME = "gemini-2.5-flash"  # choose an available Gemini model; change if needed
TOP_K = 1
MAX_TOKENS = 150
TEMPERATURE = 0.2

INDEX_PATH = f"{CONTENT_DIR}/pages.faiss"
META_PATH = f"{CONTENT_DIR}/pages_meta.json"
PAGES_PATH = f"{CONTENT_DIR}/pages.json"

# ------------------------------
# FastAPI App
# ------------------------------
from fastapi import FastAPI, Depends, HTTPException, Security, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

app = FastAPI(title="ScriptBees Assistant — Gemini Version")

_allow_credentials = True if FRONTEND_ORIGINS != ["*"] else False

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

# ------------------------------
# Models
# ------------------------------
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

# ------------------------------
# API Key Security
# ------------------------------
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

# ------------------------------
# Cache
# ------------------------------
cache = {}
def ckey(q): return hashlib.md5(q.lower().encode()).hexdigest()

# ------------------------------
# Startup: Load FAISS + Model + Gemeni client
# ------------------------------
retriever = None
generator = None

@app.on_event("startup")
async def startup():
    global retriever, generator
    logger.info("🚀 Starting ScriptBees AI Assistant (Gemini)...")

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        logger.exception("Failed to import FAISS / sentence_transformers. Make sure dependencies are installed.")
        raise SystemExit("Missing FAISS or sentence-transformers dependencies: " + str(e))

    # Retriever (unchanged logic)
    class Retriever:
        def __init__(self):
            logger.info("📦 Loading FAISS + metadata...")
            if not Path(INDEX_PATH).exists():
                raise SystemExit(f"Missing FAISS index at {INDEX_PATH}")
            if not Path(META_PATH).exists():
                raise SystemExit(f"Missing metadata file at {META_PATH}")
            if not Path(PAGES_PATH).exists():
                raise SystemExit(f"Missing pages file at {PAGES_PATH}")

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.index = faiss.read_index(INDEX_PATH)

            with open(META_PATH, "r") as f:
                self.meta = json.load(f)
            with open(PAGES_PATH, "r") as f:
                pages = json.load(f)
            self.pages = {p["id"]: p for p in pages}
            logger.info(f"✓ Loaded {getattr(self.index, 'ntotal', 'unknown')} ScriptBees pages")

        def retrieve(self, question):
            vec = self.model.encode([question], normalize_embeddings=True).astype("float32")
            scores, idxs = self.index.search(vec, TOP_K)
            results = []
            for s, idx in zip(scores[0], idxs[0]):
                if idx == -1:
                    continue
                meta = self.meta[idx]
                page = self.pages.get(meta["id"], {})
                results.append({
                    "url": meta.get("url", ""),
                    "title": meta.get("title", ""),
                    "score": float(s),
                    "text": page.get("text", "")[:1200]
                })
            return results

    # Generator using Google GenAI (Gemini)
    class LLMGenerator:
        def __init__(self):
            # Use google.genai client. If GEMINI_API_KEY present we use it,
            # otherwise client will fallback to ADC if configured on the environment.
            try:
                # google-genai library
                from google import genai
            except Exception as e:
                logger.exception("google.genai client not installed. Install 'google-genai' package.")
                raise SystemExit("Missing google-genai library: " + str(e))

            if GEMINI_API_KEY:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
            else:
                # fallback to ADC (Application Default Credentials)
                self.client = genai.Client()

        def generate(self, question, docs):
            context = docs[0]["text"]
            prompt = f"""You are ScriptBees AI Assistant.
Answer ONLY using this ScriptBees content:
{context}

Question: {question}

Give a short and correct answer based ONLY on ScriptBees website."""
            try:
                # model selection: change MODEL_NAME if you prefer a different Gemini model
                resp = self.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[prompt],
                    max_output_tokens=MAX_TOKENS
                )
            except Exception as e:
                msg = str(e)
                logger.exception("Gemini (Google GenAI) request failed")
                if "401" in msg or "unauthorized" in msg.lower():
                    raise HTTPException(status_code=502, detail="Upstream Gemini authentication error")
                if "429" in msg or "rate limit" in msg.lower():
                    raise HTTPException(status_code=429, detail="Upstream Gemini rate limit")
                raise HTTPException(status_code=502, detail=f"Upstream Gemini error: {msg}")

            # Extract text
            answer_text = ""
            try:
                # response typically has .text or choices; guard both ways
                answer_text = getattr(resp, "text", None) or resp.get("candidates", [{}])[0].get("content", "")
                if isinstance(answer_text, dict):
                    # some responses may nest it
                    answer_text = answer_text.get("text", "") or str(answer_text)
                answer_text = str(answer_text).strip()
            except Exception:
                logger.exception("Failed to parse Gemini response")

            if not answer_text:
                answer_text = "No answer returned from upstream."
            return answer_text

    retriever = Retriever()
    generator = LLMGenerator()
    logger.info("✅ ScriptBees Assistant (Gemini) is READY")

# middleware to log incoming path (helps debug double-slash issues)
@app.middleware("http")
async def log_path(request: Request, call_next):
    logger.info("Incoming request: %s %s", request.method, request.url.path)
    return await call_next(request)

# API Route
@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, key: str = Depends(verify_api_key)):
    start = time.time()

    # Check cache
    h = ckey(req.question)
    if h in cache:
        r = cache[h]
        r["cached"] = True
        r["response_time_seconds"] = time.time() - start
        return r

    docs = retriever.retrieve(req.question)
    if not docs:
        return AskResponse(
            answer="No matching information found on ScriptBees.",
            sources=[],
            retrieved=[],
            cached=False,
            response_time_seconds=time.time() - start
        )

    answer = generator.generate(req.question, docs)

    first = docs[0]
    source_obj = Source(url=first.get("url", ""), title=first.get("title", ""), score=first.get("score", 0.0))

    resp = {
        "answer": answer,
        "sources": [first.get("url", "")],
        "retrieved": [source_obj],
        "cached": False,
        "response_time_seconds": time.time() - start
    }

    cache[h] = resp
    return resp

@app.get("/")
async def home():
    return {"status": "online", "bot": "ScriptBees AI (Gemini)"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
