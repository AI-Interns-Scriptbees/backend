from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os, subprocess, json, asyncio

app = FastAPI()

class AskRequest(BaseModel):
    question: str

# Simple endpoint used by frontend: POST /api/ask
@app.post("/api/ask")
async def ask(req: AskRequest):
    question = req.question
    # Prefer Ollama local API if available
    try:
        import requests
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3",
            "prompt": question,
            "max_tokens": 200
        }, timeout=15)
        if res.ok:
            data = res.json()
            return {"answer": data.get("text", "").strip(), "sources": []}
    except Exception as e:
        # fallback to a simple echo composition using content/pages.json if available
        pass

    # Fallback: no local model running — return helpful error
    raise HTTPException(status_code=501, detail="No local LLM available. Install Ollama and run a model, or configure server_local.py to use gpt4all/llama.cpp.")
