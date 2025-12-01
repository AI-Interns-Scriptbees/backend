#!/usr/bin/env python3
import os
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from google import genai   # google-genai v1beta

# =========================
# CONFIG
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = "text-embedding-004"
BATCH_SIZE = 32
SLEEP_TIME = 0.2

CONTENT_DIR = Path("content")
PAGES_FILE = CONTENT_DIR / "pages.json"
EMBED_FILE = CONTENT_DIR / "embeddings.npy"


# =========================
# LOAD CONTENT
# =========================
if not PAGES_FILE.exists():
    raise SystemExit(f"❌ pages.json not found at {PAGES_FILE}")

with open(PAGES_FILE, "r", encoding="utf-8") as f:
    pages = json.load(f)

texts = []
for p in pages:
    t = p.get("text") or p.get("content") or p.get("snippet") or ""
    t = str(t).strip()
    if t == "":
        t = "N/A"
    texts.append(t)

print(f"Loaded {len(texts)} pages.")


# =========================
# CLIENT
# =========================
client = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# EMBEDDING FUNCTION (FINAL)
# =========================
def embed_batch(batch):
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=batch
    )

    vectors = []

    for item in resp.embeddings:
        # Prefer official field
        if hasattr(item, "embedding"):
            vectors.append(item.embedding)

        # Fallback: some SDK versions use item.values
        elif hasattr(item, "values"):
            vectors.append(item.values)

        # Fallback: raw dict
        elif isinstance(item, dict):
            vectors.append(item.get("embedding") or item.get("values"))

        else:
            raise RuntimeError(
                "❌ Could not parse embedding item: " + repr(item)
            )

    return vectors


# =========================
# MAIN LOOP
# =========================
all_vecs = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
    batch = texts[i:i + BATCH_SIZE]

    vecs = embed_batch(batch)

    for v in vecs:
        arr = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        all_vecs.append(arr)

    time.sleep(SLEEP_TIME)


# =========================
# SAVE RESULTS
# =========================
final = np.vstack(all_vecs).astype(np.float32)
np.save(EMBED_FILE, final)

print("=======================================")
print("✔ Embedding completed successfully!")
print("✔ Saved to:", EMBED_FILE)
print("✔ Shape:", final.shape)
print("=======================================")
