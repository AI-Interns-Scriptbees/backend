# debug_genai_resp.py
import os, json
from pathlib import Path
try:
    from google import genai
except Exception as e:
    print("genai import failed:", e)
    raise SystemExit(1)

CONTENT_DIR = Path("content")
PAGES_FILE = CONTENT_DIR / "pages.json"
if not PAGES_FILE.exists():
    print("content/pages.json missing; run embedder quick fastpath or create pages.json first.")
    raise SystemExit(1)

pages = json.loads(PAGES_FILE.read_text(encoding="utf8"))
texts = [(p.get("title","") + "\n" + p.get("text",""))[:8000] for p in pages]

key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=key) if key else genai.Client()

# send only first text (safe)
sample = texts[0]
print("Sending single sample to models.embed_content (model from env EMBED_MODEL or 'gemini-embedding-001')...")
model = os.getenv("EMBED_MODEL", "gemini-embedding-001")
resp = client.models.embed_content(model=model, contents=[sample])

print("\n--- Response type info ---")
print("type(resp) =", type(resp))
# print attributes
attrs = [a for a in dir(resp) if not a.startswith("_")]
print("attrs on resp (some):", attrs[:40])

# If dict-like
try:
    if isinstance(resp, dict):
        print("\nResponse keys:", list(resp.keys()))
        for k in ("embeddings","data","results","candidates"):
            if k in resp:
                print(f"First element under {k}: type={type(resp[k][0])}")
                print("repr:", repr(resp[k][0])[:800])
except Exception as e:
    print("dict inspect error:", e)

# Try attribute-based inspect
for name in ("embeddings","data","candidates"):
    try:
        val = getattr(resp, name, None)
        if val is not None:
            print(f"\nresp.{name} type={type(val)} len-or-attrs={getattr(val,'__len__',None)}")
            try:
                print("repr first item:", repr(val[0])[:1200])
            except Exception as e:
                print("cannot repr first item:", e)
    except Exception as e:
        pass

print("\nFull repr(resp) (truncated):\n", repr(resp)[:5000])
