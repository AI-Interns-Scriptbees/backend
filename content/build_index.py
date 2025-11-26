import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "content"
os.makedirs(DATA_DIR, exist_ok=True)

INDEX_PATH = os.path.join(DATA_DIR, "pages.faiss")
PAGES_PATH = os.path.join(DATA_DIR, "pages.json")
META_PATH = os.path.join(DATA_DIR, "pages_meta.json")

# Your documents (replace with real data)
documents = [
    {
        "id": 0,
        "url": "https://example.com/page1",
        "title": "Sample Page 1",
        "text": "This is sample text for page one. This will be used for RAG testing."
    },
    {
        "id": 1,
        "url": "https://example.com/page2",
        "title": "Sample Page 2",
        "text": "Another test document for building FAISS index. This contains RAG information."
    }
]

# Save pages.json (full documents)
with open(PAGES_PATH, "w", encoding="utf8") as f:
    json.dump(documents, f, indent=2)

# Create meta info
pages_meta = [
    {"id": doc["id"], "url": doc["url"], "title": doc["title"]}
    for doc in documents
]

with open(META_PATH, "w", encoding="utf8") as f:
    json.dump(pages_meta, f, indent=2)

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([doc["text"] for doc in documents])

# FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings))

faiss.write_index(index, INDEX_PATH)

print("✓ All files created:")
print(" - pages.faiss")
print(" - pages.json")
print(" - pages_meta.json")
