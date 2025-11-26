
# Fixed Chat Bot Project (Ready-to-run)

This repo is a corrected, ready-to-run version of your chat bot project. It includes:
- Static frontend with chat UI and **voice input + TTS** (Web Speech API).
- Backend proxy for local models (Ollama) at `server/server_local.py`.
- Scraper configured to crawl `https://scriptbees.com/` (edit as needed).
- FAISS index files already present in `content/` (kept from your upload).
- Fixed folder structure.

## Quick start (Windows PowerShell)

1. Open PowerShell and navigate to project folder (where `main.py` is):

```powershell
cd "C:\path\to\fixed_chat_bot"
```

2. (Optional) Create and activate virtualenv:

```powershell
python -m venv bot
.ot\Scripts\Activate.ps1
```

3. Install requirements:

```powershell
python -m pip install -r requirements.txt
pip install fastapi uvicorn[standard] requests
```

4. Run the backend server (this project provides `main.py` which should start the app):

```powershell
python main.py
```

If you prefer to run the smaller local-proxy for Ollama:

```powershell
pip install fastapi uvicorn[standard] requests
uvicorn server.server_local:app --host 0.0.0.0 --port 8000
```

5. Serve the frontend (if you run `main.py` it may serve already). Alternatively:

```powershell
cd frontend
python -m http.server 8001
# Open http://localhost:8001 in your browser
```

## Voice support
- The frontend uses the Web Speech API:
  - **SpeechRecognition** for voice input (Chrome/Edge on desktop recommended).
  - **speechSynthesis** for TTS (most browsers).
- Click the 🎤 mic button to start/stop voice recording. Finalized speech will be submitted as a question automatically.

## Local LLMs
- This project expects a local LLM like **Ollama** to be running at `http://localhost:11434`.
- If you don't have Ollama installed, the server will return HTTP 501 with instructions.
- You can modify `server/server_local.py` to integrate **gpt4all** or **llama.cpp** servers.

## Scraper
- `scraper/scraper.py` is included and set to crawl `https://scriptbees.com/` by default.
- If the target blocks scraping, run with `--url https://example.com`.

## Notes & Troubleshooting
- Make sure to run the server from the folder that contains `main.py`.
- Ensure `content/pages.faiss` exists (it is included). If not, run:
  - `python scraper/scraper.py --url https://scriptbees.com --max-pages 100`
  - `python embeddings/embedder.py`

