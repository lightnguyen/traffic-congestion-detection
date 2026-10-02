# TrafficVision backend

FastAPI backend for the dashboard in `../src/App.tsx`. It receives images/videos, extracts review frames, runs a local YOLO or RT-DETR traffic labeler first, and can send frames to a local Ollama vision model (or optionally Gemini/GPT) for a detailed review. Results/settings/logs are stored in SQLite.

## Run locally

Run these commands from the `Web traffic monitoring UI\backend` folder (the folder containing `main.py`). In PowerShell, first go there from the project root:

```powershell
Set-Location ".\Web traffic monitoring UI\backend"
```

The backend must install packages into the **same Python interpreter that starts Uvicorn**. Capture the interpreter selected by `python`, then use that executable for installation, verification, and startup:

```powershell
$backendPython = (python -c "import sys; print(sys.executable)").Trim()
Write-Output "Using Python: $backendPython"
& $backendPython -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Verify Ultralytics from the same interpreter before starting the API:

```powershell
& $backendPython -c "import sys, ultralytics; print(sys.executable); print(ultralytics.__version__)"
```

If the running API reports `Python <path> ... chưa cài ultralytics`, that `<path>` is the interpreter actually serving requests. Stop Uvicorn and install into that exact interpreter, even if it is different from `$backendPython`:

```powershell
$backendPython = "<paste the full path shown in the API error>"
& $backendPython -m pip install -r requirements.txt
& $backendPython -c "import sys, ultralytics; print(sys.executable); print(ultralytics.__version__)"
```

Then restart Uvicorn with `& $backendPython -m uvicorn ...`. This avoids installing into one Python while the backend continues running from another.

Use the same terminal and activated environment for Uvicorn. Checking `pip show ultralytics` in a different Python environment does not confirm that the backend can import it.

Install and start [Ollama](https://ollama.com/download), then download and start the Qwen vision model:

```powershell
ollama pull qwen2.5vl:7b
ollama run qwen2.5vl:7b
```

Keep Ollama running. For NVIDIA GPU inference, start the backend with a Python installation that has CUDA-enabled PyTorch. On this Windows setup, use the CUDA-enabled Python 3.11 installation explicitly (the project `.venv` currently contains CPU-only PyTorch):

```powershell
& $backendPython -m uvicorn main:app --app-dir . --host 127.0.0.1 --port 8000
```

To check the selected runtime before starting, run `& $backendPython -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"`. The output should show a CUDA build, `True`, and the NVIDIA GPU name. The `--reload` option is omitted here because its reloader child selected CPU in this setup.

If you prefer a virtual environment, create it from the CUDA Python and both install and launch through that environment instead: `& $backendPython -m venv .venv`, `& .\.venv\Scripts\python.exe -m pip install -r requirements.txt`, then `& .\.venv\Scripts\python.exe -m uvicorn main:app --app-dir . --host 127.0.0.1 --port 8000`. In that case, install a CUDA-enabled PyTorch build in `.venv` as well.

Start the Vite UI in a second PowerShell terminal. The working directory must be the `Web traffic monitoring UI` folder, which contains `package.json` (not its `backend` subfolder):

```powershell
Set-Location ".."
pnpm.cmd install
pnpm.cmd dev
```

Vite proxies `/api` to `http://127.0.0.1:8000` (override with `TRAFFIC_API_URL`). Open the Vite URL shown in the terminal. API docs are at `http://127.0.0.1:8000/docs`.

## Configuration

- Default Vision provider: Ollama at `http://127.0.0.1:11434`, model `qwen2.5vl:7b`; no API key required.
- Video bytes are saved first and the upload response returns before decoding or inference. Background processing then samples frames sequentially at up to 30 FPS (capped by source FPS and `max_frames`), tracks vehicles, and combines detection count with bounding-box occupancy to derive the traffic label. CUDA is used when available; actual inference speed depends on hardware and model.
- Qwen/Gemini/GPT analyzes the first available video sample immediately while decoding continues, then updates the session-level Vision result at the configured video-time interval (5 seconds by default). The current Vision result is stored on the session and shown beside the image; it is not written into each frame's JSON. Detector output remains per frame.
- The original video is playable as soon as upload finishes. Frame-wide/manual Vision jobs run in the background, and the dashboard polls the session so each new result appears without waiting for the full loop. Video decoding, detector inference, and first-frame Vision can overlap.
- Analysis frames are resized to a maximum edge of 640 px and JPEG quality 75 before inference. Video detector sampling defaults to 30 FPS, with a 30,000-frame ceiling. Ollama output is capped at 256 tokens to reduce generation time.
- `OLLAMA_BASE_URL`: optional Ollama server URL (defaults to `http://127.0.0.1:11434`).
- `GEMINI_API_KEY` or `OPENAI_API_KEY`: optional credentials if selecting a hosted provider. Keys stay on the backend.
- `TRAFFIC_DATA_DIR`: folder searched by the UI's “Mở video data/” selector. Defaults to the workspace `data` directory.
- `TRAFFIC_STORE_DIR`: uploaded media, extracted JPEG frames, and the SQLite database. Defaults to `backend/storage`.
- `TRAFFIC_DB_PATH`: optional explicit SQLite file path.
- `CORS_ORIGINS`: comma-separated origins if the UI is hosted separately. Same-origin Vite proxy is the default setup.

To configure camera feeds, create `data/cameras.json` as an array such as:

```json
[
  {"id": 1, "name": "Camera trung tâm", "location": "Q.1", "status": "normal", "source": 0},
  {"id": 2, "name": "Camera RTSP", "location": "Q.3", "status": "normal", "source": "rtsp://user:password@host/stream"}
]
```

`source` may be a webcam index or RTSP URL. The capture button returns a helpful configuration error until a source is set.

## API routes

- `GET /api/health`, `/api/cameras`, `/api/media`, `/api/history`, `/api/logs`, `/api/settings`
- `POST /api/sessions` accepts image/video multipart upload and extracts sampled frames.
- `POST /api/sessions/from-library` processes a video inside the configured data folder.
- `GET /api/sessions`, `/api/sessions/{id}`, `/api/sessions/{id}/frames/{frame_id}/image`
- `POST /api/sessions/{id}/analyze` queues Vision analysis for one frame (`frame_id`) or the session's exposed sampled frames and returns immediately.
- `PUT /api/settings`, `POST /api/alerts`, `GET /api/sessions/{id}/export.json`
- `GET /api/cameras/{id}/capture` grabs a still from a configured camera source.

Video uploads are sampled across the clip at up to the configured detector FPS (default 5 FPS for new databases), with a 30,000-frame ceiling. Vision AI updates the session-level result every 5 seconds by default. SQLite setup and persistence helpers live in `database.py`; API routes and media orchestration remain in `main.py`.

Analysis estimates visible conditions from sampled frames; vehicle counts and flood coverage should be treated as model estimates. This development server has no user authentication, so keep it bound to localhost or place it behind an authenticated gateway before exposing it to a network.
