from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import threading
import time
import uuid
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool
from database import DB_PATH, STORE_DIR, connect, get_settings, init_db, log

load_dotenv(Path(__file__).parent / ".env")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("TRAFFIC_DATA_DIR", str(ROOT / "data"))).resolve()
UPLOAD_DIR = STORE_DIR / "uploads"
FRAME_DIR = STORE_DIR / "frames"
for folder in (UPLOAD_DIR, FRAME_DIR, DB_PATH.parent):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="TrafficVision API", version="1.0.0", description="Backend for the traffic monitoring dashboard")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8443").split(","), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FRAME_MAX_EDGE = 640
FRAME_JPEG_QUALITY = 75
DEFAULT_CAMERAS = [
    {"id": 1, "name": "Nguyễn Văn Cừ - Cầu Chà Và", "location": "Q.5 ↔ Q.8", "status": "normal", "source": None},
    {"id": 2, "name": "Đinh Tiên Hoàng - Hồng Thập Tự", "location": "Q.1 ↔ Q.Bình Thạnh", "status": "normal", "source": None},
    {"id": 3, "name": "Nguyễn Thị Minh Khai - Cách Mạng Tháng 8", "location": "Q.1 ↔ Q.3", "status": "normal", "source": None},
    {"id": 4, "name": "Võ Văn Kiệt - Hàm Nghi", "location": "Q.1", "status": "normal", "source": None},
]
PROMPT = """Phân tích ảnh camera giao thông, chỉ trả về MỘT JSON object theo schema:
{"traffic":{"is_congested":boolean,"congestion_level":"clear|light|moderate|heavy|severe|unknown","congestion_level_vi":string,"vehicle_density":"Thấp|Trung bình|Cao|Rất cao|Không rõ","vehicles":{"motorcycle":integer|null,"car":integer|null,"bus_truck":integer|null}},"flood":{"is_flooded":boolean,"flood_area_percent":number|null,"status_vi":string},"weather":"sunny|cloudy|rainy|foggy|unknown","weather_vi":string,"safety_alert":string,"ai_description":string}
Chỉ ước lượng những gì nhìn thấy rõ; dùng null khi không thể đếm đáng tin cậy. Không suy đoán địa điểm. Không coi mặt đường tối là ngập nếu thiếu bằng chứng."""


@app.on_event("startup")
def startup():
    init_db()
    log("TrafficVision API initialized")


class SettingsUpdate(BaseModel):
    provider: str | None = None
    ollama_model: str | None = None
    gemini_model: str | None = None
    gpt_model: str | None = None
    detector_model: str | None = None
    sample_every_seconds: float | None = Field(default=None, gt=0, le=3600)
    llm_sample_every_seconds: float | None = Field(default=None, gt=0, le=3600)
    detector_fps: int | None = Field(default=None, ge=1, le=30)
    max_frames: int | None = Field(default=None, ge=1, le=30000)
    motorcycle_alert_threshold: int | None = Field(default=None, ge=1, le=10000)


class AlertRequest(BaseModel):
    session_id: str | None = None
    frame_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)
    channel: str = "local"


def public_session(session_id: str) -> dict[str, Any]:
    frame_stride = max(1, min(30, int(get_settings().get("detector_fps", 30))))
    with connect() as db:
        row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Không tìm thấy phiên phân tích")
        frames = db.execute("SELECT id,frame_index,timestamp_sec,analysis_json,analyzed_at,yolo_json FROM frames WHERE session_id=? AND (analysis_json IS NOT NULL OR frame_index % ? = 0) ORDER BY frame_index", (session_id, frame_stride)).fetchall()
    return {"id": row["id"], "camera_id": row["camera_id"], "filename": row["filename"], "media_type": row["media_type"], "created_at": row["created_at"], "duration": row["duration"], "width": row["width"], "height": row["height"], "provider": row["provider"], "model": row["model"], "processing_status": row["processing_status"], "processing_error": row["processing_error"], "llm_analysis": json.loads(row["llm_analysis_json"]) if row["llm_analysis_json"] else None, "llm_updated_at": row["llm_updated_at"], "llm_status": row["llm_status"], "frames": [{"id": f["id"], "frame_index": f["frame_index"], "timestamp_sec": f["timestamp_sec"], "analyzed": f["analysis_json"] is not None, "analysis": json.loads(f["analysis_json"]) if f["analysis_json"] else None, "yolo": ({key: value for key, value in json.loads(f["yolo_json"]).items() if key in {"model", "label", "vehicle_counts", "vehicle_count", "person_count", "occupancy", "device", "inference_ms", "tracked_vehicle_ids", "detected_at"}} if f["yolo_json"] and "label" in json.loads(f["yolo_json"]) else None), "image_url": f"/api/sessions/{session_id}/frames/{f['id']}/image"} for f in frames]}


def add_frame(session_id: str, index: int, timestamp: float, data: bytes):
    frame_id = f"{session_id}_{index:05d}"
    path = FRAME_DIR / f"{frame_id}.jpg"
    path.write_bytes(data)
    with connect() as db:
        db.execute("INSERT INTO frames(id,session_id,frame_index,timestamp_sec,image_path) VALUES (?,?,?,?,?)", (frame_id, session_id, index, timestamp, str(path)))


_detector_models: dict[str, Any] = {}


def run_yolo_for_session(session_id: str):
    """Track vehicles in sampled frames and classify traffic by box occupancy."""
    settings = get_settings()
    model_name = settings.get("detector_model", "yolo11s.pt")
    detector = _detector_models.get(model_name)
    if detector is None:
        try:
            from ultralytics import YOLO
            if model_name.startswith("rtdetr-"):
                from ultralytics import RTDETR
        except ModuleNotFoundError as exc:
            if exc.name == "ultralytics":
                detail = f"Backend đang chạy bằng Python {sys.executable}, môi trường này chưa cài ultralytics. Cài bằng đúng interpreter: python -m pip install -r requirements.txt (đang ở thư mục backend)."
            else:
                detail = f"Không nạp được dependency '{exc.name}' của ultralytics trong Python {sys.executable}: {exc}"
            raise HTTPException(503, detail) from exc
        except ImportError as exc:
            raise HTTPException(503, f"Ultralytics gặp lỗi import trong Python {sys.executable}: {exc}") from exc
        if model_name == "yolo11s.pt":
            configured = os.getenv("YOLO_MODEL_PATH")
            model_source = str(Path(configured)) if configured else str(ROOT / model_name)
            if not Path(model_source).is_file():
                raise HTTPException(503, f"Không tìm thấy model detector: {model_source}")
        else:
            model_source = model_name
        detector = RTDETR(model_source) if model_name.startswith("rtdetr-") else YOLO(model_source)
        _detector_models[model_name] = detector
    with connect() as db:
        frames = db.execute("SELECT id,image_path FROM frames WHERE session_id=? ORDER BY frame_index", (session_id,)).fetchall()
    try:
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    batch_size = 8
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start:start + batch_size]
        batch_images = [cv2.imread(frame["image_path"]) for frame in batch_frames]
        valid = [(frame, image) for frame, image in zip(batch_frames, batch_images) if image is not None]
        if not valid:
            continue
        # Batch inference reduces the per-call overhead that made later frames
        # crawl while preserving tracker updates in frame order.
        predictions = detector.track([image for _, image in valid], device=device, verbose=False, conf=0.20, persist=True, tracker="bytetrack.yaml", batch=batch_size, imgsz=640)
        updates = []
        for frame, image, prediction in zip((item[0] for item in valid), (item[1] for item in valid), predictions):
            occupied_area = 0.0
            counts = {"motorcycle": 0, "car": 0, "bus_truck": 0}
            person_count = 0
            tracked_ids = set()
            if prediction.boxes is not None:
                ids = prediction.boxes.id.int().tolist() if prediction.boxes.id is not None else []
                for index, (box, class_id) in enumerate(zip(prediction.boxes.xyxy.tolist(), prediction.boxes.cls.int().tolist())):
                    if class_id == 0:
                        person_count += 1
                    if class_id in (2, 3, 5, 7):
                        x1, y1, x2, y2 = box
                        occupied_area += max(0, x2 - x1) * max(0, y2 - y1)
                        if class_id == 3:
                            counts["motorcycle"] += 1
                        elif class_id == 2:
                            counts["car"] += 1
                        else:
                            counts["bus_truck"] += 1
                        if index < len(ids):
                            tracked_ids.add(ids[index])
            occupancy = occupied_area / max(1, image.shape[0] * image.shape[1])
            vehicle_count = sum(counts.values())
            # Small motorcycles are often missed at distance. Crowd detections
            # strengthen the traffic signal only when vehicle evidence exists.
            label = "Ùn tắc" if occupancy >= 0.10 or vehicle_count >= 14 or (person_count >= 30 and vehicle_count >= 6) else "Đông đúc" if occupancy >= 0.035 or vehicle_count >= 6 or (person_count >= 14 and vehicle_count >= 3) else "Thông thoáng"
            result = {"model": model_name, "label": label, "vehicle_counts": counts, "vehicle_count": vehicle_count, "person_count": person_count, "occupancy": round(occupancy, 4), "device": str(device), "inference_ms": round(float((prediction.speed or {}).get("inference", 0.0)), 2), "tracked_vehicle_ids": sorted(tracked_ids), "detected_at": datetime.now(timezone.utc).isoformat()}
            updates.append((json.dumps(result, ensure_ascii=False), frame["id"]))
        with connect() as db:
            db.executemany("UPDATE frames SET yolo_json=? WHERE id=?", updates)
    log(f"{model_name} traffic labels ready for {len(frames)} sampled frame(s)", "INFO", session_id)


def inspect_media(path: Path, session_id: str, sample_every: float, max_frames: int) -> dict[str, Any]:
    max_frames = max(1, int(max_frames))
    detector_fps = min(30.0, max(1.0, float(get_settings().get("detector_fps", 30))))

    def save_frame(frame, output_index: int, timestamp: float) -> bool:
        height, width = frame.shape[:2]
        if max(width, height) > FRAME_MAX_EDGE:
            scale = FRAME_MAX_EDGE / max(width, height)
            frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY])
        if ok:
            add_frame(session_id, output_index, timestamp, encoded.tobytes())
        return bool(ok)

    if path.suffix.lower() in IMAGE_EXTS:
        image = cv2.imread(str(path))
        if image is None:
            raise HTTPException(400, "Cannot read uploaded image")
        height, width = image.shape[:2]
        if not save_frame(image, 0, 0.0):
            raise HTTPException(400, "Cannot encode uploaded image")
        return {"media_type": "image", "duration": 0.0, "width": width, "height": height}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(400, "Cannot open video; check its codec and format")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = count / fps if fps > 0 and count > 0 else None
    added = 0
    try:
        if duration is not None:
            desired = min(max_frames, max(1, int(duration * detector_fps) + 1), count)
            if desired == 1:
                indices = [0]
            else:
                indices = [round(i * (count - 1) / (desired - 1)) for i in range(desired)]
            selected = set(indices)
            for index in range(count):
                ok, frame = cap.read()
                if not ok:
                    break
                if index in selected and save_frame(frame, added, index / fps):
                    added += 1
        else:
            index = 0
            step = max(1, round(fps * sample_every)) if fps > 0 else 1
            while added < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % step == 0 and save_frame(frame, added, index / fps if fps > 0 else float(index)):
                    added += 1
                index += 1
    finally:
        cap.release()
    if added == 0:
        raise HTTPException(400, "Video has no readable frames")
    return {"media_type": "video", "duration": duration, "width": width, "height": height}

def analyze_image(image_path: Path, provider: str, model: str) -> dict[str, Any]:
    raw = image_path.read_bytes()
    # Older sessions may still contain 960 px frames. Resize at inference too,
    # so those existing frames receive the same speed optimization.
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is not None:
        height, width = image.shape[:2]
        if max(width, height) > FRAME_MAX_EDGE:
            scale = FRAME_MAX_EDGE / max(width, height)
            image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY])
        if ok:
            raw = encoded.tobytes()
    prompt = PROMPT
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(raw).decode("ascii")]}],
            "format": "json",
            "stream": False,
            # Keep answers compact; long free-form descriptions slow generation
            # without improving the structured traffic estimates much.
            "options": {"temperature": 0, "num_predict": 256},
        }).encode("utf-8")
        request = urllib.request.Request(f"{base_url}/api/chat", data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = json.loads(response.read().decode("utf-8"))
            text = body.get("message", {}).get("content", "{}")
            total_s = body.get("total_duration", 0) / 1e9
            prompt_s = body.get("prompt_eval_duration", 0) / 1e9
            output_s = body.get("eval_duration", 0) / 1e9
            output_tokens = body.get("eval_count", 0)
            output_rate = output_tokens / output_s if output_s > 0 else 0
            log(f"Ollama timing: total={total_s:.2f}s, image+prompt={prompt_s:.2f}s, generation={output_s:.2f}s ({output_rate:.1f} tokens/s)")
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise HTTPException(503, f"Không kết nối được Ollama tại {base_url}. Hãy khởi động Ollama và tải model {model}. Chi tiết: {reason}") from exc
        except (KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(502, "Ollama trả về phản hồi không hợp lệ") from exc
    elif provider == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise HTTPException(503, "Thiếu GEMINI_API_KEY. Hãy cấu hình backend/.env rồi khởi động lại.")
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=model, contents=[types.Part.from_bytes(data=raw, mime_type="image/jpeg"), prompt], config=types.GenerateContentConfig(response_mime_type="application/json"))
        text = response.text or "{}"
    elif provider == "gpt":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise HTTPException(503, "Thiếu OPENAI_API_KEY. Hãy cấu hình backend/.env rồi khởi động lại.")
        from openai import OpenAI
        client = OpenAI(api_key=key)
        data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}]}], response_format={"type": "json_object"})
        text = response.choices[0].message.content or "{}"
    else:
        raise HTTPException(400, "provider phải là ollama, gemini hoặc gpt")
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise HTTPException(502, "Model trả về JSON không hợp lệ")
        result = json.loads(text[start:end + 1])
    result["analyzed_at"] = datetime.now(timezone.utc).isoformat()
    result["model"] = model
    return result


def run_analysis(session_id: str, frame_id: str, provider: str, model: str) -> dict[str, Any]:
    with connect() as db:
        frame = db.execute("SELECT * FROM frames WHERE id=? AND session_id=?", (frame_id, session_id)).fetchone()
    if not frame:
        raise HTTPException(404, "Không tìm thấy frame")
    result = analyze_image(Path(frame["image_path"]), provider, model)
    with connect() as db:
        db.execute("UPDATE sessions SET llm_analysis_json=?,llm_updated_at=?,provider=?,model=? WHERE id=?", (json.dumps(result, ensure_ascii=False), result["analyzed_at"], provider, model, session_id))
    log(f"Frame {frame['frame_index']:04d} analyzed successfully ({model})", "INFO", session_id)
    return result


def auto_analyze_session(session_id: str, first_only: bool = False, skip_first: bool = False):
    """Run the configured LLM on frames spaced by the user's interval."""
    settings = get_settings()
    provider = settings.get("provider", "ollama")
    model = settings.get({"ollama": "ollama_model", "gemini": "gemini_model", "gpt": "gpt_model"}.get(provider, "ollama_model"), "qwen2.5vl:7b")
    interval = max(0.1, float(settings.get("llm_sample_every_seconds", 5)))
    with connect() as db:
        frames = db.execute("SELECT id,timestamp_sec FROM frames WHERE session_id=? ORDER BY frame_index", (session_id,)).fetchall()
        db.execute("UPDATE sessions SET llm_status='processing' WHERE id=?", (session_id,))
    next_at = interval if skip_first else 0.0
    completed = 0
    for frame in frames:
        if frame["timestamp_sec"] + 1e-6 < next_at:
            continue
        try:
            run_analysis(session_id, frame["id"], provider, model)
            completed += 1
            if first_only:
                break
            next_at = frame["timestamp_sec"] + interval
        except Exception as exc:
            log(f"Automatic LLM analysis failed: {exc}", "ERROR", session_id)
            next_at = frame["timestamp_sec"] + interval
    log(f"Automatic LLM analysis complete: {completed} frame(s), interval={interval:g}s", "INFO", session_id)
    with connect() as db:
        db.execute("UPDATE sessions SET llm_status='complete' WHERE id=?", (session_id,))


def process_video_session(session_id: str, path: Path, max_frames: int):
    """Process an uploaded video after returning the upload response."""
    try:
        with connect() as db:
            db.execute("UPDATE sessions SET processing_status='vision_processing',processing_error=NULL WHERE id=?", (session_id,))
        interval = 1.0 / max(1, min(30, get_settings().get("detector_fps", 30)))
        # Start Vision on the first extracted sample while video decoding continues.
        first_frame_task = threading.Thread(target=analyze_first_frame_when_ready, args=(session_id,), daemon=True)
        first_frame_task.start()
        metadata = inspect_media(path, session_id, interval, max_frames)
        with connect() as db:
            db.execute("UPDATE sessions SET duration=?,width=?,height=? WHERE id=?", (metadata["duration"], metadata["width"], metadata["height"], session_id))
        run_yolo_for_session(session_id)
        first_frame_task.join()
        with connect() as db:
            db.execute("UPDATE sessions SET processing_status='llm_processing' WHERE id=?", (session_id,))
        auto_analyze_session(session_id, skip_first=True)
        with connect() as db:
            db.execute("UPDATE sessions SET processing_status='complete' WHERE id=?", (session_id,))
    except Exception as exc:
        with connect() as db:
            db.execute("UPDATE sessions SET processing_status='failed',processing_error=? WHERE id=?", (str(exc)[:2000], session_id))
        log(f"Video processing failed: {exc}", "ERROR", session_id)


def analyze_first_frame_when_ready(session_id: str):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        with connect() as db:
            frame = db.execute("SELECT id FROM frames WHERE session_id=? ORDER BY frame_index LIMIT 1", (session_id,)).fetchone()
            session = db.execute("SELECT processing_status FROM sessions WHERE id=?", (session_id,)).fetchone()
        if frame:
            try:
                settings = get_settings()
                provider = settings.get("provider", "ollama")
                model = settings.get({"ollama": "ollama_model", "gemini": "gemini_model", "gpt": "gpt_model"}.get(provider, "ollama_model"), "qwen2.5vl:7b")
                with connect() as db:
                    db.execute("UPDATE sessions SET llm_status='processing' WHERE id=?", (session_id,))
                run_analysis(session_id, frame["id"], provider, model)
                with connect() as db:
                    db.execute("UPDATE sessions SET llm_status='complete' WHERE id=?", (session_id,))
            except Exception as exc:
                log(f"Initial frame Vision analysis failed: {exc}", "ERROR", session_id)
                with connect() as db:
                    db.execute("UPDATE sessions SET llm_status='failed' WHERE id=?", (session_id,))
            return
        if not session or session["processing_status"] == "failed":
            return
        time.sleep(0.1)


def analyze_frames_task(session_id: str, frame_ids: list[str], provider: str, model: str):
    with connect() as db:
        db.execute("UPDATE sessions SET llm_status='processing' WHERE id=?", (session_id,))
    for frame_id in frame_ids:
        try:
            run_analysis(session_id, frame_id, provider, model)
        except Exception as exc:
            log(f"Background Vision analysis failed: {exc}", "ERROR", session_id)
    with connect() as db:
        db.execute("UPDATE sessions SET llm_status='complete' WHERE id=?", (session_id,))


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "TrafficVision API", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/cameras")
def cameras():
    config = DATA_DIR / "cameras.json"
    result = json.loads(config.read_text(encoding="utf-8")) if config.exists() else DEFAULT_CAMERAS
    with connect() as db:
        for camera in result:
            latest = db.execute("SELECT f.analysis_json,f.yolo_json FROM frames f JOIN sessions s ON f.session_id=s.id WHERE s.camera_id=? AND (f.analysis_json IS NOT NULL OR f.yolo_json IS NOT NULL) ORDER BY COALESCE(f.analyzed_at,json_extract(f.yolo_json,'$.detected_at')) DESC LIMIT 1", (camera["id"],)).fetchone()
            if latest:
                analysis = json.loads(latest["analysis_json"]) if latest["analysis_json"] else {}
                yolo = json.loads(latest["yolo_json"]) if latest["yolo_json"] else {}
                camera["status"] = "congested" if yolo.get("label") == "Ùn tắc" else "normal"
    return result


@app.get("/api/media")
def list_media():
    """List videos available in the configured data folder for local review."""
    if not DATA_DIR.is_dir():
        return []
    return [{"name": p.name, "size_bytes": p.stat().st_size, "extension": p.suffix.lower()} for p in sorted(DATA_DIR.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


@app.post("/api/sessions", status_code=201)
async def upload_media(background_tasks: BackgroundTasks, file: UploadFile = File(...), camera_id: int | None = Form(default=None), sample_every_seconds: float | None = Form(default=None)):
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXTS | IMAGE_EXTS:
        raise HTTPException(415, "Chỉ hỗ trợ video MP4/AVI/MOV/MKV/WEBM và ảnh JPG/PNG/WEBP")
    settings = get_settings()
    every = 1.0 / max(1, min(30, settings.get("detector_fps", 30)))
    max_frames = settings["max_frames"]
    session_id = uuid.uuid4().hex[:12]
    path = UPLOAD_DIR / f"{session_id}{ext}"
    size = 0
    too_large = False
    with path.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > 1024 * 1024 * 1024:
                too_large = True
                break
            output.write(chunk)
    await file.close()
    if too_large:
        path.unlink(missing_ok=True)
        raise HTTPException(413, "File vượt quá giới hạn 1 GB")
    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "File tải lên rỗng")
    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        media_type = "image" if ext in IMAGE_EXTS else "video"
        db.execute("INSERT INTO sessions(id,camera_id,filename,media_type,media_path,created_at,processing_status) VALUES (?,?,?,?,?,?,?)", (session_id, camera_id, name, media_type, str(path), now, "queued"))
    if media_type == "video":
        background_tasks.add_task(process_video_session, session_id, path, max_frames)
        log(f"Video uploaded; background processing queued: {name}", "INFO", session_id)
        return public_session(session_id)
    try:
        metadata = await run_in_threadpool(inspect_media, path, session_id, every, max_frames)
        await run_in_threadpool(run_yolo_for_session, session_id)
    except Exception:
        path.unlink(missing_ok=True)
        with connect() as db:
            db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        raise
    with connect() as db:
        db.execute("UPDATE sessions SET media_type=?,duration=?,width=?,height=?,processing_status='complete' WHERE id=?", (metadata["media_type"], metadata["duration"], metadata["width"], metadata["height"], session_id))
    log(f"Media uploaded: {name} ({metadata['width']}×{metadata['height']})", "INFO", session_id)
    return public_session(session_id)


class LibraryMediaRequest(BaseModel):
    filename: str
    camera_id: int | None = None
    sample_every_seconds: float | None = Field(default=None, gt=0, le=3600)


@app.post("/api/sessions/from-library", status_code=201)
def create_session_from_library(request: LibraryMediaRequest, background_tasks: BackgroundTasks):
    # Only accept direct children returned by /api/media; never treat a client path as trusted.
    source = (DATA_DIR / Path(request.filename).name).resolve()
    if source.parent != DATA_DIR or not source.is_file() or source.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(404, "Không tìm thấy video hợp lệ trong thư mục data")
    session_id = uuid.uuid4().hex[:12]
    target = UPLOAD_DIR / f"{session_id}{source.suffix.lower()}"
    shutil.copyfile(source, target)
    settings = get_settings()
    every = 1.0 / max(1, min(30, settings.get("detector_fps", 30)))
    with connect() as db:
        db.execute("INSERT INTO sessions(id,camera_id,filename,media_type,media_path,created_at,processing_status) VALUES (?,?,?,?,?,?,?)", (session_id, request.camera_id, source.name, "video", str(target), datetime.now(timezone.utc).isoformat(), "queued"))
    background_tasks.add_task(process_video_session, session_id, target, settings["max_frames"])
    log(f"Video loaded; background processing queued: {source.name}", "INFO", session_id)
    return public_session(session_id)


@app.get("/api/sessions")
def list_sessions(limit: int = 50):
    limit = max(1, min(limit, 200))
    with connect() as db:
        rows = db.execute("SELECT s.*, COUNT(f.id) frame_count, SUM(CASE WHEN f.analysis_json IS NOT NULL THEN 1 ELSE 0 END) analyzed_count FROM sessions s LEFT JOIN frames f ON f.session_id=s.id GROUP BY s.id ORDER BY s.created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str):
    return public_session(session_id)


@app.get("/api/sessions/{session_id}/frames/{frame_id}/image")
def frame_image(session_id: str, frame_id: str):
    with connect() as db:
        row = db.execute("SELECT image_path FROM frames WHERE id=? AND session_id=?", (frame_id, session_id)).fetchone()
    if not row or not Path(row[0]).is_file():
        raise HTTPException(404, "Không tìm thấy ảnh frame")
    return FileResponse(row[0], media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/sessions/{session_id}/media")
def session_media(session_id: str):
    with connect() as db:
        row = db.execute("SELECT media_path,media_type FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not row or row["media_type"] != "video" or not Path(row["media_path"]).is_file():
        raise HTTPException(404, "Không tìm thấy video của phiên")
    return FileResponse(row["media_path"], headers={"Accept-Ranges": "bytes"})


@app.post("/api/sessions/{session_id}/analyze")
def analyze_session(session_id: str, background_tasks: BackgroundTasks, frame_id: str | None = Form(default=None), provider: str | None = Form(default=None), model: str | None = Form(default=None)):
    detail = public_session(session_id)
    settings = get_settings()
    provider = (provider or settings["provider"]).lower()
    model = model or settings.get({"ollama": "ollama_model", "gemini": "gemini_model", "gpt": "gpt_model"}.get(provider, "ollama_model"), "qwen2.5vl:7b")
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(503, "Thiếu GEMINI_API_KEY. Hãy cấu hình backend/.env rồi khởi động lại.")
    if provider == "gpt" and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "Thiếu OPENAI_API_KEY. Hãy cấu hình backend/.env rồi khởi động lại.")
    targets = [f for f in detail["frames"] if f["id"] == frame_id] if frame_id else detail["frames"]
    if frame_id and not targets:
        raise HTTPException(404, "Không tìm thấy frame")
    if not targets:
        raise HTTPException(400, "Phiên không có frame")
    with connect() as db:
        db.execute("UPDATE sessions SET llm_status='queued' WHERE id=?", (session_id,))
    background_tasks.add_task(analyze_frames_task, session_id, [f["id"] for f in targets], provider, model)
    return {"session_id": session_id, "provider": provider, "model": model, "queued_frames": len(targets), "session": public_session(session_id)}


@app.get("/api/history")
def history(limit: int = 50):
    sessions = list_sessions(limit)
    with connect() as db:
        for session in sessions:
            latest = db.execute("SELECT analysis_json,analyzed_at FROM frames WHERE session_id=? AND analysis_json IS NOT NULL ORDER BY analyzed_at DESC LIMIT 1", (session["id"],)).fetchone()
            session["latest_analysis"] = json.loads(latest["analysis_json"]) if latest else None
            session["analyzed_at"] = latest["analyzed_at"] if latest else None
    return sessions


@app.get("/api/logs")
def logs(limit: int = 100):
    limit = max(1, min(limit, 500))
    with connect() as db:
        rows = db.execute("SELECT id,created_at,level,message,session_id FROM logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/settings")
def read_settings():
    values = get_settings()
    values["api_keys"] = {"gemini_configured": bool(os.getenv("GEMINI_API_KEY")), "gpt_configured": bool(os.getenv("OPENAI_API_KEY"))}
    return values


@app.put("/api/settings")
def update_settings(update: SettingsUpdate):
    values = update.model_dump(exclude_none=True)
    if "provider" in values and values["provider"] not in {"ollama", "gemini", "gpt"}:
        raise HTTPException(400, "provider phải là ollama, gemini hoặc gpt")
    if "detector_model" in values and values["detector_model"] not in {"yolo11s.pt", "rtdetr-l.pt"}:
        raise HTTPException(400, "detector_model phải là yolo11s.pt hoặc rtdetr-l.pt")
    with connect() as db:
        for key, value in values.items():
            db.execute("INSERT INTO settings(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))
    log("Settings updated")
    return read_settings()


@app.get("/api/sessions/{session_id}/export.json")
def export_session(session_id: str):
    detail = public_session(session_id)
    return detail


@app.post("/api/alerts", status_code=201)
def send_alert(alert: AlertRequest):
    log(f"ALERT [{alert.channel}]: {alert.message}", "WARN", alert.session_id)
    return {"sent": True, "channel": alert.channel, "message": alert.message, "created_at": datetime.now(timezone.utc).isoformat(), "note": "Đã ghi cảnh báo vào nhật ký hệ thống"}


@app.get("/api/cameras/{camera_id}/capture")
def capture_camera(camera_id: int):
    cams = cameras()
    camera = next((c for c in cams if c["id"] == camera_id), None)
    if camera is None:
        raise HTTPException(404, "Không tìm thấy camera")
    source = camera.get("source")
    if source is None:
        raise HTTPException(503, "Camera chưa cấu hình nguồn. Đặt source là chỉ số webcam hoặc RTSP URL trong data/cameras.json")
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(503, "Không lấy được hình từ camera")
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(500, "Không m? hóa được ảnh camera")
    path = UPLOAD_DIR / f"capture-{camera_id}-{uuid.uuid4().hex[:6]}.jpg"
    path.write_bytes(encoded.tobytes())
    log(f"Captured still image from CAM_{camera_id:03d}")
    return FileResponse(path, media_type="image/jpeg")
