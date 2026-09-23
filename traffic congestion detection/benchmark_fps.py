"""Benchmark the complete YOLO tracking and traffic-diagnosis pipeline."""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = next(iter(sorted((ROOT / "data").glob("*.mp4"))), None)
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}
ROI_POINTS = [(0.2, 0.15), (0.3, 0.15), (1.0, 0.75), (1.0, 0.98), (0.01, 0.98)]
MOVING_SPEED_THRESHOLD = 12.0
MIN_TRACK_POINTS = 3
TEMPORAL_WINDOW_S = 3.0
MIN_STATE_SAMPLES = 3
CROWDED_COUNT = 8
JAM_COUNT = 12
CROWDED_SPEED = 30.0
JAM_SPEED = 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark full tracking and traffic diagnosis FPS.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Video source.")
    parser.add_argument("--model", type=Path, default=ROOT / "yolo11s.pt", help="YOLO model path.")
    parser.add_argument("--backend", choices=("openvino", "pytorch"), default="openvino", help="Inference backend.")
    parser.add_argument("--device", default=None, help="OpenVINO device or PyTorch device; backend default is used when omitted.")
    parser.add_argument("--imgsz", type=int, default=576, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--warmup", type=int, default=20, help="Warm-up frames excluded from timing.")
    parser.add_argument("--max-frames", type=int, default=300, help="Timed frames; use 0 for the whole video.")
    parser.add_argument("--sample-fps", type=float, default=0.0, help="Process every N FPS; 0 means every frame.")
    parser.add_argument("--compare-cpu", action="store_true",
                        help="Compare PyTorch CPU with OpenVINO CPU for the same .pt model.")
    return parser.parse_args()


def synchronize(device: str) -> None:
    if not device.startswith("intel") and device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def roi_polygon(width: int, height: int) -> np.ndarray:
    return np.array([(round(x * width), round(y * height)) for x, y in ROI_POINTS], dtype=np.int32)


def inside_roi(point, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0


def prune_window(history, now_s: float) -> None:
    while history and now_s - history[0][0] > TEMPORAL_WINDOW_S:
        history.popleft()


def track_speed(history):
    if len(history) < MIN_TRACK_POINTS:
        return None
    elapsed = history[-1][0] - history[0][0]
    if elapsed <= 0:
        return None
    (_, x0, y0), (_, x1, y1) = history[0], history[-1]
    return math.hypot(x1 - x0, y1 - y0) / elapsed


def stable_motion_state(history, now_s: float):
    prune_window(history, now_s)
    if len(history) < MIN_STATE_SAMPLES:
        return None
    return sum(is_moving for _, is_moving in history) / len(history) >= 0.5


def traffic_status(vehicle_count: int, moving_ratio: float, mean_speed: float) -> str:
    if vehicle_count >= JAM_COUNT and moving_ratio <= 0.25 and mean_speed < JAM_SPEED:
        return "UN TAC"
    if vehicle_count >= CROWDED_COUNT or (vehicle_count > 0 and moving_ratio <= 0.55 and mean_speed < CROWDED_SPEED):
        return "DONG DUC"
    return "THONG THOANG"


def draw_label(frame, text: str, origin, color) -> None:
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    x, y = origin
    cv2.rectangle(frame, (x, y - height - baseline - 6), (x + width + 8, y + 4), color, -1)
    cv2.putText(frame, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)


def load_model(model_path: Path, backend: str, imgsz: int):
    if backend == "pytorch":
        return YOLO(str(model_path)), model_path.name
    if model_path.is_dir():
        return YOLO(str(model_path)), model_path.name
    exported_path = model_path.with_name(f"{model_path.stem}_openvino_model")
    if not exported_path.exists():
        print(f"Dang export {model_path.name} sang OpenVINO...")
        exported_path = Path(YOLO(str(model_path)).export(format="openvino", imgsz=imgsz, device="cpu"))
    return YOLO(str(exported_path)), exported_path.name


def process_frame(model, frame, timestamp_s, polygon, position_histories, motion_histories, status_history, device, imgsz, conf):
    result = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=list(VEHICLE_CLASS_IDS),
                         conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    boxes = []
    speeds = []
    moving = stopped = unknown = 0
    if result.boxes is not None and result.boxes.id is not None:
        coordinates = result.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = result.boxes.id.int().cpu().tolist()
        for box, track_id in zip(coordinates, track_ids):
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if not inside_roi(center, polygon):
                continue
            history = position_histories[track_id]
            history.append((timestamp_s, *center))
            prune_window(history, timestamp_s)
            speed = track_speed(history)
            if speed is not None:
                speeds.append(speed)
                motion_histories[track_id].append((timestamp_s, speed >= MOVING_SPEED_THRESHOLD))
            state = stable_motion_state(motion_histories[track_id], timestamp_s)
            moving += state is True
            stopped += state is False
            unknown += state is None
            boxes.append((box, track_id, speed, state))

    classified = moving + stopped
    moving_ratio = moving / classified if classified else 0.0
    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    raw_label = traffic_status(len(boxes), moving_ratio, mean_speed)
    status_history.append((timestamp_s, raw_label))
    prune_window(status_history, timestamp_s)
    labels = [label for _, label in status_history]
    status = max(set(labels), key=lambda label: (labels.count(label), max(i for i, value in enumerate(labels) if value == label)))

    annotated = frame.copy()
    cv2.fillPoly(annotated, [polygon], (255, 255, 0))
    cv2.addWeighted(annotated, 0.12, frame, 0.88, 0, annotated)
    cv2.polylines(annotated, [polygon], True, (255, 255, 0), 2)
    for box, track_id, speed, state in boxes:
        x1, y1, x2, y2 = box
        color = (0, 220, 0) if state is True else ((0, 0, 255) if state is False else (180, 180, 180))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        speed_text = "warming up" if speed is None else f"{speed:.1f} px/s"
        draw_label(annotated, f"ID {track_id} | {speed_text}", (x1, max(25, y1)), color)
    draw_label(annotated, f"{status} | ROI xe: {len(boxes)} | chay: {moving_ratio:.0%}", (15, 45), (0, 165, 255))
    return annotated, len(boxes), status, moving, stopped, unknown


def benchmark(args: argparse.Namespace) -> dict:
    if args.source is None or not args.source.exists():
        raise FileNotFoundError("Khong tim thay video. Hay dung --source duong/dan/video.mp4")
    if not args.model.exists():
        raise FileNotFoundError(f"Khong tim thay model: {args.model}")

    if args.backend == "openvino" and not args.model.is_dir() and not args.model.exists():
        raise FileNotFoundError(f"Khong tim thay model: {args.model}")
    device = args.device or ("intel:cpu" if args.backend == "openvino" else ("0" if torch.cuda.is_available() else "cpu"))
    model, loaded_model_name = load_model(args.model, args.backend, args.imgsz)
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc source: {args.source}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    polygon = roi_polygon(width, height)
    frame_stride = max(1, round(source_fps / args.sample_fps)) if args.sample_fps > 0 else 1
    ok, warmup_frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Source khong co frame doc duoc.")

    def new_state():
        return defaultdict(deque), defaultdict(deque), deque()

    position_histories, motion_histories, status_history = new_state()
    for _ in range(max(0, args.warmup)):
        process_frame(model, warmup_frame, 0.0, polygon, position_histories, motion_histories, status_history, device, args.imgsz, args.conf)
    synchronize(device)
    position_histories, motion_histories, status_history = new_state()

    cap = cv2.VideoCapture(str(args.source))
    measured_count = 0
    decoded_count = 0
    start = time.perf_counter()
    with torch.inference_mode():
        while args.max_frames == 0 or decoded_count < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if decoded_count % frame_stride == 0:
                process_frame(model, frame, decoded_count / source_fps, polygon, position_histories, motion_histories, status_history, device, args.imgsz, args.conf)
                measured_count += 1
            decoded_count += 1
    cap.release()
    synchronize(device)
    elapsed = time.perf_counter() - start
    fps = measured_count / elapsed if elapsed else 0.0

    print("\n=== Full tracking + diagnosis FPS benchmark ===")
    print(f"Source:          {args.source}")
    print(f"Backend:         {args.backend}")
    print(f"Model:           {loaded_model_name}")
    print(f"Device:          {device}")
    print(f"Frame size:      {width}x{height}")
    print(f"YOLO imgsz:      {args.imgsz}")
    print(f"Measured frames: {measured_count} / {total_frames or 'unknown'}")
    print(f"Video FPS:       {source_fps:.2f}")
    print(f"Pipeline FPS:    {fps:.2f} (track + bounding box + diagnosis + drawing)")
    print(f"Realtime factor: {fps / source_fps:.2f}x")
    return {
        "backend": args.backend,
        "model": loaded_model_name,
        "fps": fps,
        "elapsed": elapsed,
        "device": device,
        "frames": measured_count,
    }


def compare_cpu(args: argparse.Namespace) -> None:
    if args.model.is_dir():
        raise ValueError("--compare-cpu cần đường dẫn model .pt để tạo bản OpenVINO tương ứng.")
    compare_args = argparse.Namespace(**vars(args))
    compare_args.device = "cpu"

    results = []
    for backend in ("pytorch", "openvino"):
        compare_args.backend = backend
        if backend == "openvino":
            compare_args.device = "intel:cpu"
        results.append(benchmark(compare_args))

    pytorch_result, openvino_result = results
    print("\n=== CPU comparison ===")
    print(f"PyTorch CPU:  {pytorch_result['fps']:.2f} FPS")
    print(f"OpenVINO CPU: {openvino_result['fps']:.2f} FPS")
    print(f"OpenVINO speed-up: {openvino_result['fps'] / pytorch_result['fps']:.2f}x")


if __name__ == "__main__":
    args = parse_args()
    if args.compare_cpu:
        compare_cpu(args)
    else:
        benchmark(args)