"""
Driving License Detection – WebSocket server (GPU-accelerated).

Runs on the GPU VM. Clients send JPEG frames via WebSocket,
server runs license detection (YOLO FP16 + optional Qwen OCR on CUDA)
and returns annotated frames.

Start:
    python main.py                          # defaults to 0.0.0.0:5001
    HOST=0.0.0.0 PORT=5001 python main.py   # explicit
"""

import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.websocket import router as ws_router

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5001"))
RELOAD = os.environ.get("RELOAD", "0").strip() in {"1", "true", "yes", "on"}
PRELOAD_QWEN = os.environ.get("PRELOAD_QWEN", "1").strip() in {"1", "true", "yes", "on"}
QWEN_WARMUP = os.environ.get("QWEN_WARMUP", "1").strip() in {"1", "true", "yes", "on"}


def _configure_gpu():
    """Set CUDA global options for maximum throughput."""
    if not torch.cuda.is_available():
        print("[main] WARNING: CUDA not available, running on CPU")
        return

    dev = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    cap = torch.cuda.get_device_capability(0)
    print(f"[main] GPU: {dev}  |  VRAM: {mem:.1f} GB  |  Compute: {cap[0]}.{cap[1]}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_gpu()

    from api.websocket import _get_yolo, DEFAULT_WEIGHTS
    _get_yolo(DEFAULT_WEIGHTS)

    if PRELOAD_QWEN:
        try:
            from utils.qwen_ocr import _load_qwen  # type: ignore
            _load_qwen()
            if QWEN_WARMUP:
                import numpy as np
                import cv2
                from utils.qwen_ocr import qwen_ocr_and_validate

                # Small warmup crop to force first-token latency and kernel init now.
                dummy = np.zeros((384, 384, 3), dtype=np.uint8)
                dummy[:] = (30, 30, 30)
                cv2.putText(dummy, "WARMUP", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
                t0 = time.perf_counter()
                _ = qwen_ocr_and_validate(dummy)
                dt = time.perf_counter() - t0
                print(f"[main] Qwen warmup done in {dt:.1f}s")
        except Exception as e:
            # Don't crash the service if Qwen can't preload; /validate can still run YOLO-only.
            print(f"[main] Qwen preload failed: {e}")

    yield


app = FastAPI(
    title="Driving License Detection",
    description="WebSocket stream: send JPEG frames, receive annotated frames with license detections",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)


@app.get("/")
def root():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {
        "service": "Driving License Detection",
        "device": gpu,
        "websocket": f"ws://<host>:{PORT}/stream",
        "protocol": "Send JPEG frame bytes (binary), receive annotated JPEG",
        "optional_query": "?weights=weights/best.pt&qwen=1",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    info = {"status": "ok"}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_used_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2))
        info["gpu_mem_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2))
    return info


if __name__ == "__main__":
    import uvicorn

    print(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        # Reload is great for local dev, but it can cause frequent restarts on
        # GPU boxes / network-forwarded setups, which looks like client timeouts
        # and "slow" streaming. Keep it opt-in.
        reload=RELOAD,
        reload_excludes=["client.py"] if RELOAD else None,
    )
