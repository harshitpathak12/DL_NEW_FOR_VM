"""
Driving License Detection – WebSocket server.

Same pattern as driver_safety_system: client sends JPEG frames via WebSocket,
server runs license detection and returns annotated frames.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.websocket import router as ws_router

app = FastAPI(
    title="Driving License Detection",
    description="WebSocket stream: send JPEG frames, receive annotated frames with license detections",
    version="1.0.0",
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
    return {
        "service": "Driving License Detection",
        "websocket": "ws://<host>:5001/stream",
        "protocol": "Send JPEG frame bytes (binary), receive annotated JPEG",
        "optional_query": "?weights=weights/best.pt&ocr=1&validate=1&vit_weights=weights/vit_validator.pt",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)
