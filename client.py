#!/usr/bin/env python3
"""
Driving License Validation Client — press 'e' to capture & validate.

Flow:
  1. Checks server connectivity on startup (fails fast if unreachable).
  2. Live webcam/video preview (local, no server roundtrip).
  3. Hold 'e' → capture mode (green border, collecting frames).
  4. Release 'e' → best frame POSTed to server for full pipeline.
  5. Result displayed on frame + printed in terminal.
  6. Press 'x' to quit.

Env vars:
  SERVER_URL          HTTP base URL          (default http://127.0.0.1:5001)
  VALIDATE_TIMEOUT_S  HTTP timeout (s)       (default 120)
  FPS                 Preview rate cap       (default 30)
  WIDTH / HEIGHT      Webcam resolution      (default 640x480)
  JPEG_QUALITY        Encode quality 1-100   (default 80)
  NO_GUI              Force headless mode    (1/true/yes)
  VIDEO_SOURCE        Same as CLI arg        (file path, URL, or camera index)

Usage:
  # On the same machine as the server
  python client.py

  # From a laptop via SSH tunnel
  ssh -L 5001:localhost:5001 user@gpu-server   # in one terminal
  SERVER_URL=http://127.0.0.1:5001 python client.py   # in another
"""

import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:5001")
VALIDATE_URL = f"{SERVER_URL.rstrip('/')}/validate"
HEALTH_URL = f"{SERVER_URL.rstrip('/')}/health"


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(float(v))
    except Exception:
        return default


JPEG_QUALITY = _env_int("JPEG_QUALITY", 80)
CAP_WIDTH = _env_int("WIDTH", 640)
CAP_HEIGHT = _env_int("HEIGHT", 480)
TARGET_FPS = _env_int("FPS", 30)
VALIDATE_TIMEOUT_S = _env_int("VALIDATE_TIMEOUT_S", 120)
WINDOW_NAME = "DL Validator — press 'e' to capture, 'x' to quit"

NO_GUI = os.environ.get("NO_GUI", "0").strip().lower() in {"1", "true", "yes", "on"}
_has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
GUI_ENABLED = (not NO_GUI) and _has_display

_encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]


# ---------------------------------------------------------------------------
# Server connectivity check
# ---------------------------------------------------------------------------
def _check_server() -> bool:
    """Hit GET /health with a short timeout. Returns True if reachable."""
    print(f"\n[startup] Checking server connectivity: {HEALTH_URL}")
    req = urllib.request.Request(HEALTH_URL, method="GET")
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = json.loads(resp.read().decode())
            dt = time.perf_counter() - t0
            gpu = body.get("gpu", "unknown")
            print(f"[startup] Server OK ({dt*1000:.0f}ms) — GPU: {gpu}")
            return True
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        print(f"[startup] CANNOT REACH SERVER: {reason}")
    except Exception as e:
        print(f"[startup] CANNOT REACH SERVER: {type(e).__name__}: {e}")

    print()
    print("=" * 60)
    print("  SERVER IS NOT REACHABLE!")
    print(f"  Tried: {HEALTH_URL}")
    print()
    print("  If the server is on a remote machine, set up an SSH tunnel:")
    print()
    print("    ssh -L 5001:localhost:5001 user@gpu-server")
    print()
    print("  Then run this client:")
    print()
    print("    SERVER_URL=http://127.0.0.1:5001 python client.py")
    print()
    print("  Or if the server IP is directly reachable:")
    print()
    print("    SERVER_URL=http://<server-ip>:5001 python client.py")
    print("=" * 60)
    return False


# ---------------------------------------------------------------------------
# Capture source
# ---------------------------------------------------------------------------
def _open_capture(source):
    if source is None or isinstance(source, int):
        idx = source if isinstance(source, int) else 0
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        return cap
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ---------------------------------------------------------------------------
# Frame quality (Laplacian variance = sharpness)
# ---------------------------------------------------------------------------
def _sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ---------------------------------------------------------------------------
# Server communication
# ---------------------------------------------------------------------------
def _post_validate(frame: np.ndarray) -> dict:
    """Send a JPEG frame to POST /validate and return the JSON response."""
    ok, buf = cv2.imencode(".jpg", frame, _encode_params)
    if not ok:
        return {"verdict": "error", "message": "JPEG encode failed"}

    jpeg_bytes = buf.tobytes()
    print(f"[validate] Sending {len(jpeg_bytes)} bytes to {VALIDATE_URL} (timeout={VALIDATE_TIMEOUT_S}s)")

    req = urllib.request.Request(
        VALIDATE_URL,
        data=jpeg_bytes,
        headers={"Content-Type": "image/jpeg"},
        method="POST",
    )
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=VALIDATE_TIMEOUT_S) as resp:
            raw = resp.read().decode()
            dt = time.perf_counter() - t0
            result = json.loads(raw)
            print(f"[validate] Response received in {dt:.1f}s")
            return result
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"verdict": "error", "message": f"HTTP {e.code}: {body[:300]}"}
    except urllib.error.URLError as e:
        return {"verdict": "error", "message": f"Connection failed: {e.reason}"}
    except TimeoutError:
        return {"verdict": "error", "message": f"Timed out after {VALIDATE_TIMEOUT_S}s — server unreachable?"}
    except Exception as e:
        return {"verdict": "error", "message": f"{type(e).__name__}: {e}"}


def _print_result(result: dict):
    """Print full result to terminal."""
    verdict = result.get("verdict", "?")
    message = result.get("message", "")
    dl_nums = result.get("dl_numbers", [])
    ocr_text = result.get("ocr_text", "") or ""
    timings = result.get("timings", {})
    det_conf = result.get("detection_confidence", 0)

    print()
    print("=" * 60)
    if verdict == "valid":
        print("  >>> VALID DRIVING LICENCE <<<")
    elif verdict == "invalid":
        print("  >>> NOT A VALID LICENCE <<<")
    elif verdict == "error":
        print(f"  >>> ERROR: {message} <<<")
    else:
        print(f"  >>> {verdict.upper()}: {message} <<<")

    if verdict not in ("error",):
        print(f"  Message     : {message}")
        print(f"  Confidence  : {det_conf:.2f}")
        if dl_nums:
            print(f"  DL Number   : {', '.join(dl_nums)}")
        if ocr_text:
            print(f"  OCR Text    : {ocr_text[:200]}")
        if result.get("qwen_verdict"):
            print(f"  Qwen Verdict: {result['qwen_verdict']} (conf={result.get('qwen_confidence', 0):.2f})")
            print(f"  Qwen Reason : {result.get('qwen_reason', '')}")
        if result.get("rule_reason"):
            print(f"  Rule Reason : {result['rule_reason']}")
        if timings:
            print(f"  Timings     : YOLO {timings.get('yolo_ms', 0)}ms | OCR {timings.get('ocr_ms', 0)}ms | Total {timings.get('total_ms', 0)}ms")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------
def _draw_capture_border(frame: np.ndarray, n_frames: int):
    h, w = frame.shape[:2]
    t = 4
    cv2.rectangle(frame, (t, t), (w - t, h - t), (0, 255, 0), t)
    cv2.putText(
        frame, f"CAPTURING... ({n_frames} frames)",
        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
    )


def _draw_processing(frame: np.ndarray):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(
        frame, "PROCESSING... (YOLO -> OCR -> Validation)",
        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA,
    )


def _draw_result(frame: np.ndarray, result: dict):
    h, w = frame.shape[:2]
    verdict = result.get("verdict", "unknown")
    message = result.get("message", "")
    dl_nums = result.get("dl_numbers", [])
    timings = result.get("timings", {})
    det_conf = result.get("detection_confidence", 0)

    if verdict == "valid":
        color = (0, 200, 0)
        banner = "VALID DRIVING LICENCE"
    elif verdict == "invalid":
        color = (0, 0, 230)
        banner = "NOT A VALID LICENCE"
    elif verdict == "error":
        color = (0, 0, 255)
        banner = f"ERROR: {message[:40]}"
    else:
        color = (0, 165, 255)
        banner = "COULD NOT DETERMINE"

    cv2.rectangle(frame, (0, 0), (w, 120), (30, 30, 30), -1)
    cv2.putText(
        frame, banner, (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA,
    )
    detail = message
    if dl_nums:
        detail += f"  |  DL: {dl_nums[0]}"
    cv2.putText(
        frame, detail[:80], (15, 72),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
    )
    if timings:
        t_str = f"YOLO {timings.get('yolo_ms',0)}ms  OCR {timings.get('ocr_ms',0)}ms  Total {timings.get('total_ms',0)}ms  Conf {det_conf:.2f}"
        cv2.putText(
            frame, t_str, (15, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
        )

    bbox = result.get("bbox")
    if bbox:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    cv2.putText(
        frame, "Press 'e' to capture again  |  'x' to quit", (15, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA,
    )


def _draw_idle_hint(frame: np.ndarray):
    h, w = frame.shape[:2]
    cv2.putText(
        frame, "Press 'e' to capture & validate  |  'x' to quit",
        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (200, 200, 200), 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Background validation thread
# ---------------------------------------------------------------------------
_validate_lock = threading.Lock()
_validate_result: dict | None = None
_validate_busy = False
_validate_frame: np.ndarray | None = None


def _validate_worker(frame: np.ndarray):
    global _validate_result, _validate_busy, _validate_frame
    t0 = time.perf_counter()
    result = _post_validate(frame)
    dt = time.perf_counter() - t0

    _print_result(result)

    with _validate_lock:
        _validate_result = result
        _validate_frame = frame.copy()
        _validate_busy = False


def _start_validation(frame: np.ndarray):
    global _validate_busy, _validate_result
    with _validate_lock:
        _validate_busy = True
        _validate_result = None
    t = threading.Thread(target=_validate_worker, args=(frame,), daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Main — GUI mode
# ---------------------------------------------------------------------------
def _run_gui(cap):
    state = "idle"
    captured_frames: list[np.ndarray] = []
    result_display_until = 0.0
    interval = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0

    while True:
        t0 = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            print("[client] Capture ended.")
            break

        display = frame.copy()
        key = cv2.waitKey(1) & 0xFF

        if state == "idle":
            _draw_idle_hint(display)
            if key == ord("e"):
                state = "capturing"
                captured_frames = [frame.copy()]
                print("[client] Capture started — hold 'e'")

        elif state == "capturing":
            captured_frames.append(frame.copy())
            _draw_capture_border(display, len(captured_frames))
            if key != ord("e"):
                state = "processing"
                print(f"[client] Captured {len(captured_frames)} frames, picking sharpest...")
                best = max(captured_frames, key=_sharpness)
                _start_validation(best)

        elif state == "processing":
            _draw_processing(display)
            with _validate_lock:
                if not _validate_busy and _validate_result is not None:
                    state = "result"
                    result_display_until = time.monotonic() + 15.0

        elif state == "result":
            with _validate_lock:
                res = _validate_result
                res_frame = _validate_frame
            if res and res_frame is not None:
                display = res_frame.copy()
                _draw_result(display, res)
            if time.monotonic() > result_display_until or key == ord("e"):
                state = "idle" if key != ord("e") else "capturing"
                if state == "capturing":
                    captured_frames = [frame.copy()]
                    print("[client] Capture started — hold 'e'")

        if key == ord("x"):
            print("Exit requested")
            break

        cv2.imshow(WINDOW_NAME, display)

        elapsed = time.monotonic() - t0
        remaining = interval - elapsed
        if remaining > 0.001:
            time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main — headless mode (e.g. running on the VM itself)
# ---------------------------------------------------------------------------
def _run_headless(cap):
    import select

    print("[headless] Type 'e' + Enter to capture & validate.")
    print("[headless] Type 'x' + Enter to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[headless] Capture ended.")
            break

        if sys.stdin in select.select([sys.stdin], [], [], 0.03)[0]:
            line = sys.stdin.readline().strip().lower()
            if line == "x":
                break
            if line == "e":
                result = _post_validate(frame)
                _print_result(result)

    cap.release()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    source_arg = sys.argv[1] if len(sys.argv) > 1 else None
    env_source = os.environ.get("VIDEO_SOURCE")
    raw_source = source_arg or env_source
    source = None
    if raw_source is not None:
        s = str(raw_source).strip()
        source = int(s) if s.isdigit() else s

    print("=" * 60)
    print("  Driving License Validator")
    print(f"  Server  : {SERVER_URL}")
    print(f"  Validate: {VALIDATE_URL}")
    print(f"  Timeout : {VALIDATE_TIMEOUT_S}s")
    if source is None:
        print(f"  Source   : webcam(0) {CAP_WIDTH}x{CAP_HEIGHT} @ {TARGET_FPS} FPS")
    else:
        print(f"  Source   : {source!r}")
    print(f"  Display  : {'GUI' if GUI_ENABLED else 'headless'}")
    print(f"  Controls : hold 'e' = capture, release = validate, 'x' = quit")
    print("=" * 60)

    # --- Connectivity check FIRST ---
    if not _check_server():
        return

    cap = _open_capture(source)
    if not cap.isOpened():
        print("Error: cannot open capture source")
        return

    try:
        if GUI_ENABLED:
            _run_gui(cap)
        else:
            _run_headless(cap)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if cap.isOpened():
            cap.release()
        print("Done.")


if __name__ == "__main__":
    main()
