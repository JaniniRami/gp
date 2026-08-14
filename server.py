#!/usr/bin/env python3
"""ProactMAD presentation demo -- serve the live-replay UI and optional ESP serial.

Usage (Mac):
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python server.py
  # optional hardware:
  python server.py --esp auto
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
PACK_PATH = ROOT / "data" / "pack.json"

_esp_lock = threading.Lock()
_esp = None  # type: ignore[assignment]


def _open_serial(port: str, baud: int = 115200):
    import serial  # pyserial

    ser = serial.Serial(port, baudrate=baud, timeout=0.2)
    return ser


def _autodetect_port() -> str | None:
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    ports = list(list_ports.comports())
    prefer = []
    for p in ports:
        blob = f"{p.device} {p.description} {p.hwid}".lower()
        if any(k in blob for k in ("usbserial", "usbmodem", "cp210", "ch340", "wch", "silicon", "esp32", "uart")):
            prefer.append(p.device)
    if prefer:
        return prefer[0]
    return ports[0].device if ports else None


def create_app(esp_port: str | None) -> FastAPI:
    app = FastAPI(title="ProactMAD live demo")
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/api/pack")
    def pack():
        if not PACK_PATH.is_file():
            raise HTTPException(500, "data/pack.json missing")
        return json.loads(PACK_PATH.read_text(encoding="utf-8"))

    @app.get("/api/esp/status")
    def esp_status():
        with _esp_lock:
            linked = _esp is not None and getattr(_esp, "is_open", False)
        return {"linked": bool(linked), "port": esp_port}

    @app.post("/api/esp/cmd")
    def esp_cmd(body: dict):
        cmd = str(body.get("cmd", "")).strip()
        if not cmd:
            raise HTTPException(400, "cmd required")
        with _esp_lock:
            if _esp is None or not getattr(_esp, "is_open", False):
                raise HTTPException(503, "ESP not connected")
            _esp.write((cmd + "\n").encode("ascii", errors="ignore"))
            _esp.flush()
        return {"ok": True, "cmd": cmd}

    global _esp
    if esp_port:
        port = _autodetect_port() if esp_port == "auto" else esp_port
        if not port:
            print("WARNING: no serial port found; UI will still run", flush=True)
        else:
            try:
                _esp = _open_serial(port)
                print(f"ESP serial open on {port}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: could not open {port}: {exc}", flush=True)
                _esp = None
    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--esp", default=None, help="serial device, or 'auto'")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not PACK_PATH.is_file():
        print("Missing data/pack.json", file=sys.stderr)
        return 1

    app = create_app(args.esp)
    url = f"http://{args.host}:{args.port}/"
    print(f"ProactMAD demo ? {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
