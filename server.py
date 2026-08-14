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
import time
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
PACK_PATH = ROOT / "data" / "pack.json"
CLIP_DIR = ROOT / "data" / "clips"
CLIP_INDEX = CLIP_DIR / "index.json"

_esp_lock = threading.Lock()
_esp = None  # type: ignore[assignment]

# Link telemetry so the UI can show that the board actually answers.
_esp_rx: dict = {"last": None, "at": 0.0, "count": 0, "tx": None, "tx_count": 0}


def _open_serial(port: str, baud: int = 115200):
    import serial  # pyserial

    ser = serial.Serial(port, baudrate=baud, timeout=0.2)
    return ser


def _reader_loop() -> None:
    """Drain ESP replies ("READY ProactMAD", "OK ADVANCE", ...) into _esp_rx."""
    while True:
        with _esp_lock:
            ser = _esp
        if ser is None or not getattr(ser, "is_open", False):
            return
        try:
            raw = ser.readline()
        except Exception:  # noqa: BLE001
            return
        line = raw.decode("ascii", errors="replace").strip()
        if line:
            _esp_rx["last"] = line
            _esp_rx["at"] = time.time()
            _esp_rx["count"] += 1


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
    # also served raw so the UI keeps working behind a plain file server
    app.mount("/data", StaticFiles(directory=ROOT / "data"), name="data")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/api/clips")
    def clips():
        """Example library: one entry per presentation story (see _build_clips.py)."""
        if not CLIP_INDEX.is_file():
            return {"default": None, "clips": []}
        return json.loads(CLIP_INDEX.read_text(encoding="utf-8"))

    @app.get("/api/pack")
    def pack(clip: str | None = None):
        path = PACK_PATH
        if clip:
            if not clip.replace("_", "").isalnum():
                raise HTTPException(400, "bad clip id")
            candidate = CLIP_DIR / f"{clip}.json"
            if not candidate.is_file():
                raise HTTPException(404, f"clip {clip} not found")
            path = candidate
        if not path.is_file():
            raise HTTPException(500, "data/pack.json missing")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/esp/status")
    def esp_status():
        with _esp_lock:
            linked = _esp is not None and getattr(_esp, "is_open", False)
            port_name = getattr(_esp, "port", None) if _esp is not None else None
        last_at = _esp_rx["at"]
        return {
            "linked": bool(linked),
            "port": port_name or esp_port,
            "last_rx": _esp_rx["last"],
            "last_rx_age_sec": round(time.time() - last_at, 1) if last_at else None,
            "rx_count": _esp_rx["count"],
            "last_tx": _esp_rx["tx"],
            "tx_count": _esp_rx["tx_count"],
        }

    def _write(cmd: str) -> None:
        with _esp_lock:
            if _esp is None or not getattr(_esp, "is_open", False):
                raise HTTPException(503, "ESP not connected")
            _esp.write((cmd + "\n").encode("ascii", errors="ignore"))
            _esp.flush()
        _esp_rx["tx"] = cmd
        _esp_rx["tx_count"] += 1

    @app.post("/api/esp/cmd")
    def esp_cmd(body: dict):
        cmd = str(body.get("cmd", "")).strip()
        if not cmd:
            raise HTTPException(400, "cmd required")
        _write(cmd)
        return {"ok": True, "cmd": cmd}

    @app.post("/api/esp/ping")
    def esp_ping():
        """Ask the firmware for help text and wait briefly for its reply."""
        before = _esp_rx["count"]
        _write("?")
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if _esp_rx["count"] > before:
                return {"ok": True, "reply": _esp_rx["last"], "rx_count": _esp_rx["count"]}
            time.sleep(0.05)
        return {"ok": False, "reply": None, "rx_count": _esp_rx["count"]}

    global _esp
    if esp_port:
        port = _autodetect_port() if esp_port == "auto" else esp_port
        if not port:
            print("WARNING: no serial port found; UI will still run", flush=True)
        else:
            try:
                _esp = _open_serial(port)
                threading.Thread(target=_reader_loop, daemon=True).start()
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

    if CLIP_INDEX.is_file():
        clips = json.loads(CLIP_INDEX.read_text(encoding="utf-8")).get("clips", [])
        print(f"clip library: {len(clips)} examples", flush=True)
        for c in clips:
            print(f"  - {c['id']:15s} MESA {c.get('subject_id')}  {c.get('title')}", flush=True)
    else:
        print("clip library: none (data/clips/index.json missing)", file=sys.stderr)

    app = create_app(args.esp)
    url = f"http://{args.host}:{args.port}/"
    print(f"ProactMAD demo -> {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
