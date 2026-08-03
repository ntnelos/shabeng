"""
server.py — FastAPI web server & WebSocket progress stream for Shabeng
========================================================================

Runs locally on http://localhost:8000
Connects the local web UI (Deep Purple & Fuchsia design) to Feature A & Feature B.
"""

import asyncio
import os
import subprocess
import pathlib
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from shabeng.config import (
    DEFAULT_CLIPS_DIR,
    DEFAULT_AUDIO_DIR,
    DEFAULT_OUTPUT_DIR,
    ORIGINAL_AUDIO_GAIN_PERCENT,
    EXTERNAL_AUDIO_GAIN_PERCENT,
    GEMINI_API_KEY,
    GEMINI_MODEL
)
from shabeng.sync_engine import sync_and_export_clips
from shabeng.ingest import scan_clips, scan_audio
from shabeng.analyzer import analyze_all
from shabeng.director import select_clips, compute_audio_sync
from shabeng.fcpxml import generate_fcpxml
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("shabeng.server")

app = FastAPI(title="Shabeng Automated Wedding Reel & Audio Sync")

_ui_dir = pathlib.Path(__file__).parent / "shabeng" / "ui"
app.mount("/static", StaticFiles(directory=str(_ui_dir)), name="static")

# WebSocket Connection Manager for broadcasting live progress
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_progress(self, target: str, msg: str, pct: int):
        payload = {"type": "progress", "target": target, "msg": msg, "pct": pct}
        for connection in list(self.active_connections):
            try:
                await connection.send_json(payload)
            except Exception:
                pass

manager = ConnectionManager()

# Event loop helper for thread-safe websocket broadcasts
_loop: Optional[asyncio.AbstractEventLoop] = None

@app.on_event("startup")
async def startup_event():
    global _loop
    _loop = asyncio.get_running_loop()

def _send_ws_progress_threadsafe(target: str, msg: str, pct: int):
    if _loop and _loop.is_running():
        asyncio.run_coroutine_threadsafe(
            manager.broadcast_progress(target, msg, pct), _loop
        )

# API Request Models
class SyncRequest(BaseModel):
    clips_dir: str = DEFAULT_CLIPS_DIR
    audio_dir: str = DEFAULT_AUDIO_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    camera_vol_pct: float = ORIGINAL_AUDIO_GAIN_PERCENT
    mixer_vol_pct: float = EXTERNAL_AUDIO_GAIN_PERCENT

class AiRequest(BaseModel):
    clips_dir: str = DEFAULT_CLIPS_DIR
    audio_dir: str = DEFAULT_AUDIO_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    api_key: Optional[str] = None
    model: str = "gemini-1.5-flash"
    min_clips: int = 10
    max_clips: int = 15

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = _ui_dir / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

def _pick_folder_native(prompt_text: str) -> str:
    script = f'''
    tell application "System Events"
        activate
        set selectedFolder to choose folder with prompt "{prompt_text}"
        return POSIX path of selectedFolder
    end tell
    '''
    try:
        res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=60)
        if res.returncode == 0 and res.stdout.strip():
            path = res.stdout.strip()
            return path[:-1] if path.endswith("/") else path
    except Exception as exc:
        logger.warning("AppleScript folder pick failed: %s", exc)

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title=prompt_text)
        root.destroy()
        if folder:
            return folder
    except Exception as exc:
        logger.warning("Tkinter folder pick failed: %s", exc)

    return ""

def _pick_file_native(prompt_text: str, file_type: str = "video") -> str:
    if file_type == "video":
        type_str = 'of type {"public.movie", "com.apple.quicktime-movie", "public.mpeg-4"}'
    elif file_type == "audio":
        type_str = 'of type {"public.audio", "public.mp3", "com.apple.m4a-audio", "public.wave-audio"}'
    else:
        type_str = ""

    script = f'''
    tell application "System Events"
        activate
        set selectedFile to choose file with prompt "{prompt_text}" {type_str}
        return POSIX path of selectedFile
    end tell
    '''
    try:
        res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=60)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception as exc:
        logger.warning("AppleScript file pick failed: %s", exc)

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if file_type == "video":
            types = [("Video files", "*.mp4 *.mov *.m4v *.mxf"), ("All files", "*.*")]
        else:
            types = [("Audio files", "*.wav *.mp3 *.aac *.m4a *.aif *.aiff"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(title=prompt_text, filetypes=types)
        root.destroy()
        if file_path:
            return file_path
    except Exception as exc:
        logger.warning("Tkinter file pick failed: %s", exc)

    return ""

@app.post("/api/browse-folder")
async def browse_folder(data: dict):
    prompt = data.get("prompt", "בחר תיקייה")
    loop = asyncio.get_running_loop()
    chosen_path = await loop.run_in_executor(None, _pick_folder_native, prompt)
    return {"path": chosen_path}

@app.post("/api/browse-file")
async def browse_file(data: dict):
    prompt = data.get("prompt", "בחר קובץ")
    file_type = data.get("file_type", "video")
    loop = asyncio.get_running_loop()
    chosen_path = await loop.run_in_executor(None, _pick_file_native, prompt, file_type)
    return {"path": chosen_path}

@app.get("/api/media-file")
async def get_media_file(path: str):
    p = pathlib.Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(p))

@app.websocket("/ws/progress")
async def websocket_progress_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ── Feature A API Route ────────────────────────────────────────────────
@app.post("/api/sync")
async def api_run_sync(req: SyncRequest):
    logger.info("Starting Audio Sync Feature A...")

    def progress_cb(msg: str, pct: int, total: int):
        _send_ws_progress_threadsafe("sync", msg, pct)

    # Run in background executor to avoid blocking event loop
    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(
            None,
            sync_and_export_clips,
            req.clips_dir,
            req.audio_dir,
            req.output_dir,
            req.camera_vol_pct,
            req.mixer_vol_pct,
            0.25,
            progress_cb
        )
        return {"status": "success", "results": [r.to_dict() for r in results]}
    except Exception as exc:
        logger.exception("Audio sync failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

# ── Feature B API Route ────────────────────────────────────────────────
@app.post("/api/analyze")
async def api_run_ai(req: AiRequest):
    logger.info("Starting AI Director Feature B...")

    if req.api_key:
        os.environ["GEMINI_API_KEY"] = req.api_key
        genai.configure(api_key=req.api_key)

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="Gemini API Key is missing! Please provide it in the UI or in .env"
        )

    def progress_cb(msg: str, pct: int, total: int):
        _send_ws_progress_threadsafe("ai", msg, pct)

    loop = asyncio.get_running_loop()

    def _ai_task():
        progress_cb("Scanning video clips and audio files...", 5, 100)
        clips = scan_clips(req.clips_dir)
        audio_files = scan_audio(req.audio_dir)

        if not clips:
            raise ValueError(f"No video clips found in {req.clips_dir}")

        progress_cb("Uploading and analyzing clips with Gemini AI...", 20, 100)
        
        # Override default model if selected
        import shabeng.config as cfg
        cfg.GEMINI_MODEL = req.model

        analyses = analyze_all(clips)

        progress_cb("Ranking clips & computing director cut...", 80, 100)
        selected = select_clips(analyses, req.min_clips, req.max_clips)

        progress_cb("Computing audio alignment timeline...", 90, 100)
        sync_plan = compute_audio_sync(selected, audio_files)

        output_path = pathlib.Path(req.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fcpxml_file = output_path / "wedding_reel.fcpxml"

        progress_cb("Generating Final Cut Pro FCPXML timeline...", 95, 100)
        generate_fcpxml(selected, sync_plan, str(fcpxml_file))

        progress_cb("AI Reel Generation complete!", 100, 100)

        selected_dict = []
        for a in selected:
            selected_dict.append({
                "clip_filename": a.clip.filename,
                "category": a.category,
                "energy": a.energy_level,
                "emotion": a.emotion_level,
                "best_start": a.best_start_sec,
                "best_end": a.best_end_sec,
                "description": a.description
            })

        return {
            "status": "success",
            "fcpxml_path": str(fcpxml_file),
            "selected_clips": selected_dict
        }

    try:
        res = await loop.run_in_executor(None, _ai_task)
        return res
    except Exception as exc:
        logger.exception("AI Reel generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Shabeng Local Web UI running on http://localhost:8000")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
