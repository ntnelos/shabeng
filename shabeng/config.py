"""
config.py — Central configuration & environment loading
========================================================
"""

import os
import pathlib
from dotenv import load_dotenv

# ── Load .env from project root ────────────────────────────────────
_project_root = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# ── Gemini ──────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
# Default to gemini-1.5-flash for ultra low cost & high speed video analysis
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ── Paths (defaults — overridable via CLI/Web UI) ──────────────────
DEFAULT_CLIPS_DIR: str = str(_project_root / "clips")
DEFAULT_AUDIO_DIR: str = str(_project_root / "audio")
DEFAULT_OUTPUT_DIR: str = str(_project_root / "output")

# ── Timeline ────────────────────────────────────────────────────────
TIMELINE_WIDTH: int = 1080
TIMELINE_HEIGHT: int = 1920
TIMELINE_FPS: int = 30          # 30 fps — common for reels
TIMELINE_NAME: str = "Wedding Reel"

# ── Director ────────────────────────────────────────────────────────
MIN_CLIPS: int = 10
MAX_CLIPS: int = 15
CLIP_MAX_DURATION_SEC: float = 6.0   # max seconds per clip segment
CLIP_MIN_DURATION_SEC: float = 1.5   # min seconds per clip segment

# ── Audio mix defaults ──────────────────────────────────────────────
ORIGINAL_AUDIO_GAIN_PERCENT: float = 30.0   # 30% volume
EXTERNAL_AUDIO_GAIN_PERCENT: float = 70.0   # 70% volume

ORIGINAL_AUDIO_GAIN_DB: float = -10.5   # ≈ 30 %
EXTERNAL_AUDIO_GAIN_DB: float = -3.0    # ≈ 70 %

# ── Audio Sync Engine ───────────────────────────────────────────────
SYNC_CORRELATION_THRESHOLD: float = 0.05   # Minimum 2D Mel-Spectrogram ZNCC peak score for valid sync

# ── Rate-limit guard for Gemini uploads ─────────────────────────────
GEMINI_UPLOAD_DELAY_SEC: float = 2.0    # seconds between uploads
GEMINI_MAX_RETRIES: int = 3
