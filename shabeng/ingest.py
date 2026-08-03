"""
ingest.py — Scan a local directory for video clips & audio files
=================================================================
"""

import pathlib
import subprocess
import json
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

# Supported extensions (case-insensitive)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mxf"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".m4a", ".aif", ".aiff"}


@dataclass
class ClipInfo:
    """Metadata container for a single video clip on disk."""
    path: pathlib.Path
    filename: str
    duration_sec: float = 0.0
    fps: float = 30.0
    total_frames: int = 0
    width: int = 0
    height: int = 0


@dataclass
class AudioFileInfo:
    """Metadata container for an external audio file."""
    path: pathlib.Path
    filename: str
    duration_sec: float = 0.0
    sample_rate: int = 48000


def _probe_video(path: pathlib.Path) -> dict:
    """Use ffprobe to extract duration, fps, and resolution."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout)
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", path.name, exc)
        return {}


def _parse_fps(stream: dict) -> float:
    """Parse frame rate from ffprobe stream data."""
    # Try r_frame_rate first (e.g. "30000/1001")
    rfr = stream.get("r_frame_rate", "")
    if "/" in rfr:
        num, den = rfr.split("/")
        try:
            return round(int(num) / int(den), 3)
        except (ValueError, ZeroDivisionError):
            pass
    # Fallback to avg_frame_rate
    afr = stream.get("avg_frame_rate", "")
    if "/" in afr:
        num, den = afr.split("/")
        try:
            return round(int(num) / int(den), 3)
        except (ValueError, ZeroDivisionError):
            pass
    return 30.0


def _probe_audio(path: pathlib.Path) -> dict:
    """Use ffprobe for audio file metadata."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout)
    except Exception as exc:
        logger.warning("ffprobe failed for audio %s: %s", path.name, exc)
        return {}


def scan_clips(target: str) -> List[ClipInfo]:
    """
    Scan *target* (directory or single video file) and return a list of ClipInfo.
    Uses ffprobe for accurate metadata (duration, fps, resolution).
    """
    root = pathlib.Path(target)
    if not root.exists():
        raise FileNotFoundError(f"Clips path not found: {target}")

    if root.is_file():
        candidate_files = [root]
    else:
        candidate_files = sorted([f for f in root.rglob("*") if f.is_file()])

    clips: List[ClipInfo] = []
    for file in candidate_files:
        if file.suffix.lower() in VIDEO_EXTENSIONS:
            probe = _probe_video(file)

            duration = 0.0
            fps = 30.0
            width = 0
            height = 0

            # Extract from format
            fmt = probe.get("format", {})
            duration = float(fmt.get("duration", 0))

            # Extract from first video stream
            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "video":
                    fps = _parse_fps(stream)
                    width = int(stream.get("width", 0))
                    height = int(stream.get("height", 0))
                    if not duration:
                        duration = float(stream.get("duration", 0))
                    break

            total_frames = int(round(duration * fps))

            clip = ClipInfo(
                path=file,
                filename=file.name,
                duration_sec=round(duration, 3),
                fps=fps,
                total_frames=total_frames,
                width=width,
                height=height,
            )
            clips.append(clip)
            logger.info("Found clip: %s  (%.2fs, %.1f fps, %dx%d)",
                        file.name, duration, fps, width, height)

    logger.info("Ingested %d video clip(s) from %s", len(clips), target)
    return clips


def scan_audio(target: str) -> List[AudioFileInfo]:
    """
    Scan *target* (directory or single audio file) and return AudioFileInfo list.
    """
    root = pathlib.Path(target)
    if not root.exists():
        logger.warning("Audio path not found: %s — skipping audio sync.", target)
        return []

    if root.is_file():
        candidate_files = [root]
    else:
        candidate_files = sorted([f for f in root.rglob("*") if f.is_file()])

    audio_files: List[AudioFileInfo] = []
    for file in candidate_files:
        if file.suffix.lower() in AUDIO_EXTENSIONS:
            probe = _probe_audio(file)

            duration = 0.0
            sample_rate = 48000

            fmt = probe.get("format", {})
            duration = float(fmt.get("duration", 0))

            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "audio":
                    sample_rate = int(stream.get("sample_rate", 48000))
                    if not duration:
                        duration = float(stream.get("duration", 0))
                    break

            af = AudioFileInfo(
                path=file,
                filename=file.name,
                duration_sec=round(duration, 3),
                sample_rate=sample_rate,
            )
            audio_files.append(af)
            logger.info("Found audio: %s  (%.2fs, %d Hz)",
                        file.name, duration, sample_rate)

    logger.info("Ingested %d audio file(s) from %s", len(audio_files), target)
    return audio_files
