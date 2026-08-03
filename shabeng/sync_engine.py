"""
sync_engine.py — High-precision local audio synchronization & lossless video export
===================================================================================

Feature A implementation:
- Extracts audio envelopes using FFmpeg.
- Computes FFT cross-correlation to find exact timestamp alignment against master mixer audio.
- If synced: Muxes video losslessly (-c:v copy) + blends camera & mixer audio levels.
- Orders clips chronologically based on mixer timeline position (01_..., 02_...).
- If sync fails: Moves clip to 'unsynchronized/' output folder.
"""

import os
import shutil
import tempfile
import pathlib
import subprocess
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
from scipy import signal

from shabeng.ingest import scan_clips, scan_audio, ClipInfo, AudioFileInfo
from shabeng.config import SYNC_CORRELATION_THRESHOLD

logger = logging.getLogger(__name__)

# Sample rate for fast cross-correlation audio extraction
SAMPLE_RATE = 11025


@dataclass
class SyncResult:
    clip_filename: str
    original_path: str
    synced: bool
    offset_sec: float = 0.0
    correlation_score: float = 0.0
    output_path: Optional[str] = None
    chronological_index: Optional[int] = None
    error_msg: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_pcm_audio(input_path: pathlib.Path, output_wav: pathlib.Path, sample_rate: int = SAMPLE_RATE) -> bool:
    """Extract downsampled mono 16-bit PCM WAV using FFmpeg."""
    cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav",
        str(output_wav)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return res.returncode == 0 and output_wav.exists() and output_wav.stat().st_size > 0
    except Exception as exc:
        logger.warning("Failed to extract PCM audio for %s: %s", input_path.name, exc)
        return False


def _load_wav_data(wav_path: pathlib.Path) -> np.ndarray:
    """Load raw PCM WAV bytes into numpy float array."""
    try:
        import wave
        with wave.open(str(wav_path), "rb") as wf:
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio /= max_val
            return audio
    except Exception as exc:
        logger.warning("Error reading WAV data from %s: %s", wav_path.name, exc)
        return np.array([], dtype=np.float32)


def _find_audio_offset(clip_audio: np.ndarray, master_audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Tuple[float, float]:
    """
    Find best alignment offset of clip_audio within master_audio using normalized cross-correlation.
    Returns (best_offset_sec, peak_correlation_score).
    """
    if len(clip_audio) < sample_rate * 0.5 or len(master_audio) < len(clip_audio):
        return 0.0, 0.0

    # Compute energy envelope using Hilbert transform / magnitude moving average to speed up & improve robustness
    win_len = int(sample_rate * 0.02)  # 20ms window
    if win_len > 1:
        clip_env = np.convolve(np.abs(clip_audio), np.ones(win_len)/win_len, mode='same')
        master_env = np.convolve(np.abs(master_audio), np.ones(win_len)/win_len, mode='same')
    else:
        clip_env = np.abs(clip_audio)
        master_env = np.abs(master_audio)

    # Downsample envelope for faster FFT correlation
    ds_factor = 4
    clip_env_ds = clip_env[::ds_factor]
    master_env_ds = master_env[::ds_factor]
    ds_sr = sample_rate / ds_factor

    # Remove DC mean
    clip_env_ds -= np.mean(clip_env_ds)
    master_env_ds -= np.mean(master_env_ds)

    # FFT cross correlation
    corr = signal.fftconvolve(master_env_ds, clip_env_ds[::-1], mode='valid')

    # Normalize correlation peak
    clip_norm = np.linalg.norm(clip_env_ds)
    if clip_norm == 0:
        return 0.0, 0.0

    best_idx = int(np.argmax(corr))
    peak_val = float(corr[best_idx])

    # Local norm estimation
    window_norm = np.linalg.norm(master_env_ds[best_idx:best_idx + len(clip_env_ds)])
    norm_score = peak_val / (clip_norm * window_norm + 1e-7)

    offset_sec = float(best_idx / ds_sr)
    return round(offset_sec, 3), round(float(norm_score), 3)


def sync_and_export_clips(
    clips_dir: str,
    audio_dir: str,
    output_dir: str,
    camera_vol_pct: float = 30.0,
    mixer_vol_pct: float = 70.0,
    correlation_threshold: float = SYNC_CORRELATION_THRESHOLD,
    progress_cb: Optional[Callable[[str, int, int], None]] = None
) -> List[SyncResult]:
    """
    Synchronizes clips with mixer audio and exports them losslessly with blended audio.
    """
    output_path = pathlib.Path(output_dir)
    unsynced_path = output_path / "unsynchronized"
    output_path.mkdir(parents=True, exist_ok=True)
    unsynced_path.mkdir(parents=True, exist_ok=True)

    clips = scan_clips(clips_dir)
    audio_files = scan_audio(audio_dir)

    results: List[SyncResult] = []

    if not clips:
        logger.warning("No clips found in %s", clips_dir)
        return results

    # Convert volume percentages to gains (0.0 to 2.0)
    cam_gain = max(0.0, min(2.0, camera_vol_pct / 100.0))
    mix_gain = max(0.0, min(2.0, mixer_vol_pct / 100.0))

    with tempfile.TemporaryDirectory(prefix="shabeng_sync_") as tmp_dir_str:
        tmp_dir = pathlib.Path(tmp_dir_str)

        # Step 1 — Concatenate/prepare master mixer audio
        master_wav = tmp_dir / "master_mixer.wav"
        has_audio = False

        if audio_files:
            if progress_cb:
                progress_cb("Preparing master mixer audio track...", 5, 100)
            
            # Sort audio files by filename
            sorted_audio = sorted(audio_files, key=lambda a: a.filename)
            if len(sorted_audio) == 1:
                has_audio = _extract_pcm_audio(sorted_audio[0].path, master_wav)
            else:
                # Concat audio files using ffmpeg concat demuxer
                concat_list = tmp_dir / "concat_list.txt"
                with open(concat_list, "w") as f:
                    for af in sorted_audio:
                        f.write(f"file '{af.path.resolve()}'\n")
                cmd = [
                    "ffmpeg", "-y", "-v", "quiet",
                    "-f", "concat", "-safe", "0", "-i", str(concat_list),
                    "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", str(master_wav)
                ]
                subprocess.run(cmd, capture_output=True, timeout=300)
                has_audio = master_wav.exists() and master_wav.stat().st_size > 0

        master_pcm = _load_wav_data(master_wav) if has_audio else np.array([])

        total_clips = len(clips)
        temp_synced_matches: List[Tuple[ClipInfo, float, float]] = []

        # Step 2 — Process each clip for audio correlation
        for idx, clip in enumerate(clips, start=1):
            pct = int(10 + (idx / total_clips) * 50)
            msg = f"Analyzing audio waveform for {clip.filename} ({idx}/{total_clips})..."
            if progress_cb:
                progress_cb(msg, pct, 100)

            clip_wav = tmp_dir / f"clip_{idx}.wav"
            extracted = _extract_pcm_audio(clip.path, clip_wav)
            
            if extracted and len(master_pcm) > 0:
                clip_pcm = _load_wav_data(clip_wav)
                offset_sec, score = _find_audio_offset(clip_pcm, master_pcm)

                logger.info("Clip %s: score=%.3f, offset=%.2fs", clip.filename, score, offset_sec)

                if score >= correlation_threshold:
                    temp_synced_matches.append((clip, offset_sec, score))
                else:
                    # Sync failed -> Copy to unsynchronized
                    target_unsynced = unsynced_path / clip.filename
                    shutil.copy2(clip.path, target_unsynced)
                    results.append(SyncResult(
                        clip_filename=clip.filename,
                        original_path=str(clip.path),
                        synced=False,
                        correlation_score=score,
                        output_path=str(target_unsynced),
                        error_msg=f"Correlation score ({score:.2f}) below threshold ({correlation_threshold})"
                    ))
            else:
                # No audio or extraction failed -> Unsynchronized
                target_unsynced = unsynced_path / clip.filename
                shutil.copy2(clip.path, target_unsynced)
                results.append(SyncResult(
                    clip_filename=clip.filename,
                    original_path=str(clip.path),
                    synced=False,
                    correlation_score=0.0,
                    output_path=str(target_unsynced),
                    error_msg="Failed to extract audio or no master mixer track present"
                ))

        # Step 3 — Sort synced clips chronologically based on offset_sec
        temp_synced_matches.sort(key=lambda item: item[1])

        # Step 4 — Lossless Video Export (-c:v copy) & Chronological Renaming
        synced_count = len(temp_synced_matches)
        for i, (clip, offset_sec, score) in enumerate(temp_synced_matches, start=1):
            pct = int(60 + (i / synced_count) * 35) if synced_count > 0 else 95
            if progress_cb:
                progress_cb(f"Exporting lossless synced video {i}/{synced_count}: {clip.filename}...", pct, 100)

            # Chronological naming format: 01_filename.mp4, 02_filename.mp4...
            clean_name = clip.filename.lstrip("0123456789_")
            out_filename = f"{i:02d}_{clean_name}"
            out_file = output_path / out_filename

            # Lossless ffmpeg muxing command:
            # -c:v copy preserves video stream 100% untouched
            # amix filter blends camera audio + mixer audio at specified volume gains
            cmd = [
                "ffmpeg", "-y", "-v", "warning",
                "-i", str(clip.path),
                "-ss", str(offset_sec), "-i", str(master_wav),
                "-t", str(clip.duration_sec),
                "-filter_complex",
                f"[0:a]volume={cam_gain:.2f}[a0];[1:a]volume={mix_gain:.2f}[a1];[a0][a1]amix=inputs=2:duration=first[aout]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "[aout]",
                str(out_file)
            ]

            res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if res.returncode == 0 and out_file.exists():
                results.append(SyncResult(
                    clip_filename=clip.filename,
                    original_path=str(clip.path),
                    synced=True,
                    offset_sec=offset_sec,
                    correlation_score=score,
                    output_path=str(out_file),
                    chronological_index=i
                ))
            else:
                logger.error("FFmpeg mux failed for %s: %s", clip.filename, res.stderr)
                # Fallback to unsynchronized
                target_unsynced = unsynced_path / clip.filename
                shutil.copy2(clip.path, target_unsynced)
                results.append(SyncResult(
                    clip_filename=clip.filename,
                    original_path=str(clip.path),
                    synced=False,
                    correlation_score=score,
                    output_path=str(target_unsynced),
                    error_msg=f"Muxing failed: {res.stderr[:100]}"
                ))

        if progress_cb:
            progress_cb("Audio synchronization and export complete!", 100, 100)

    return results
