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

# High quality sample rate for capturing full spectral harmonics (22.05 kHz)
SAMPLE_RATE = 22050


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
    """Extract mono 16-bit PCM WAV at 22050 Hz using FFmpeg."""
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
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio /= max_val
            return audio
    except Exception as exc:
        logger.warning("Error reading WAV data from %s: %s", wav_path.name, exc)
        return np.array([], dtype=np.float32)


def _fast_zncc_1d(master: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """
    Computes exact Zero-Mean Normalized Cross-Correlation (ZNCC) for a single frequency band.
    """
    L = len(clip)
    N = len(master)
    if L > N or L == 0:
        return np.zeros(max(0, N - L + 1), dtype=np.float64)

    c_zero = clip - np.mean(clip)
    c_norm = float(np.linalg.norm(c_zero))
    if c_norm == 0:
        return np.zeros(N - L + 1, dtype=np.float64)

    raw_corr = signal.fftconvolve(master, c_zero[::-1], mode='valid')

    ones = np.ones(L, dtype=np.float64)
    m_sum = signal.fftconvolve(master.astype(np.float64), ones, mode='valid')
    m_sq_sum = signal.fftconvolve((master.astype(np.float64))**2, ones, mode='valid')

    m_var = m_sq_sum - (m_sum**2 / L)
    m_var = np.maximum(m_var, 1e-9)
    m_std = np.sqrt(m_var)

    return raw_corr / (m_std * c_norm + 1e-7)


def _compute_spectrogram_bands(audio: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = 32) -> np.ndarray:
    """
    Computes 2D Spectrogram across 32 log-spaced frequency bands (100Hz - 8000Hz).
    Isolates speech/music harmonics from crowd noise and acoustic reverb.
    """
    if len(audio) < sr * 0.2:
        return np.array([[]], dtype=np.float32)

    try:
        nyq = sr / 2.0
        b, a = signal.butter(3, [100.0 / nyq, 8000.0 / nyq], btype='band')
        filt = signal.filtfilt(b, a, audio)
    except Exception:
        filt = audio

    n_fft = 1024
    hop_length = 220  # ~10ms time resolution

    f, t, Sxx = signal.spectrogram(filt, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, window='hann')

    S_db = np.log1p(10.0 * np.abs(Sxx))
    n_freqs = S_db.shape[0]

    band_edges = np.logspace(np.log10(2), np.log10(n_freqs - 1), n_mels + 1).astype(int)

    bands = []
    for b in range(n_mels):
        low_idx = band_edges[b]
        high_idx = max(low_idx + 1, band_edges[b + 1])
        band_energy = np.mean(S_db[low_idx:high_idx, :], axis=0)
        bands.append(band_energy)

    return np.array(bands, dtype=np.float32)


def _find_audio_offset(clip_audio: np.ndarray, master_audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Tuple[float, float]:
    """
    Final Cut Pro level 2D Multi-Band Spectrogram ZNCC Sync engine.
    Returns (best_offset_sec, peak_correlation_score).
    """
    if len(clip_audio) < sample_rate * 0.4 or len(master_audio) < len(clip_audio):
        return 0.0, 0.0

    m_spec = _compute_spectrogram_bands(master_audio, sr=sample_rate, n_mels=32)
    c_spec = _compute_spectrogram_bands(clip_audio, sr=sample_rate, n_mels=32)

    if m_spec.size == 0 or c_spec.size == 0 or m_spec.shape[1] < c_spec.shape[1]:
        return 0.0, 0.0

    n_bands, n_master = m_spec.shape
    _, n_clip = c_spec.shape

    total_zncc = np.zeros(n_master - n_clip + 1, dtype=np.float64)

    valid_bands = 0
    for b in range(n_bands):
        m_b = m_spec[b, :]
        c_b = c_spec[b, :]

        zncc = _fast_zncc_1d(m_b, c_b)
        if len(zncc) == len(total_zncc):
            total_zncc += zncc
            valid_bands += 1

    if valid_bands > 0:
        total_zncc /= valid_bands

    best_idx = int(np.argmax(total_zncc))
    peak_score = float(total_zncc[best_idx])

    # Hop length = 220 samples at 22050Hz (~10ms)
    hop_dur = 220.0 / sample_rate
    offset_sec = float(best_idx * hop_dur)

    return round(offset_sec, 3), round(peak_score, 3)


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
    # Debug log to file for diagnosing app-level issues
    import datetime
    _debug_log = pathlib.Path.home() / "Desktop" / "shabeng_debug.log"
    def _dbg(msg):
        with open(_debug_log, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] {msg}\n")
        logger.info(msg)

    _dbg(f"=== sync_and_export_clips START ===")
    _dbg(f"  clips_dir={clips_dir}")
    _dbg(f"  audio_dir={audio_dir}")
    _dbg(f"  output_dir={output_dir}")
    _dbg(f"  camera_vol_pct={camera_vol_pct}, mixer_vol_pct={mixer_vol_pct}")
    _dbg(f"  correlation_threshold={correlation_threshold}")
    _dbg(f"  SYNC_CORRELATION_THRESHOLD (config)={SYNC_CORRELATION_THRESHOLD}")

    output_path = pathlib.Path(output_dir).expanduser()
    unsynced_path = output_path / "unsynchronized"
    output_path.mkdir(parents=True, exist_ok=True)
    unsynced_path.mkdir(parents=True, exist_ok=True)

    clips = scan_clips(clips_dir)
    audio_files = scan_audio(audio_dir)

    _dbg(f"  Found {len(clips)} clips, {len(audio_files)} audio files")

    results: List[SyncResult] = []

    if not clips:
        _dbg(f"  ERROR: No clips found in {clips_dir}")
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
            _dbg(f"  Audio files: {[a.filename for a in sorted_audio]}")
            if len(sorted_audio) == 1:
                has_audio = _extract_pcm_audio(sorted_audio[0].path, master_wav)
                _dbg(f"  Single audio extraction: has_audio={has_audio}")
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
                _dbg(f"  Concat audio extraction: has_audio={has_audio}")

        master_pcm = _load_wav_data(master_wav) if has_audio else np.array([])
        _dbg(f"  Master PCM length: {len(master_pcm)} samples ({len(master_pcm)/SAMPLE_RATE:.1f}s)" if len(master_pcm) > 0 else "  Master PCM: EMPTY")

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
            _dbg(f"  Clip {clip.filename}: extracted={extracted}")
            
            if extracted and len(master_pcm) > 0:
                clip_pcm = _load_wav_data(clip_wav)
                _dbg(f"    clip_pcm length: {len(clip_pcm)} samples ({len(clip_pcm)/SAMPLE_RATE:.1f}s)")
                offset_sec, score = _find_audio_offset(clip_pcm, master_pcm)

                _dbg(f"    RESULT: score={score:.4f}, offset={offset_sec:.3f}s, threshold={correlation_threshold}")
                logger.info("Clip %s: score=%.3f, offset=%.2fs", clip.filename, score, offset_sec)

                if score >= correlation_threshold:
                    _dbg(f"    SYNCED! score {score:.4f} >= threshold {correlation_threshold}")
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
