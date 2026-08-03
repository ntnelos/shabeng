"""
director.py — The Director Logic
==================================

Filters, ranks, and selects the best 10-15 clips for the reel.
Prioritizes: Is_Usable → Dancing category → high Energy → high Emotion.
Also computes audio sync alignment for external mixer recordings.
"""

import logging
from typing import List, Tuple

from shabeng.analyzer import ClipAnalysis
from shabeng.ingest import AudioFileInfo
from shabeng.config import MIN_CLIPS, MAX_CLIPS

logger = logging.getLogger(__name__)


def _score(analysis: ClipAnalysis) -> float:
    """
    Compute a composite score for clip ranking.
    Dancing clips get a significant boost.
    """
    base = analysis.energy_level * 2 + analysis.emotion_level
    if analysis.category.lower() == "dancing":
        base += 10
    elif analysis.category.lower() in ("entrance", "couple"):
        base += 5
    elif analysis.category.lower() in ("chuppah", "singer"):
        base += 3
    return float(base)


def select_clips(
    analyses: List[ClipAnalysis],
    min_clips: int = MIN_CLIPS,
    max_clips: int = MAX_CLIPS,
) -> List[ClipAnalysis]:
    """
    Filter and rank clips, returning the top *max_clips* usable clips.
    Falls back to fewer clips if not enough are usable.
    """
    # Step 1 — Filter usable only
    usable = [a for a in analyses if a.is_usable and a.error is None]
    logger.info("Usable clips: %d / %d total", len(usable), len(analyses))

    if not usable:
        logger.warning("No usable clips found! Returning empty selection.")
        return []

    # Step 2 — Score & sort descending
    ranked = sorted(usable, key=_score, reverse=True)

    # Step 3 — Take top N
    selected = ranked[:max_clips]

    if len(selected) < min_clips:
        logger.warning(
            "Only %d usable clips found (target: %d-%d).",
            len(selected), min_clips, max_clips
        )

    for i, a in enumerate(selected, 1):
        logger.info(
            "  #%02d  score=%.0f  %s  [%s]  %.1f-%.1fs  %s",
            i, _score(a), a.clip.filename, a.category,
            a.best_start_sec, a.best_end_sec, a.description[:60]
        )

    return selected


def compute_audio_sync(
    selected: List[ClipAnalysis],
    audio_files: List[AudioFileInfo],
) -> List[Tuple[AudioFileInfo, float, float]]:
    """
    For the MVP, we do a simple positional sync:
    - Concatenate the external audio files in order.
    - Map each clip's timeline position to a matching offset in the
      concatenated audio stream.

    Returns a list of (AudioFileInfo, start_offset_sec, duration_sec)
    tuples — one per clip — indicating which part of which audio file
    to lay on the timeline aligned with that clip.

    In Phase 2, we can do real audio fingerprint matching.
    """
    if not audio_files:
        logger.info("No external audio files — skipping audio sync.")
        return []

    # Build a timeline of the audio files (concatenated)
    audio_timeline: List[Tuple[AudioFileInfo, float, float]] = []
    cursor = 0.0
    for af in sorted(audio_files, key=lambda a: a.filename):
        audio_timeline.append((af, cursor, af.duration_sec))
        cursor += af.duration_sec
    total_audio_dur = cursor
    logger.info("Total external audio duration: %.1fs across %d file(s)",
                total_audio_dur, len(audio_files))

    # Walk the clip timeline and map each clip to an audio region
    sync_plan: List[Tuple[AudioFileInfo, float, float]] = []
    clip_cursor = 0.0
    for analysis in selected:
        seg_dur = analysis.best_end_sec - analysis.best_start_sec
        # Find which audio file covers clip_cursor
        matched = False
        for af, af_start, af_dur in audio_timeline:
            af_end = af_start + af_dur
            if af_start <= clip_cursor < af_end:
                offset_in_file = clip_cursor - af_start
                sync_plan.append((af, offset_in_file, seg_dur))
                matched = True
                break
        if not matched:
            # If clips extend beyond audio, loop/wrap
            wrapped_pos = clip_cursor % total_audio_dur if total_audio_dur > 0 else 0
            for af, af_start, af_dur in audio_timeline:
                af_end = af_start + af_dur
                if af_start <= wrapped_pos < af_end:
                    offset_in_file = wrapped_pos - af_start
                    sync_plan.append((af, offset_in_file, seg_dur))
                    break

        clip_cursor += seg_dur

    logger.info("Audio sync plan computed for %d clips.", len(sync_plan))
    return sync_plan
