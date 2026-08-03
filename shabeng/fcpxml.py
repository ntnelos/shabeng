"""
fcpxml.py — FCPXML 1.11 Generator for Final Cut Pro
=====================================================

Builds a structurally valid .fcpxml file (version 1.11) that FCP can import.
Places selected clips on a 9:16 (1080×1920) timeline with correct in/out
points, and lays external audio on a separate lane with gain adjustments.

FCPXML time format: "{numerator}/{denominator}s"  (e.g. "30/30s" = 1 second)
We use frame-accurate rational time with the timeline's frame rate as
denominator to avoid floating-point drift.
"""

import pathlib
import logging
from typing import List, Tuple, Optional
from urllib.request import pathname2url
from xml.etree.ElementTree import (
    Element, SubElement, ElementTree, indent,
)

from shabeng.analyzer import ClipAnalysis
from shabeng.ingest import AudioFileInfo
from shabeng.config import (
    TIMELINE_WIDTH,
    TIMELINE_HEIGHT,
    TIMELINE_FPS,
    TIMELINE_NAME,
    ORIGINAL_AUDIO_GAIN_DB,
    EXTERNAL_AUDIO_GAIN_DB,
)

logger = logging.getLogger(__name__)


def _file_uri(p: pathlib.Path) -> str:
    """Convert a Path to a file:// URI (works on Python 3.10+)."""
    try:
        return p.as_uri()
    except AttributeError:
        # Fallback for Python < 3.13
        return "file://" + pathname2url(str(p.resolve()))


def _frames(seconds: float, fps: int = TIMELINE_FPS) -> int:
    """Convert seconds to whole frames."""
    return int(round(seconds * fps))


def _ftime(seconds: float, fps: int = TIMELINE_FPS) -> str:
    """Convert seconds to FCPXML rational time string '{frames}/{fps}s'."""
    frames = _frames(seconds, fps)
    return f"{frames}/{fps}s"


def _make_ref_id(index: int) -> str:
    """Generate a unique asset reference id."""
    return f"r{index + 1}"


def _make_audio_ref_id(index: int, base: int) -> str:
    """Generate a unique asset reference id for audio files."""
    return f"r{base + index + 1}"


def generate_fcpxml(
    selected: List[ClipAnalysis],
    audio_sync: List[Tuple[AudioFileInfo, float, float]],
    output_path: str,
    audio_files: Optional[List[AudioFileInfo]] = None,
) -> str:
    """
    Build and write the FCPXML file.

    Args:
        selected:    Ordered list of ClipAnalysis (the reel sequence).
        audio_sync:  Per-clip audio sync data from director.compute_audio_sync().
        output_path: Where to write the .fcpxml file.
        audio_files: All unique audio files (for resource declarations).

    Returns:
        The absolute path to the written file.
    """
    fps = TIMELINE_FPS

    # ── Collect unique audio files for resource declarations ─────────
    unique_audio: List[AudioFileInfo] = []
    if audio_files:
        seen_paths = set()
        for af in audio_files:
            if str(af.path) not in seen_paths:
                seen_paths.add(str(af.path))
                unique_audio.append(af)

    audio_ref_base = len(selected)

    # ── Root <fcpxml> ───────────────────────────────────────────────
    fcpxml = Element("fcpxml", version="1.11")

    # ── <resources> ─────────────────────────────────────────────────
    resources = SubElement(fcpxml, "resources")

    # Format resource (9:16 vertical reel)
    SubElement(resources, "format", {
        "id": "r0",
        "name": f"FFVideoFormat{TIMELINE_HEIGHT}p{fps}",
        "frameDuration": f"1/{fps}s",
        "width": str(TIMELINE_WIDTH),
        "height": str(TIMELINE_HEIGHT),
    })

    # Asset resources for each video clip
    for i, analysis in enumerate(selected):
        clip = analysis.clip
        ref_id = _make_ref_id(i)
        asset = SubElement(resources, "asset", {
            "id": ref_id,
            "name": clip.filename,
            "start": "0/1s",
            "duration": _ftime(clip.duration_sec, fps),
            "hasVideo": "1",
            "hasAudio": "1",
            "format": "r0",
        })
        SubElement(asset, "media-rep", {
            "kind": "original-media",
            "src": _file_uri(clip.path),
        })

    # Asset resources for each audio file
    for i, af in enumerate(unique_audio):
        ref_id = _make_audio_ref_id(i, audio_ref_base)
        asset = SubElement(resources, "asset", {
            "id": ref_id,
            "name": af.filename,
            "start": "0/1s",
            "duration": _ftime(af.duration_sec, fps),
            "hasVideo": "0",
            "hasAudio": "1",
        })
        SubElement(asset, "media-rep", {
            "kind": "original-media",
            "src": _file_uri(af.path),
        })

    # ── <library> → <event> → <project> → <sequence> ───────────────
    library = SubElement(fcpxml, "library")
    event = SubElement(library, "event", name="Shabeng Reel")
    project = SubElement(event, "project", name=TIMELINE_NAME)

    # Calculate total timeline duration
    total_dur = sum(
        a.best_end_sec - a.best_start_sec for a in selected
    )

    sequence = SubElement(project, "sequence", {
        "format": "r0",
        "duration": _ftime(total_dur, fps),
        "tcStart": "0/1s",
        "tcFormat": "NDF",
        "audioLayout": "stereo",
        "audioRate": "48k",
    })

    spine = SubElement(sequence, "spine")

    # ── Build the spine: video clips with trimmed in/out ────────────
    timeline_cursor_sec = 0.0

    # Build an audio-file → ref-id lookup
    audio_ref_map = {}
    for i, af in enumerate(unique_audio):
        audio_ref_map[str(af.path)] = _make_audio_ref_id(i, audio_ref_base)

    for idx, analysis in enumerate(selected):
        clip = analysis.clip
        ref_id = _make_ref_id(idx)

        seg_start = analysis.best_start_sec
        seg_end = analysis.best_end_sec
        seg_dur = seg_end - seg_start

        # <asset-clip> on the spine
        asset_clip = SubElement(spine, "asset-clip", {
            "ref": ref_id,
            "name": clip.filename,
            "offset": _ftime(timeline_cursor_sec, fps),
            "start": _ftime(seg_start, fps),
            "duration": _ftime(seg_dur, fps),
            "tcFormat": "NDF",
            "format": "r0",
        })

        # Adjust original audio volume to 30%
        adjust_orig = SubElement(asset_clip, "adjust-volume", {
            "amount": f"{ORIGINAL_AUDIO_GAIN_DB}dB",
        })

        # ── External audio lane (if sync data exists) ──────────────
        if idx < len(audio_sync):
            af, offset_in_file, duration = audio_sync[idx]
            audio_ref = audio_ref_map.get(str(af.path))
            if audio_ref:
                # Attached audio on lane -1 (below the primary storyline)
                audio_clip = SubElement(asset_clip, "audio", {
                    "ref": audio_ref,
                    "lane": "-1",
                    "name": af.filename,
                    "offset": _ftime(timeline_cursor_sec, fps),
                    "start": _ftime(offset_in_file, fps),
                    "duration": _ftime(duration, fps),
                    "role": "dialogue",
                })
                SubElement(audio_clip, "adjust-volume", {
                    "amount": f"{EXTERNAL_AUDIO_GAIN_DB}dB",
                })

        timeline_cursor_sec += seg_dur

    # ── Write to disk ───────────────────────────────────────────────
    tree = ElementTree(fcpxml)
    indent(tree, space="  ", level=0)

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Write with XML declaration and DOCTYPE
    with open(out, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE fcpxml>\n')
        tree.write(f, encoding="unicode", xml_declaration=False)

    abs_path = str(out.resolve())
    logger.info("FCPXML written to: %s", abs_path)
    logger.info("Timeline: %.1fs total, %d clips, %dx%d @ %dfps",
                total_dur, len(selected), TIMELINE_WIDTH, TIMELINE_HEIGHT, fps)

    return abs_path
