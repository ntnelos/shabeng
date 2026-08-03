"""
analyzer.py — Gemini AI video analysis (The Brain)
====================================================

Uploads each clip to the Gemini File API, waits for processing,
then prompts gemini-1.5-pro to return structured JSON describing
the clip's energy, emotion, category, usability, and best segment.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import google.generativeai as genai

from shabeng.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_UPLOAD_DELAY_SEC,
    GEMINI_MAX_RETRIES,
    CLIP_MAX_DURATION_SEC,
    CLIP_MIN_DURATION_SEC,
)
from shabeng.ingest import ClipInfo

logger = logging.getLogger(__name__)

# ── Configure the Gemini SDK ───────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)

# ── Analysis prompt ─────────────────────────────────────────────────
_ANALYSIS_PROMPT = f"""You are a professional wedding video editor.
Analyze this short wedding clip and return ONLY a valid JSON object — no markdown,
no code fences, no explanation. The JSON must contain exactly these keys:

{{
  "Energy_Level": <integer 1-10, how energetic / dynamic the clip is>,
  "Emotion_Level": <integer 1-10, how emotionally impactful>,
  "Category": "<one of: Dancing, Chuppah, Singer, Crowd, Couple, Decoration, Food, Speech, Entrance, Other>",
  "Is_Usable": <boolean — false if camera is too shaky, blurry, dark, or pointing at the floor>,
  "Best_Start_Sec": <float — recommended start second for the best {CLIP_MIN_DURATION_SEC}-{CLIP_MAX_DURATION_SEC}s segment>,
  "Best_End_Sec": <float — recommended end second for that segment>,
  "Description": "<one-sentence description of what happens>"
}}

Rules:
- Best_Start_Sec and Best_End_Sec must define a segment between {CLIP_MIN_DURATION_SEC} and {CLIP_MAX_DURATION_SEC} seconds long.
- Best_Start_Sec must be >= 0.
- Best_End_Sec must not exceed the clip's total duration.
- Pick the segment with the highest visual energy and emotion.
- If the entire clip is shorter than {CLIP_MIN_DURATION_SEC}s, use the full clip.
- Return ONLY the JSON. Nothing else.
"""


@dataclass
class ClipAnalysis:
    """Parsed analysis result for one clip."""
    clip: ClipInfo
    energy_level: int = 0
    emotion_level: int = 0
    category: str = "Other"
    is_usable: bool = False
    best_start_sec: float = 0.0
    best_end_sec: float = 0.0
    description: str = ""
    raw_json: dict = field(default_factory=dict)
    error: Optional[str] = None


def _upload_and_wait(clip: ClipInfo) -> Optional[object]:
    """Upload a video file to the Gemini File API and poll until ready."""
    logger.info("Uploading %s to Gemini…", clip.filename)
    try:
        uploaded = genai.upload_file(
            path=str(clip.path),
            display_name=clip.filename,
        )
    except Exception as exc:
        logger.error("Upload failed for %s: %s", clip.filename, exc)
        return None

    # Poll until the file finishes processing
    max_wait = 120  # seconds
    poll_interval = 5
    elapsed = 0
    while uploaded.state.name == "PROCESSING" and elapsed < max_wait:
        logger.debug("  %s still processing… (%ds)", clip.filename, elapsed)
        time.sleep(poll_interval)
        uploaded = genai.get_file(uploaded.name)
        elapsed += poll_interval

    if uploaded.state.name == "ACTIVE":
        logger.info("  %s ready for analysis.", clip.filename)
        return uploaded
    else:
        logger.error("  %s stuck in state %s after %ds",
                      clip.filename, uploaded.state.name, elapsed)
        return None


def _parse_response(text: str) -> dict:
    """Try to extract valid JSON from the model response."""
    # Strip possible markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove first line (```json) and last line (```)
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]).strip()

    return json.loads(cleaned)


def analyze_clip(clip: ClipInfo) -> ClipAnalysis:
    """
    Upload one clip to Gemini, prompt for analysis, return ClipAnalysis.
    Retries on transient errors up to GEMINI_MAX_RETRIES times.
    """
    analysis = ClipAnalysis(clip=clip)

    uploaded_file = _upload_and_wait(clip)
    if uploaded_file is None:
        analysis.error = "Upload/processing failed"
        return analysis

    model = genai.GenerativeModel(model_name=GEMINI_MODEL)

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                [uploaded_file, _ANALYSIS_PROMPT],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=512,
                ),
            )
            data = _parse_response(response.text)

            # Populate the analysis dataclass
            analysis.energy_level = int(data.get("Energy_Level", 0))
            analysis.emotion_level = int(data.get("Emotion_Level", 0))
            analysis.category = str(data.get("Category", "Other"))
            analysis.is_usable = bool(data.get("Is_Usable", False))
            analysis.best_start_sec = float(data.get("Best_Start_Sec", 0))
            analysis.best_end_sec = float(data.get("Best_End_Sec", clip.duration_sec))
            analysis.description = str(data.get("Description", ""))
            analysis.raw_json = data

            # Clamp in/out points to valid range
            analysis.best_start_sec = max(0.0, analysis.best_start_sec)
            analysis.best_end_sec = min(clip.duration_sec, analysis.best_end_sec)
            if analysis.best_end_sec <= analysis.best_start_sec:
                analysis.best_start_sec = 0.0
                analysis.best_end_sec = min(CLIP_MAX_DURATION_SEC, clip.duration_sec)

            logger.info(
                "  ✓ %s → Energy=%d, Emotion=%d, Cat=%s, Usable=%s, Seg=%.1f-%.1fs",
                clip.filename, analysis.energy_level, analysis.emotion_level,
                analysis.category, analysis.is_usable,
                analysis.best_start_sec, analysis.best_end_sec
            )
            break  # success

        except json.JSONDecodeError as exc:
            logger.warning("Attempt %d: JSON parse error for %s: %s",
                           attempt, clip.filename, exc)
            if attempt == GEMINI_MAX_RETRIES:
                analysis.error = f"JSON parse failed after {GEMINI_MAX_RETRIES} attempts"
        except Exception as exc:
            logger.warning("Attempt %d: Gemini error for %s: %s",
                           attempt, clip.filename, exc)
            if attempt < GEMINI_MAX_RETRIES:
                time.sleep(GEMINI_UPLOAD_DELAY_SEC * attempt)  # exponential-ish backoff
            else:
                analysis.error = str(exc)

    # Clean up the uploaded file
    try:
        genai.delete_file(uploaded_file.name)
    except Exception:
        pass

    return analysis


def analyze_all(clips: List[ClipInfo]) -> List[ClipAnalysis]:
    """
    Analyze every clip in the list, respecting rate limits.
    Returns a list of ClipAnalysis objects.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Create a .env file with GEMINI_API_KEY=your-key"
        )

    results: List[ClipAnalysis] = []
    for i, clip in enumerate(clips):
        logger.info("── Analyzing clip %d/%d: %s ──", i + 1, len(clips), clip.filename)
        analysis = analyze_clip(clip)
        results.append(analysis)

        # Rate-limit pause between uploads
        if i < len(clips) - 1:
            logger.debug("Rate-limit pause (%.1fs)…", GEMINI_UPLOAD_DELAY_SEC)
            time.sleep(GEMINI_UPLOAD_DELAY_SEC)

    ok = sum(1 for r in results if r.error is None)
    logger.info("Analysis complete: %d/%d clips analyzed successfully.", ok, len(clips))
    return results
