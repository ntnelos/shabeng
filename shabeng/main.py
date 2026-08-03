"""
main.py — CLI entry-point for the Shabeng pipeline
====================================================

Usage:
    python -m shabeng.main --clips ./clips --audio ./audio --output ./output

Or simply:
    python -m shabeng.main      (uses default paths from config)
"""

import argparse
import json
import logging
import pathlib
import sys
import time

from shabeng.config import (
    DEFAULT_CLIPS_DIR,
    DEFAULT_AUDIO_DIR,
    DEFAULT_OUTPUT_DIR,
    TIMELINE_NAME,
)
from shabeng.ingest import scan_clips, scan_audio
from shabeng.analyzer import analyze_all
from shabeng.director import select_clips, compute_audio_sync
from shabeng.fcpxml import generate_fcpxml


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shabeng",
        description="Shabeng — Automated Wedding Reel Pipeline",
    )
    parser.add_argument(
        "--clips", "-c",
        default=DEFAULT_CLIPS_DIR,
        help="Path to directory containing raw video clips (default: ./clips)",
    )
    parser.add_argument(
        "--audio", "-a",
        default=DEFAULT_AUDIO_DIR,
        help="Path to directory containing external audio recordings (default: ./audio)",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the FCPXML file (default: ./output)",
    )
    parser.add_argument(
        "--min-clips",
        type=int, default=None,
        help="Override minimum number of clips to select",
    )
    parser.add_argument(
        "--max-clips",
        type=int, default=None,
        help="Override maximum number of clips to select",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and analyze clips but skip FCPXML generation",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger("shabeng")
    logger.info("=" * 60)
    logger.info("  🎬  SHABENG — Wedding Reel Pipeline  🎬")
    logger.info("=" * 60)

    start_time = time.time()

    # ── 1. Ingestion ────────────────────────────────────────────────
    logger.info("\n📂 STEP 1: Ingesting clips from %s", args.clips)
    clips = scan_clips(args.clips)
    if not clips:
        logger.error("No video clips found in %s. Exiting.", args.clips)
        sys.exit(1)

    logger.info("\n🎵 STEP 1b: Scanning audio from %s", args.audio)
    audio_files = scan_audio(args.audio)

    # ── 2. AI Analysis ──────────────────────────────────────────────
    logger.info("\n🧠 STEP 2: Analyzing clips with Gemini AI…")
    analyses = analyze_all(clips)

    # Save raw analysis to JSON for debugging
    analysis_out = pathlib.Path(args.output) / "analysis_results.json"
    analysis_out.parent.mkdir(parents=True, exist_ok=True)
    analysis_data = []
    for a in analyses:
        analysis_data.append({
            "filename": a.clip.filename,
            "path": str(a.clip.path),
            "duration_sec": a.clip.duration_sec,
            "energy_level": a.energy_level,
            "emotion_level": a.emotion_level,
            "category": a.category,
            "is_usable": a.is_usable,
            "best_start_sec": a.best_start_sec,
            "best_end_sec": a.best_end_sec,
            "description": a.description,
            "error": a.error,
        })
    with open(analysis_out, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    logger.info("Analysis saved to %s", analysis_out)

    # ── 3. Director Selection ───────────────────────────────────────
    logger.info("\n🎬 STEP 3: Selecting best clips…")
    kwargs = {}
    if args.min_clips is not None:
        kwargs["min_clips"] = args.min_clips
    if args.max_clips is not None:
        kwargs["max_clips"] = args.max_clips

    selected = select_clips(analyses, **kwargs)
    if not selected:
        logger.error("No clips selected. Check your footage quality.")
        sys.exit(1)

    # ── 4. Audio Sync ───────────────────────────────────────────────
    logger.info("\n🔊 STEP 4: Computing audio sync…")
    audio_sync = compute_audio_sync(selected, audio_files)

    if args.dry_run:
        logger.info("\n🏁 Dry run complete. Skipping FCPXML generation.")
        elapsed = time.time() - start_time
        logger.info("Total time: %.1fs", elapsed)
        return

    # ── 5. FCPXML Generation ────────────────────────────────────────
    logger.info("\n📝 STEP 5: Generating FCPXML…")
    output_file = str(pathlib.Path(args.output) / "wedding_reel.fcpxml")
    result_path = generate_fcpxml(
        selected=selected,
        audio_sync=audio_sync,
        output_path=output_file,
        audio_files=audio_files,
    )

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("  ✅  DONE!")
    logger.info("  FCPXML: %s", result_path)
    logger.info("  Clips: %d selected from %d analyzed", len(selected), len(clips))
    logger.info("  Time:  %.1fs", elapsed)
    logger.info("=" * 60)
    logger.info("\n💡 Open the .fcpxml file in Final Cut Pro to import your reel.")


if __name__ == "__main__":
    main()
