"""
End-to-end sync test on real files - bypasses server/UI entirely.
Uses the same sync_and_export_clips function that the app uses.
"""
import sys, os, logging

# Enable verbose logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shabeng.sync_engine import sync_and_export_clips
from shabeng.config import SYNC_CORRELATION_THRESHOLD

CLIP_PATH = "/Users/netanelyosef/Library/CloudStorage/GoogleDrive-kolotmusic@gmail.com/האחסון שלי/קולות וידאו/אריאלה ואנדרו 2.8.26/ריל/IMG_4607.MOV"
AUDIO_PATH = "/Users/netanelyosef/Library/CloudStorage/GoogleDrive-kolotmusic@gmail.com/האחסון שלי/קולות וידאו/אריאלה ואנדרו 2.8.26/סאונד/hupa 2.8.26.mp3"
OUTPUT_DIR = "/tmp/shabeng_test_output"

def progress(msg, pct, total):
    print(f"  [{pct}%] {msg}")

if __name__ == "__main__":
    print(f"=== Shabeng End-to-End Sync Test ===")
    print(f"  Clip:      {CLIP_PATH}")
    print(f"  Audio:     {AUDIO_PATH}")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"  Threshold: {SYNC_CORRELATION_THRESHOLD}")
    print(f"  Clip exists: {os.path.exists(CLIP_PATH)}")
    print(f"  Audio exists: {os.path.exists(AUDIO_PATH)}")
    print()

    results = sync_and_export_clips(
        clips_dir=CLIP_PATH,
        audio_dir=AUDIO_PATH,
        output_dir=OUTPUT_DIR,
        camera_vol_pct=30.0,
        mixer_vol_pct=70.0,
        correlation_threshold=SYNC_CORRELATION_THRESHOLD,
        progress_cb=progress
    )

    print(f"\n=== RESULTS ({len(results)} clips) ===")
    for r in results:
        d = r.to_dict()
        print(f"  Clip: {d['clip_filename']}")
        print(f"    synced:     {d['synced']}")
        print(f"    score:      {d['correlation_score']}")
        print(f"    offset:     {d['offset_sec']}s")
        print(f"    output:     {d['output_path']}")
        print(f"    error:      {d['error_msg']}")
        print()
