# 🎬 Shabeng — Automated Wedding Reel Pipeline

Shabeng analyzes raw wedding video clips using **Gemini AI**, selects the best moments,
syncs external mixer audio, and outputs a **Final Cut Pro FCPXML** file for a 9:16 Instagram Reel.

---

## Architecture

```
clips/  ──→  Ingest  ──→  Gemini AI Analysis  ──→  Director  ──→  FCPXML
audio/  ──────────────────────────────────────────→  Audio Sync ──↗
```

| Module         | Purpose                                                      |
|----------------|--------------------------------------------------------------|
| `ingest.py`    | Scans directories, probes video/audio metadata with ffprobe  |
| `analyzer.py`  | Uploads clips to Gemini, extracts structured JSON analysis   |
| `director.py`  | Filters/ranks clips, computes audio sync alignment           |
| `fcpxml.py`    | Generates valid FCPXML 1.11 with trimmed clips + audio lanes |
| `main.py`      | CLI entry-point orchestrating the full pipeline              |

---

## Prerequisites

1. **Python 3.10+**
2. **ffprobe** (comes with FFmpeg):
   ```bash
   brew install ffmpeg
   ```
3. **Gemini API Key** — get one at [Google AI Studio](https://aistudio.google.com/apikey)

---

## Setup

```bash
# 1. Clone / navigate to the project
cd Shabeng

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY
```

---

## Usage

### Basic (default directories)

```bash
# Place your .mp4/.MOV clips in ./clips/
# Place your mixer audio (.wav/.mp3) in ./audio/
python -m shabeng.main
```

The FCPXML will be written to `./output/wedding_reel.fcpxml`.

### Custom directories

```bash
python -m shabeng.main \
  --clips /path/to/wedding/footage \
  --audio /path/to/mixer/recordings \
  --output /path/to/output
```

### Options

| Flag              | Description                                |
|-------------------|--------------------------------------------|
| `--clips`, `-c`   | Video clips directory                      |
| `--audio`, `-a`   | External audio directory                   |
| `--output`, `-o`  | Output directory                           |
| `--min-clips`     | Minimum clips to select (default: 10)      |
| `--max-clips`     | Maximum clips to select (default: 15)      |
| `--verbose`, `-v` | Debug-level logging                        |
| `--dry-run`       | Analyze clips but skip FCPXML generation   |

---

## How It Works

### 1. Ingestion
Scans `./clips/` for `.mp4`/`.MOV` files and `./audio/` for `.wav`/`.mp3` files.
Uses **ffprobe** to extract duration, fps, resolution, and sample rate.

### 2. AI Analysis (The Brain)
Each clip is uploaded to the **Gemini File API** and analyzed by `gemini-1.5-pro`.
The model returns structured JSON:
```json
{
  "Energy_Level": 8,
  "Emotion_Level": 7,
  "Category": "Dancing",
  "Is_Usable": true,
  "Best_Start_Sec": 2.5,
  "Best_End_Sec": 7.0,
  "Description": "Crowd dancing in circle with bride"
}
```

### 3. Director Selection
- Filters out unusable clips (`Is_Usable=false`)
- Scores clips: `Energy×2 + Emotion + category_bonus`
- Dancing clips get a +10 bonus
- Selects the top 10–15 clips

### 4. Audio Sync
- Concatenates external audio files in filename order
- Maps each clip's timeline position to the corresponding audio offset
- Wraps around if clips extend beyond audio length

### 5. FCPXML Generation
- Creates a valid FCPXML 1.11 document
- 9:16 timeline (1080×1920) at 30fps
- Each clip is trimmed to its AI-recommended segment (`Best_Start_Sec` → `Best_End_Sec`)
- Original clip audio: **30% volume** (-10.5 dB)
- External mixer audio: **70% volume** (-3 dB)

---

## Output

```
output/
├── analysis_results.json   ← Full AI analysis for every clip
└── wedding_reel.fcpxml     ← Import this into Final Cut Pro
```

---

## Importing into Final Cut Pro

1. Open **Final Cut Pro**
2. Go to **File → Import → XML…**
3. Select `wedding_reel.fcpxml`
4. A new Event called "Shabeng Reel" will appear with the project inside
5. The timeline will have all clips trimmed and audio synced — ready to fine-tune!

---

## Configuration

Edit `shabeng/config.py` to adjust:

| Setting                  | Default  | Description                          |
|--------------------------|----------|--------------------------------------|
| `TIMELINE_FPS`           | 30       | Timeline frame rate                  |
| `CLIP_MAX_DURATION_SEC`  | 6.0      | Max segment length per clip          |
| `CLIP_MIN_DURATION_SEC`  | 1.5      | Min segment length per clip          |
| `MIN_CLIPS`              | 10       | Minimum clips in the reel            |
| `MAX_CLIPS`              | 15       | Maximum clips in the reel            |
| `ORIGINAL_AUDIO_GAIN_DB` | -10.5    | Original audio volume (≈30%)         |
| `EXTERNAL_AUDIO_GAIN_DB` | -3.0     | External audio volume (≈70%)         |

---

## Phase 2 (Planned)
- [ ] Google Drive API integration for remote clip ingestion
- [ ] Audio fingerprint-based sync (replace positional sync)
- [ ] Beat detection for cut timing
- [ ] Multiple reel templates (highlights, cinematic, party)
