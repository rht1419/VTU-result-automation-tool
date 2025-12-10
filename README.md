# BUILD BY 5th SEM VTU STUDENTS for Mini-Project
  Rohit.R trained the CNN Model.
  Shreyas built the Automation. 
  
# VTU Results Automation System

An automated system for fetching results from VTU (Visvesvaraya Technological University) using Playwright browser automation and a locally trained PyTorch model for CAPTCHA recognition. Includes a real‑time web interface with ultra‑fast batch processing.

# Steps to follow to install in your system:
1. Download & Extract: Unzip the file
2. Run Setup: Double-click setup_wizard.bat
3. Follow Prompts: They'll be asked for permission before each dependency installation
4. Launch App: Use the desktop shortcut or double-click launch_app.bat
5. Access Web Interface: Open browser to http://localhost:5000

## Features

- Ultra‑fast result fetching with Playwright (Chromium) and aggressive optimizations
- AI-powered local CAPTCHA recognition (no external API keys)
- Modern web interface with real‑time status via WebSockets
- Batch naming with per‑batch PDF folders
- Direct form URL input (works across different VTU result batches)
- JSON batch summary output

## Project layout

- `vtu_ultra_fast_web_interface.py` — Flask + Socket.IO web server (uses ultra‑fast automation)
- `latest_vtu_automation_ultra_fast.py` — Ultra‑fast automation core (used by the UI)
- `latest_vtu_automation_2.py` — Safer/slower variant (handy for CLI/debug)
- `captcha_model_recognizer.py` — Loads the trained model and recognizes CAPTCHA images
- `train_captcha_19k.py` — Training script for the CAPTCHA model
- `templates/` — Web UI HTML files
- `pdf_results/` — Generated PDFs (organized by batch)
- `results/` — Batch summary JSON files
- `captchas/` — Temporary CAPTCHA screenshots
- `runs/exp_19k/` — Trained model checkpoint(s)

## Requirements

All dependencies are listed in `requirements.txt`. Key ones:

- Flask, Flask‑SocketIO
- Playwright (automation): Chromium is required
- PyTorch + TorchVision
- Pillow

## Setup

1) Create and activate a virtual environment

```bash
python -m venv vtu_automation_env
# Windows
vtu_automation_env\Scripts\activate
# macOS/Linux
source vtu_automation_env/bin/activate
```

2) Install Python dependencies

```bash
pip install -r requirements.txt
```

3) Install Playwright browser runtime (Chromium)

```bash
playwright install chromium
```

4) Ensure the CAPTCHA model exists

- Expected model path: `runs/exp_19k/best_epoch_134_fullacc_0.9951.pth`
- The recognizer imports components from `train_captcha_19k.py` and will log the model path on first use.

## Running the web interface

```bash
python vtu_ultra_fast_web_interface.py
```

Then open `http://localhost:5000`.

1. Enter the VTU results form URL (e.g., `https://results.vtu.ac.in/JJEcbcs25/index.php`).
2. Set a Batch Name (e.g., `Web Batch`, `Aids_batch`).
3. Paste USNs (one per line).
4. Choose Headless and PDF options.
5. Click Start Processing.

### Batch naming and PDF output

- PDFs are saved per batch in a dedicated subfolder:
  - `pdf_results/<BatchName>/<USN>_<YYYYMMDD_HHMMSS>.pdf`
- Batch names are sanitized for filesystem safety (non‑alphanumeric become `_`).
- Example: Batch Name = `Aids_batch` →
  - `pdf_results/Aids_batch/1SP23AD027_20251106_140102.pdf`

### Results and summaries

- Live status and logs stream to the UI via WebSockets.
- Batch summaries are written to `results/` as JSON, e.g.:
  - `results/summary_ultrafast_<BatchName>_<YYYYMMDD_HHMMSS>.json`

## CLI usage (optional)

You can also run the core scripts directly for testing:

```bash
# Ultra-fast test (uses hardcoded sample in script)
python latest_vtu_automation_ultra_fast.py

# Safer/slower variant
python latest_vtu_automation_2.py
```

## Troubleshooting

- Playwright says browsers not installed
  - Run: `playwright install chromium`

- CAPTCHA model not found / import error
  - Ensure `runs/exp_19k/best_epoch_134_fullacc_0.9951.pth` exists
  - Keep `train_captcha_19k.py` in the project root (it provides model/transforms)

- PDFs not saving under batch folder
  - Confirm you started via the web UI and set a Batch Name
  - The web UI sets the active batch so PDFs save to `pdf_results/<BatchName>/...`

- Low accuracy on CAPTCHA
  - Check that CAPTCHA screenshots look correct in `captchas/`
  - Verify the model file path and GPU/CPU availability logs

## Notes on performance

The ultra‑fast mode uses aggressive Chromium flags and tight timeouts. If your network or the VTU site is slow/unstable, consider the standard script (`latest_vtu_automation_2.py`) which uses more conservative settings.

## License

This project is licensed under the MIT License.
This project is for educational purpose only.

TRAINED & BUILT BY ROHIT.R and TEAM
