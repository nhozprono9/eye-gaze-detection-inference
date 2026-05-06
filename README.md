# eye-gaze-detection-inference

Realtime eye-gaze detection from webcam using a trained YOLO model (`best.pt`) and simple pupil-center direction estimation.

## Features

- Webcam inference with YOLO (`ultralytics`)
- Gaze direction estimation (`Left`, `Right`, `Up`, `Center`)
- Warning and cheating timer overlays
- Optional recording to MP4 output

## Project Structure

- `main.py`: inference entry script
- `requirements.txt`: runtime dependencies
- `models/best.pt`: model file (download manually, not committed)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py --model models/best.pt --camera 0
```

## Model Setup 

This repository does **not** include `best.pt` to keep it lightweight and avoid publishing binary weights.

1. Create/keep the `models/` folder (already included with `.gitkeep`).
2. Download your trained weight file from:

   [Google Drive - best.pt](https://drive.google.com/file/d/1PgRexpTtoscmkN3CmQKGuvuD_QqH8kpw/view?usp=drive_link)
3. Rename/copy it to:

```bash
models/best.pt
```

If your model file is stored elsewhere, pass a custom path:

```bash
python main.py --model /path/to/your/best.pt --camera 0
```

Optional flags:

```bash
python main.py \
  --model models/best.pt \
  --camera 0 \
  --conf 0.4 \
  --warning-sec 1.0 \
  --cheating-sec 2.0 \
  --save output.mp4
```

Press `q` to quit.

## Publish Notes

- This repo is inference-focused, so training data and training runs are ignored by `.gitignore`.
- Do not commit secrets (API keys, tokens).
- Model weights are intentionally excluded. Share them via cloud link or GitHub Release assets.
