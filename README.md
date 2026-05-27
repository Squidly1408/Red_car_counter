![Logo](<https://raw.githubusercontent.com/Squidly1408/Squidly1408/refs/heads/main/images/Squidly1408%20banner%20(Black%20Button%20Background).png>)

# Red Car Counter (Live Webcam Mosaic)

This project loads still images from traffic webcam URLs, combines them into a mosaic, runs YOLO object detection, and counts vehicles that appear red based on HSV color filtering inside each detected bounding box.

The app opens a live OpenCV window and updates every few seconds.

## Files

- `red_car_live.py` - Main script
- `urls.txt` - Webcam image URLs (one per line)
- `yolov8n.pt` - YOLO model weights

## Requirements

- Python 3.9+
- pip
- Internet access (for webcam URLs)

Python packages:

- ultralytics
- opencv-python
- numpy
- requests

Install packages:

```bash
pip install ultralytics opencv-python numpy requests
```

## Quick Start

From this folder, run:

```bash
python red_car_live.py
```

Then press `ESC` in the OpenCV window to quit.

## URL File Format

`urls.txt` should contain one image URL per line. Empty lines and lines starting with `#` are ignored.

Example:

```txt
https://example.com/cam1.jpg
https://example.com/cam2.jpg
# https://example.com/disabled.jpg
```

## Common Usage Examples

Default run:

```bash
python red_car_live.py
```

Faster refresh and larger model:

```bash
python red_car_live.py --refresh 2 --model yolov8s.pt
```

Count multiple vehicle classes:

```bash
python red_car_live.py --vehicles car,truck,bus,motorcycle
```

Custom tile size and fixed columns:

```bash
python red_car_live.py --tile-size 640x360 --cols 3
```

Tune red sensitivity:

```bash
python red_car_live.py --min-red-ratio 0.12
```

## CLI Options

- `--urls-file` (default: `urls.txt`)  
  Text file with one URL per line.
- `--model` (default: `yolov8n.pt`)  
  YOLO weights file (for example `yolov8n.pt`, `yolov8s.pt`).
- `--refresh` (default: `5.0`)  
  Seconds between refresh cycles.
- `--tile-size` (default: `960x540`)  
  Mosaic tile size as `WIDTHxHEIGHT`.
- `--cols` (default: `0`)  
  Number of mosaic columns (`0` = auto).
- `--conf` (default: `0.35`)  
  YOLO confidence threshold.
- `--iou` (default: `0.45`)  
  YOLO IoU threshold.
- `--min-red-ratio` (default: `0.08`)  
  Minimum red-pixel ratio in a detected vehicle box to classify it as red.
- `--vehicles` (default: `car`)  
  Comma-separated YOLO class names to treat as vehicles.
- `--per-cycle-unique`  
  Enable simple near-duplicate suppression per cycle.

## Notes

- The script counts detections per refresh cycle and keeps a running total.
- Red classification is HSV-based and may need tuning depending on lighting/weather.
- If a webcam URL fails or returns non-image content, it is skipped for that cycle.

## Troubleshooting

- If `python` is not found, use your full Python path or launcher (`py`).
- If model loading fails, confirm `yolov8n.pt` exists in this folder or pass `--model` with a valid path.
- If nothing appears, verify URLs in `urls.txt` return valid images in a browser.
