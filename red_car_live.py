import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO


# ----------------------------
# URL loading
# ----------------------------


def load_urls_from_file(path: str) -> List[str]:
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # strip trailing ? or &
            while line.endswith("?") or line.endswith("&"):
                line = line[:-1]
            urls.append(line)
    return urls


# ----------------------------
# Downloading (robust for webcam endpoints)
# ----------------------------


def download_image(
    url: str, timeout: int = 20, max_retries: int = 2, backoff: float = 0.5
) -> Optional[np.ndarray]:
    """
    Download an image from a URL and return as OpenCV BGR ndarray.
    Adds cache-busting and browser-ish headers for webcam endpoints.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "image/jpeg,image/*,*/*",
        "Referer": "https://www.livetraffic.com/",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            # cache busting
            ts = int(time.time() * 1000)
            sep = "&" if "?" in url else "?"
            url_cb = f"{url}{sep}t={ts}"

            r = requests.get(url_cb, timeout=timeout, headers=headers)
            r.raise_for_status()

            data = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)

            if img is None or img.size == 0:
                raise ValueError("Decode failed (not an image or empty content).")

            return img

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff * attempt)

    print(f"[WARN] Failed to download/decode: {url}\n       {last_err}")
    return None


# ----------------------------
# Mosaic building
# ----------------------------


def make_mosaic(
    images: List[np.ndarray], tile_size: Tuple[int, int], cols: int
) -> np.ndarray:
    """
    Resize images into tiles and pack into a grid.
    """
    tile_w, tile_h = tile_size
    cols = max(1, cols)
    rows = math.ceil(len(images) / cols)

    mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols

        resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        y0, y1 = r * tile_h, (r + 1) * tile_h
        x0, x1 = c * tile_w, (c + 1) * tile_w
        mosaic[y0:y1, x0:x1] = resized

    return mosaic


def auto_cols(n: int) -> int:
    if n <= 1:
        return 1
    return max(1, int(math.ceil(math.sqrt(n))))


# ----------------------------
# Red detection inside bbox
# ----------------------------


@dataclass
class RedThresholds:
    # HSV ranges for red in OpenCV HSV: H=[0..179]
    low1: Tuple[int, int, int] = (0, 80, 60)
    high1: Tuple[int, int, int] = (10, 255, 255)
    low2: Tuple[int, int, int] = (170, 80, 60)
    high2: Tuple[int, int, int] = (179, 255, 255)


def red_ratio_in_roi(bgr_roi: np.ndarray, thresh: RedThresholds) -> float:
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    m1 = cv2.inRange(hsv, np.array(thresh.low1), np.array(thresh.high1))
    m2 = cv2.inRange(hsv, np.array(thresh.low2), np.array(thresh.high2))
    mask = cv2.bitwise_or(m1, m2)

    # reduce speckle noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    red_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    return red_pixels / max(1, total_pixels)


def is_red_vehicle(
    roi: np.ndarray, min_red_ratio: float, thresh: RedThresholds
) -> Tuple[bool, float]:
    rr = red_ratio_in_roi(roi, thresh)
    return (rr >= min_red_ratio), rr


# ----------------------------
# Live loop
# ----------------------------


def live_loop(
    urls,
    model,
    vehicle_names,
    refresh_seconds,
    tile_size,
    cols,
    conf,
    iou,
    min_red_ratio,
    per_cycle_unique=False,  # add this
):

    red_thresh = RedThresholds()
    total_red = 0

    window = "LIVE Red Vehicle Counter (ESC to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Start large but user can resize
    cv2.resizeWindow(window, 1400, 900)

    print("[INFO] Live loop started. Press ESC to quit.")

    while True:
        cycle_start = time.time()
        frames = []

        # Download images
        for u in urls:
            img = download_image(u)
            if img is not None:
                frames.append(img)

        if not frames:
            time.sleep(refresh_seconds)
            continue

        use_cols = cols if cols > 0 else auto_cols(len(frames))

        # BIGGER TILES
        mosaic = make_mosaic(frames, tile_size, use_cols)
        display = mosaic.copy()
        h, w = mosaic.shape[:2]

        # YOLO detect
        rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb, conf=conf, iou=iou, verbose=False)

        cycle_red = 0

        if results and results[0].boxes is not None:
            r0 = results[0]
            names = r0.names

            for box in r0.boxes:
                cls_id = int(box.cls.item())
                cls_name = names.get(cls_id, str(cls_id))

                if cls_name not in vehicle_names:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                roi = mosaic[y1:y2, x1:x2]
                ok, rr = is_red_vehicle(roi, min_red_ratio, red_thresh)

                color = (0, 255, 0) if ok else (0, 165, 255)

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display,
                    f"{cls_name} {rr:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                if ok:
                    cycle_red += 1

        total_red += cycle_red

        # Counters
        cv2.putText(
            display,
            f"Cycle Red: {cycle_red}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

        cv2.putText(
            display,
            f"Total Red: {total_red}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3,
        )

        # ---- AUTO SCALE TO WINDOW ----
        win_w = cv2.getWindowImageRect(window)[2]
        win_h = cv2.getWindowImageRect(window)[3]

        if win_w > 0 and win_h > 0:
            scale = min(win_w / w, win_h / h)
            new_size = (int(w * scale), int(h * scale))
            display = cv2.resize(display, new_size)

        cv2.imshow(window, display)

        if cv2.waitKey(1) == 27:
            break

        sleep_left = refresh_seconds - (time.time() - cycle_start)
        if sleep_left > 0:
            time.sleep(sleep_left)

    cv2.destroyAllWindows()


# ----------------------------
# Main
# ----------------------------


def parse_wh(s: str) -> Tuple[int, int]:
    try:
        a, b = s.lower().split("x")
        return int(a), int(b)
    except Exception:
        raise SystemExit("tile-size must be like 640x360")


def main():
    ap = argparse.ArgumentParser(
        description="Live red car counter from image URLs (webcams). Press ESC to quit."
    )
    ap.add_argument(
        "--urls-file", default="urls.txt", help="Text file with one URL per line."
    )
    ap.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO weights (e.g. yolov8n.pt, yolov8s.pt).",
    )
    ap.add_argument(
        "--refresh", type=float, default=5.0, help="Seconds between refresh cycles."
    )
    ap.add_argument(
        "--tile-size",
        default="960x540",
    )
    ap.add_argument("--cols", type=int, default=0, help="Mosaic columns (0 = auto).")
    ap.add_argument(
        "--conf", type=float, default=0.35, help="YOLO confidence threshold."
    )
    ap.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    ap.add_argument(
        "--min-red-ratio",
        type=float,
        default=0.08,
        help="Min red pixel ratio in bbox to count as red.",
    )
    ap.add_argument(
        "--vehicles",
        default="car",
        help="Comma-separated class names to treat as vehicles.",
    )
    ap.add_argument(
        "--per-cycle-unique",
        action="store_true",
        help="Try to avoid counting near-duplicate overlapping boxes within a cycle (simple IoU filter).",
    )
    args = ap.parse_args()

    urls = load_urls_from_file(args.urls_file)
    if not urls:
        raise SystemExit(f"No URLs found in {args.urls_file}")

    vehicle_names = tuple(x.strip() for x in args.vehicles.split(",") if x.strip())
    tile_size = parse_wh(args.tile_size)

    print(f"[INFO] URLs loaded: {len(urls)}")
    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    live_loop(
        urls=urls,
        model=model,
        vehicle_names=vehicle_names,
        refresh_seconds=args.refresh,
        tile_size=tile_size,
        cols=args.cols,
        conf=args.conf,
        iou=args.iou,
        min_red_ratio=args.min_red_ratio,
        per_cycle_unique=args.per_cycle_unique,
    )


if __name__ == "__main__":
    main()
