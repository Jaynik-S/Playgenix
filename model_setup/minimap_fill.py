"""
MAPCALLOUT SCRIPT
-----------------
Color-codes VALORANT minimap callout regions and writes a JSON mapping.

Deps:
  pip install opencv-python pillow numpy pytesseract
  # If Tesseract is not on PATH, install it and set TESSERACT_CMD below.

Set the paths below, then run:
  python minimap_fill.py
"""

import json
import os
import random

import cv2
import numpy as np
from PIL import Image

try:
    import pytesseract
    TESS_AVAILABLE = True
    # If needed, point pytesseract to your tesseract binary:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    TESS_AVAILABLE = False


# ---------- Utilities ----------
def to_hex_bgr(color_bgr):
    b, g, r = [int(c) for c in color_bgr]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def distinct_palette(n):
    """
    Returns n distinct-ish BGR colors (no pure black/white).
    Uses HLS sampling for good separation.
    """
    colors = []
    for i in range(n):
        h = int(180 * i / max(1, n))  # OpenCV HSV hue range
        s = 200
        v = 230
        col = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        colors.append(col)
    random.shuffle(colors)
    return colors

def safe_dilate(mask, k=3, it=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(mask, kernel, iterations=it)

def safe_erode(mask, k=3, it=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.erode(mask, kernel, iterations=it)

def overlay_alpha(base_bgr, overlay_bgr, alpha):
    return (overlay_bgr * alpha + base_bgr * (1 - alpha)).astype(np.uint8)

# ---------- Core steps ----------
def find_label_contours(img_bgr):
    """
    Find callout label blobs by detecting white text and fusing characters into a single contour.
    Returns a list of cv2 bounding rects (x, y, w, h) for labels.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # White-ish text mask (labels are white text)
    # Tune thresholds if needed for other styles.
    white_mask = cv2.inRange(gray, 200, 255)

    # Remove small specks, fuse letters into boxes
    white_mask = safe_dilate(white_mask, 3, 2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5)), iterations=1)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    H, W = gray.shape
    area_img = H * W

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # Heuristics to keep only label-sized boxes
        if area_img * 0.00015 < area < area_img * 0.05 and 1.5 < (w / max(1, h)) < 10:
            rects.append((x, y, w, h))

    # Optional: merge overlapping rects (two-line labels)
    merged = True
    while merged:
        merged = False
        new_rects = []
        used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]:
                continue
            (x1, y1, w1, h1) = rects[i]
            ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
            merged_any = False
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                (x2, y2, w2, h2) = rects[j]
                bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
                # If boxes overlap/are very close, merge
                pad = 6
                if not (ax2 + pad < bx1 or bx2 + pad < ax1 or ay2 + pad < by1 or by2 + pad < ay1):
                    nx1, ny1 = min(ax1, bx1), min(ay1, by1)
                    nx2, ny2 = max(ax2, bx2), max(ay2, by2)
                    x1, y1, w1, h1 = nx1, ny1, nx2 - nx1, ny2 - ny1
                    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
                    used[j] = True
                    merged = True
                    merged_any = True
            used[i] = True
            new_rects.append((x1, y1, w1, h1))
        rects = new_rects

    return rects

def ocr_label(img_bgr, rect):
    """
    OCR the label text inside rect. Falls back to 'Region_<idx>' if OCR unavailable/empty.
    """
    (x, y, w, h) = rect
    crop = img_bgr[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2]
    if not TESS_AVAILABLE:
        return None
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_gray = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -l eng"
    text = pytesseract.image_to_string(crop_gray, config=config)
    text = text.strip().replace("\n", " ")
    # Minor cleanup of common OCR artifacts
    text = text.replace("|", "I").replace("—", "-")
    return text if text else None

def build_playable_mask(img_bgr):
    """
    Create a binary mask of traversable space (everything except black walls/background).
    In most reference maps, walls/background are near black; corridors/sites are grayish.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 'Blocked' where very dark; playable otherwise
    blocked = cv2.inRange(gray, 0, 25)
    playable = cv2.bitwise_not(blocked)
    # Clean thin seams
    playable = cv2.morphologyEx(playable, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return playable

def flood_region_from_seed(playable_mask, seed_xy):
    """
    Flood-fill the connected component reachable from seed within playable area.
    Returns boolean mask of that component.
    """
    H, W = playable_mask.shape
    visited = np.zeros((H, W), np.uint8)
    q = [seed_xy]
    visited[seed_xy[1], seed_xy[0]] = 1
    while q:
        x, y = q.pop()
        for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            if 0 <= nx < W and 0 <= ny < H:
                if playable_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = 1
                    q.append((nx, ny))
    return visited.astype(bool)

def colorize_minimap(img_bgr, alpha=0.35, prefer_ocr=True, map_name=None):
    """
    Full pipeline: find labels -> flood-fill regions -> colorize -> collect mapping.
    Returns (colored_bgr, mapping_dict)
    """
    H, W = img_bgr.shape[:2]
    playable_mask = build_playable_mask(img_bgr)
    rects = find_label_contours(img_bgr)

    # Prepare colors
    colors = distinct_palette(max(1, len(rects)))
    color_iter = iter(colors)

    # Keep track of already-colored pixels to avoid overlap
    painted = np.zeros((H, W), np.uint8)
    overlay = img_bgr.copy()

    mapping = {}            # name -> HEX
    unnamed_counter = 1

    for idx, rect in enumerate(sorted(rects, key=lambda r: (r[1], r[0]))):
        x, y, w, h = rect
        seed = (x + w // 2, y + h // 2)

        # Expand outwards a bit to ensure seed lands on playable pixels (not on label box)
        sx, sy = seed
        for r_ in range(0, max(W, H)//300 + 4):  # small search radius
            found = False
            for dx in range(-r_, r_+1):
                for dy in range(-r_, r_+1):
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < W and 0 <= ny < H and playable_mask[ny, nx]:
                        seed = (nx, ny)
                        found = True
                        break
                if found: break
            if found: break

        component = flood_region_from_seed(playable_mask & (painted == 0), seed)
        if component.sum() < 250:  # discard tiny blobs
            continue

        color = next(color_iter, [int(x) for x in np.random.randint(50, 230, size=3)])
        hex_code = to_hex_bgr(color)

        # Region name via OCR (optional)
        name = None
        if prefer_ocr:
            name = ocr_label(img_bgr, rect)
        if not name:
            prefix = f"{map_name}_" if map_name else ""
            name = f"{prefix}Region_{unnamed_counter}"
            unnamed_counter += 1

        # Paint overlay & mark as used
        mask3 = np.dstack([component]*3).astype(np.uint8)
        region_img = np.zeros_like(img_bgr)
        region_img[mask3 == 1] = color
        overlay = overlay_alpha(overlay, region_img, alpha)
        painted[component] = 1

        # Record mapping
        mapping[name] = hex_code

    return overlay, mapping


# ---------- CLI ----------
INPUT_DIR = "assets/minimaps_callouts"
OUTPUT_IMG_DIR = "assets/minimaps_coded"
OUTPUT_JSON_DIR = "assets/json_data/map_coding"
ALPHA = 0.35
USE_OCR = True
VALID_EXTS = (".png", ".jpg", ".jpeg")


def main():
    if not INPUT_DIR:
        raise ValueError("Set INPUT_DIR before running the script.")

    input_dir = os.path.abspath(INPUT_DIR)
    out_img_dir = os.path.abspath(OUTPUT_IMG_DIR)
    out_json_dir = os.path.abspath(OUTPUT_JSON_DIR)

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_json_dir, exist_ok=True)

    valid_exts = tuple(ext.lower() for ext in VALID_EXTS)
    tess_warned = False

    for entry in sorted(os.listdir(input_dir)):
        src_path = os.path.join(input_dir, entry)
        if not os.path.isfile(src_path):
            continue
        if not entry.lower().endswith(valid_exts):
            continue

        map_name, src_ext = os.path.splitext(entry)
        print(f"[INFO] Processing {entry} -> map '{map_name}'")

        img_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Skipping {entry}: unable to read image.")
            continue

        colored, mapping = colorize_minimap(
            img_bgr,
            alpha=ALPHA,
            prefer_ocr=USE_OCR,
            map_name=map_name
        )

        out_image_path = os.path.join(out_img_dir, f"{map_name}{src_ext}")
        out_json_path = os.path.join(out_json_dir, f"{map_name}.json")

        cv2.imwrite(out_image_path, colored)
        payload = {map_name: mapping}
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved colored minimap -> {out_image_path}")
        print(f"[OK] Saved color mapping   -> {out_json_path}")
        if not TESS_AVAILABLE and USE_OCR and not tess_warned:
            print("[Note] pytesseract not found; labels were named Region_#, "
                  "re-run with Tesseract installed for real names.")
            tess_warned = True


if __name__ == "__main__":
    main()
