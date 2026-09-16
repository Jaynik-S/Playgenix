"""Ability charge and icon detection helpers."""
import cv2
import os
import numpy as np


def _resize_with_interp(image, size):
    target_w, target_h = size
    if target_w <= 0 or target_h <= 0:
        return image
    h, w = image.shape[:2]
    if target_w < w or target_h < h:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    return cv2.resize(image, (target_w, target_h), interpolation=interp)


def _normalize_icon_bgr(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _edge_map(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blurred, 50, 150)


def _multi_scale_match(image, template, scales):
    best = float("-inf")
    image_h, image_w = image.shape[:2]
    template_h, template_w = template.shape[:2]

    for scale in scales:
        scaled_w = int(round(template_w * scale))
        scaled_h = int(round(template_h * scale))
        if scaled_w < 2 or scaled_h < 2:
            continue
        if scaled_w > image_w or scaled_h > image_h:
            continue
        resized = _resize_with_interp(template, (scaled_w, scaled_h))
        res = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        if score > best:
            best = score

    return best


def _dump_debug_image(debug_dir, name, image):
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, name), image)


def preprocess_charge_crop(crop, target_size=(128, 32), debug_dump_dir=None, debug_tag=None):
    target_w, target_h = target_size
    if crop.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.fastNlMeansDenoising(l_channel, None, 7, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_channel = clahe.apply(l_channel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    l_channel = cv2.morphologyEx(l_channel, cv2.MORPH_OPEN, kernel)
    l_channel = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, kernel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    cleaned = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    crop_h, crop_w = cleaned.shape[:2]
    scale = min(target_w / max(crop_w, 1), target_h / max(crop_h, 1))
    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))
    resized = _resize_with_interp(cleaned, (resized_w, resized_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    canvas[offset_y:offset_y + resized_h, offset_x:offset_x + resized_w] = resized

    if debug_dump_dir:
        tag = debug_tag or "charge"
        charge_dir = os.path.join(debug_dump_dir, "charges")
        _dump_debug_image(charge_dir, f"charge_raw_{tag}.png", crop)
        _dump_debug_image(charge_dir, f"charge_pre_{tag}.png", canvas)

    return canvas


def match_icon(
    screenshot,
    icon_folder,
    bbox,
    visualize=False,
    use_edges=False,
    scales=None,
    debug_dump_dir=None,
    debug_tag=None,
):
    x, y, w, h = bbox
    cropped = screenshot[y:y+h, x:x+w]
    if cropped.size == 0:
        return None

    if visualize:
        cv2.imshow("Screenshot", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if scales is None:
        scales = (0.85, 0.95, 1.0, 1.1)

    cropped_norm = _normalize_icon_bgr(cropped)
    cropped_edges = _edge_map(cropped_norm) if use_edges else None

    if debug_dump_dir:
        tag = debug_tag or "icon"
        icon_dir = os.path.join(debug_dump_dir, "icons")
        _dump_debug_image(icon_dir, f"icon_raw_{tag}.png", cropped)
        _dump_debug_image(icon_dir, f"icon_norm_{tag}.png", cropped_norm)
        if use_edges:
            _dump_debug_image(icon_dir, f"icon_edge_{tag}.png", cropped_edges)

    best_score = float("-inf")
    best_match = None

    for icon_name in os.listdir(icon_folder):
        icon_path = os.path.join(icon_folder, icon_name)
        icon = cv2.imread(icon_path)
        if icon is None:
            continue

        icon_h, icon_w = icon.shape[:2]
        scale_fit = min(w / max(icon_w, 1), h / max(icon_h, 1))
        scale_factors = [scale_fit * scale for scale in scales]

        icon_norm = _normalize_icon_bgr(icon)
        score = max(
            _multi_scale_match(cropped_norm, icon_norm, scale_factors),
            _multi_scale_match(cropped, icon, scale_factors),
        )
        if use_edges:
            icon_edges = _edge_map(icon_norm)
            score = max(score, _multi_scale_match(cropped_edges, icon_edges, scale_factors))

        if score > best_score:
            best_score = score
            best_match = icon_name

    return best_match
