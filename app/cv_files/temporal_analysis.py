"""Temporal helpers for frame extraction and timestamp logic."""
import cv2
import os
import pytesseract


def extract_frames(video_path, interval=1):
    """Extract frames from a video at specified time intervals (in seconds)."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    video.release()
    return frames


def time_capture(screenshot, bbox, visualize=False):
    x, y, w, h = bbox
    cropped = screenshot[y:y+h, x:x+w]

    if visualize:
        cv2.imshow("Screenshot", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:'
    text = pytesseract.image_to_string(clean, config=custom_config)
    text = text.strip()

    return text


def format_init_time(initial_time, inc):
    minutes, seconds = initial_time.split(':')
    minutes = int(minutes)
    seconds = int(seconds)
    seconds += minutes * 60 + inc
    return seconds


def initialize_timestamp(visualize, clock_bbox, frames, spike_planted):
    if spike_planted:
        return 0
    else:
        m, s, index = "0", "0", -1
        while not (len(m) == 1 and len(s) == 2):
            if index + 1 < len(frames):
                index += 1
            else:
                initial_time = "0:0"
                break
            initial_time = time_capture(frames[index], clock_bbox, visualize)
            split = initial_time.split(':')
            if len(split) != 2:
                continue
            m, s = split
        return format_init_time(initial_time, index) + 1


def spike_check(screenshot, bbox, visualize=False):
    x, y, w, h = bbox
    cropped = screenshot[y:y+h, x:x+w]

    if visualize:
        cv2.imshow("Screenshot", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    icon = cv2.imread("assets/Spike.png")

    h, w = cropped.shape[:2]
    if icon.shape[:2] != (h, w):
        icon = cv2.resize(icon, (w, h))

    res = cv2.matchTemplate(cropped, icon, cv2.TM_CCOEFF_NORMED)
    _, score, _, _ = cv2.minMaxLoc(res)

    return score >= 0.2
