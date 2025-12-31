import cv2
import os
from collections import defaultdict
import pytesseract
from img_preprocess import match_charge_cnn
import numpy as np
import json

# GLOBAL
BASE_W, BASE_H = 1920, 1080
BASE_BOXES = {
    "ability_bbox": (759, 974, 60, 60),
    "charge_bboxes": [(740, 1036, 100, 23), (852, 1036, 100, 23), (965, 1036, 100, 23), (1078, 1036, 100, 23)],
    "clock_bbox": (930, 27, 65, 44),
    "spike_bbox": (918, 11, 85, 85),
    "team_status": [(444, 27, 45, 44), (509, 27, 45, 44), (573, 27, 45, 44), (641, 27, 45, 44), (708, 27, 45, 44)],
    "team_ultimate": [(444, 16, 45, 14), (509, 16, 45, 14), (573, 16, 45, 14), (641, 16, 45, 14), (708, 16, 45, 14)],
    "enemy_status": [(1169, 27, 45, 44), (1235, 27, 45, 44), (1301, 27, 45, 44), (1368, 27, 45, 44), (1434, 27, 45, 44)],
    "enemy_ultimate": [(1169, 16, 45, 14), (1235, 16, 45, 14), (1301, 16, 45, 14), (1368, 16, 45, 14), (1434, 16, 45, 14)],
}


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


#############################
# ICON
#############################
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
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()  # Close the window after key press

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

    best_score = float('-inf')
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
        #print(f"{icon_path}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_match = icon_name

    return best_match

#############################
# AGENT DETECTION
#############################
def detect_agent(screenshot, bbox, ult_bbox, agent_folder, visualize=False, threshold=0.15):
    """Detect which agent (if any) occupies a slot and whether their ultimate is ready."""
    x, y, w, h = bbox
    cropped = screenshot[y:y+h, x:x+w]
    
    if visualize:
        cv2.imshow("Screenshot", cropped)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
    best_score = float('-inf')
    best_match = None

    # Compare against each known agent icon
    for agent_name in os.listdir(agent_folder):
        agent_path = os.path.join(agent_folder, agent_name)
        icon = cv2.imread(agent_path)

        # Resize icon to match slot dimensions
        h, w = cropped.shape[:2]
        if icon.shape[:2] != (h, w):
            icon = cv2.resize(icon, (w, h))

        # Template‐matching score
        res = cv2.matchTemplate(cropped, icon, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        
        #print(f"{agent_path}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_match = agent_name

    print(f"=====Best match: {best_match} with score {best_score:.4f}=====")
    
    if is_blank_slot_edges(cropped):
        print("=====Slot is blank=====")
        return None, None
    
    agent = os.path.splitext(best_match)[0]
    ult_ready = detect_ultimate(screenshot, ult_bbox, visualize)

    return agent, ult_ready


def is_blank_slot_edges(cropped_image, edge_threshold=0.05):
    """Check if slot is blank using edge detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Calculate edge density
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_pixels = cv2.countNonZero(edges)
    edge_ratio = edge_pixels / total_pixels
    
    # If very few edges, likely a solid color
    return edge_ratio < edge_threshold


def detect_ultimate(screenshot, bbox, visualize=False, threshold=0.1):
    x, y, w, h = bbox
    cropped = screenshot[y:y+h, x:x+w]
    
    if visualize:
        cv2.imshow("Ultimate Screenshot", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Convert BGR to HSV for better color detection
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([25,  40, 140])   # [25,  10, 70]
    upper_yellow = np.array([100, 120, 220])    # [120, 120, 210]

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    yellow_pixels = cv2.countNonZero(yellow_mask)
    total_pixels = cropped.shape[0] * cropped.shape[1]
    yellow_ratio = yellow_pixels / float(total_pixels)

    if visualize:
        cv2.imshow("Cropped Ultimate Slot", cropped)
        cv2.imshow("Yellow Mask", yellow_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"{yellow_ratio > threshold}: Ratio: {yellow_ratio:.2f}")

    return yellow_ratio > threshold

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

#############################
# HELPER FUNCTIONS
#############################

def _scale_box(box, sx, sy):
    x, y, w, h = box
    return (int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy)))

def scale_boxes(frame_w, frame_h):
    sx, sy = frame_w / BASE_W, frame_h / BASE_H
    out = {}
    for k, v in BASE_BOXES.items():
        if isinstance(v, list):
            out[k] = [_scale_box(b, sx, sy) for b in v]
        else:
            out[k] = _scale_box(v, sx, sy)
    return out

#############################
# TIMESTAMPING
#############################

def time_capture(screenshot, bbox, visualize=False):
    x, y, w, h = bbox 
    cropped = screenshot[y:y+h, x:x+w]
    
    if visualize:
        cv2.imshow("Screenshot", cropped)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
    
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:'
    text = pytesseract.image_to_string(clean, config=custom_config)
    text = text.strip()
    
    # text = pytesseract.image_to_string(cropped)
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
            if len(split) != 2: continue
            m, s = split
        return format_init_time(initial_time, index) + 1


#############################
# FORMATTING OUTPUT
#############################

def formatted_agents(data):
    for key, values in data.items():
        print(key)
        # for value in values:
        #     print("  " + ": ".join(map(str, value)))

def formatted_ability(data):
    for key, values in data.items():
        print(key)
        for value in values:
            #print("  " + value[0] + ": " + value[1])
            print(value[1])

def show_video_frames(frames):
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    for frame in frames:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(0):
            continue
    cv2.destroyWindow("Frame")

def save_to_json(ability, status, slot_matches, file_name):
    agent_path = f"assets/json_data/ability_to_agent.json"
    with open(agent_path, "r") as f:
        data = json.load(f)
    agent = data[ability]
    
    ability_path = f"assets/json_data/agent_to_ability.json"
    with open(ability_path, "r") as f:
        data = json.load(f)
    agent_abilities = data[agent]

    converted = {}
    for timestamp, entries in slot_matches.items():
        converted[timestamp] = {agent_abilities[slot_key]: charge for slot_key, charge in entries}

    session_dir = os.path.join("app", "session_data")
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    print(f"\n\n\nBASENAME:, {base_name}\n\n\n")
    
    output_path = os.path.join(session_dir, f"{base_name}_slot_matches.json")
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=4)
    print(f"Slot matches saved to {output_path}")

    output_path = os.path.join(session_dir, f"{base_name}_game_status.json")
    with open(output_path, "w") as f:
        json.dump(_clean_null_agents(status), f, indent=4)
    print(f"Status saved to {output_path}")

def _clean_null_agents(status):
    cleaned = {}
    for timestamp, snapshot in status.items():
        team = snapshot.get("team", {})
        enemy = snapshot.get("enemy", {})
        cleaned[timestamp] = {
            "team": {k: v for k, v in team.items() if k not in (None, "null")},
            "enemy": {k: v for k, v in enemy.items() if k not in (None, "null")},
        }
    return cleaned

#############################
# MAIN METHOD
#############################

def main(file_name: str, visualize: bool = False, debug_dump_dir=None, use_edge_matching=True):
    #video_path = f"app/static/uploads/{file_name}"
    video_path = file_name
    
    cap = cv2.VideoCapture(video_path) 
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, screenshot = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read frame from video")
        exit()
    
    print(f"\n\nVideo resolution: {frame_width}x{frame_height}")
    boxes = scale_boxes(frame_width, frame_height)
    ability_bbox   = boxes["ability_bbox"]
    charge_bboxes  = boxes["charge_bboxes"]
    clock_bbox     = boxes["clock_bbox"]
    spike_bbox     = boxes["spike_bbox"]
    team_status    = boxes["team_status"]
    team_ultimate  = boxes["team_ultimate"]
    enemy_status   = boxes["enemy_status"]
    enemy_ultimate = boxes["enemy_ultimate"]
    
    ability_matches = defaultdict(int)
    slot_matches = defaultdict(list)
    team_status_dict = defaultdict(dict)
    enemy_status_dict = defaultdict(dict)

    frames = extract_frames(video_path, interval=1)

    if visualize:
        show_video_frames(frames)

###    
    spike_planted = spike_check(frames[0], spike_bbox, visualize)
    timestamp_str = initialize_timestamp(visualize, clock_bbox, frames, spike_planted)
    print("STARTING FRAME PROCESSING")
    for frame_idx, frame in enumerate(frames):
        timestamp_str -= 1
        print(f"\n{timestamp_str}\n")

        if not spike_planted:
            spike_planted = spike_check(frame, spike_bbox, visualize)
            if spike_planted: 
                timestamp_str = -1 

        ##########################################
        
        if max(ability_matches.values(), default=0) < 3:
            ability_icon = match_icon(
                frame,
                "assets/abilities",
                ability_bbox,
                visualize,
                use_edges=use_edge_matching,
                debug_dump_dir=debug_dump_dir,
                debug_tag=f"{timestamp_str}_f{frame_idx}",
            )
            print(f"Ability icon matched: {ability_icon}")
            ability_matches[ability_icon] += 1

        ##########################################

        for slot_idx in range(len(charge_bboxes)):
            x, y, w, h = charge_bboxes[slot_idx]
            cropped = frame[y:y+h, x:x+w]

            target_size = (128, 32)
            resized_crop = preprocess_charge_crop(
                cropped,
                target_size=target_size,
                debug_dump_dir=debug_dump_dir,
                debug_tag=f"{timestamp_str}_f{frame_idx}_slot{slot_idx + 1}",
            )
            
            if visualize:
                cv2.imshow("Resized Crop", resized_crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            charge_icon = match_charge_cnn(resized_crop, None, (0, 0, target_size[0], target_size[1]))
            slot_matches[timestamp_str].append((f"Ability #{slot_idx + 1}", charge_icon[0:3]))
            print(f"Ability icon matched: {charge_icon[0:3]}")
        
        ##########################################
        
        for slot_idx in range(len(team_status)):
            agent, ult_ready = detect_agent(frame, team_status[slot_idx], team_ultimate[slot_idx],  "assets/agents/normal", visualize)
            print(f"Agent detected: {agent}, Ultimate ready: {ult_ready}")
            team_status_dict[timestamp_str][agent] = bool(ult_ready)
            
            agent, ult_ready = detect_agent(frame, enemy_status[slot_idx], enemy_ultimate[slot_idx],  "assets/agents/flipped", visualize)
            print(f"Agent detected: {agent}, Ultimate ready: {ult_ready}")
            enemy_status_dict[timestamp_str][agent] = bool(ult_ready)

    #######
    # Combine status
    #######    
    combined_status = {}
    for timestamp in team_status_dict.keys():
        combined_status[timestamp] = {
            'team': team_status_dict[timestamp],
            'enemy': enemy_status_dict[timestamp]
        }
    max_key = max(ability_matches, key=ability_matches.get)
    
    #######
    # Outputs
    #######
    print(f'Ability match: {max_key[:len(max_key)-4]}')
    print("=== COMBINED STATUS ===")
    for timestamp, status in combined_status.items():
        print(f"{timestamp}:")
        print(f"  Team : {status['team']}\n  Enemy: {status['enemy']}")
    
    print("=== TEAM STATUS ===")
    formatted_agents(team_status_dict)
    print("=== ENEMY STATUS ===")
    formatted_agents(enemy_status_dict)
    
    print("=== SLOT MATCHES ===")
    formatted_ability(slot_matches)
     
    print("=== JSON SAVING ===")
    save_to_json(max_key[:len(max_key)-4], combined_status, slot_matches, file_name)


if __name__ == "__main__":
    video_paths = ["v720", "v720-2", "v1080", "v1440", "v1080-2", "v1440-2", "v1080-3", "v1440-3"]
    #video_paths = ["1440"]
    for video in video_paths:
        main(f"videos/{video}.mp4", False)
    
