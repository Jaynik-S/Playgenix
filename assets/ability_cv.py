import cv2
import os
from collections import defaultdict
from PIL import Image
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
def match_icon(screenshot, icon_folder, bbox, visualize=False):
    x, y, w, h = bbox
    cropped = screenshot[y:y+h, x:x+w]
    
    if visualize:
        cv2.imshow("Screenshot", cropped)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()  # Close the window after key press

    best_score = float('-inf')
    best_match = None

    for icon_name in os.listdir(icon_folder):
        icon_path = os.path.join(icon_folder, icon_name)
        icon = cv2.imread(icon_path)

        if icon is None or icon.shape[:2] != cropped.shape[:2]:
            icon = cv2.resize(icon, (w, h))
    
        res = cv2.matchTemplate(cropped, icon, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
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
        json.dump(status, f, indent=4)
    print(f"Status saved to {output_path}")


def main(file_name: str, visualize: bool = False):
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
    timestamp_str = 0
    spike_planted = False

    frames = extract_frames(video_path, interval=1)

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    for frame in frames:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(0):
            continue
    cv2.destroyWindow("Frame")

###    
    spike_planted = spike_check(frames[0], spike_bbox, visualize)
    if spike_planted:
        timestamp_str = 0
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
        timestamp_str = format_init_time(initial_time, index) + 1
###

    # # After reading frames = extract_frames(...):
    # fps_cap = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    # fail_guard = 0
    # m, s, index = "0", "0", -1
    # while not (len(m) == 1 and len(s) == 2) and fail_guard < 15:
    #     index += 1; fail_guard += 1
    #     initial_time = time_capture(frames[index], clock_bbox, visualize)
    #     split = initial_time.split(':')
    #     if len(split) == 2: m, s = split
    # timestamp_str = format_init_time(initial_time, index) + 1 if fail_guard < 15 else int(len(frames))

'''
    print("STARTING FRAME PROCESSING")
    for i, frame in enumerate(frames):
        timestamp_str -= 1
        print(f"\n{timestamp_str}\n")

        if not spike_planted:
            spike_planted = spike_check(frame, spike_bbox, visualize)
            if spike_planted: 
                timestamp_str = -1 

        ##########################################
        
        if max(ability_matches.values(), default=0) < 3: 
            ability_icon = match_icon(frame, "assets/abilities", ability_bbox, visualize)
            print(f"Ability icon matched: {ability_icon}")
            ability_matches[ability_icon] += 1

        ##########################################

        for i in range(len(charge_bboxes)):
            x, y, w, h = charge_bboxes[i]
            cropped = frame[y:y+h, x:x+w]
            
            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            target_size = (128, 32)
            new_img = Image.new('RGB', target_size, (0, 0, 0))
            offset = ((target_size[0] - pil_img.width) // 2, (target_size[1] - pil_img.height) // 2)
            new_img.paste(pil_img, offset)
            resized_crop = cv2.cvtColor(np.array(new_img), cv2.COLOR_RGB2BGR)
            
            if visualize:
                cv2.imshow("Resized Crop", resized_crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            charge_icon = match_charge_cnn(resized_crop, None, (0, 0, target_size[0], target_size[1]))
            slot_matches[timestamp_str].append((f"Ability #{i+1}", charge_icon[0:3]))
            print(f"Ability icon matched: {charge_icon[0:3]}")
        
        ##########################################
        
        for i in range(len(team_status)):
            agent, ult_ready = detect_agent(frame, team_status[i], team_ultimate[i],  "assets/agents/normal", visualize)
            print(f"Agent detected: {agent}, Ultimate ready: {ult_ready}")
            team_status_dict[timestamp_str][agent] = bool(ult_ready)
            
            agent, ult_ready = detect_agent(frame, enemy_status[i], enemy_ultimate[i],  "assets/agents/flipped", visualize)
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
'''

if __name__ == "__main__":
    video_paths = ["720"]
    for video in video_paths:
        main(f"videos/v{video}.mp4", False)
    


'''
    #V10 (5K JETT)
    correct_answer =  {'0:00': ['3-3', '2-2', '0-1', '1-1'],
            75: ['3-3', '2-2', '0-1', '1-1'],
            74: ['2-3', '2-2', '0-1', '1-1'],
            73: ['2-3', '2-2', '0-1', '1-1'],
            72: ['2-3', '2-2', '0-1', '1-1'],
            71: ['2-3', '2-2', '0-1', '1-1'],
            70: ['2-3', '2-2', '1-1', '1-1'],
            69: ['2-3', '2-2', '1-1', '1-1'],
            68: ['2-3', '2-2', '1-1', '1-1'],
            67: ['2-3', '1-2', '1-1', '1-1'],
            66: ['2-3', '1-2', '1-1', '1-1'],
            65: ['2-3', '1-2', '1-1', '1-1'],
            64: ['2-3', '1-2', '1-1', '1-1'],
            63: ['2-3', '1-2', '0-1', '1-1'],
            62: ['2-3', '1-2', '0-1', '1-1'],
            61: ['2-3', '1-2', '0-1', '1-1'],
            60: ['2-3', '1-2', '0-1', '1-1'],
            59: ['2-3', '1-2', '0-1', '1-1'],
            58: ['2-3', '1-2', '0-1', '1-1'],
            57: ['2-3', '1-2', '0-1', '1-1']}
    
    print("\n=== ACCURACY COMPARISON ===")
    
    total_comparisons, correct_matches = 0, 0
    
    for timestamp, predicted_slots in slot_matches.items():
            expected_slots = correct_answer[timestamp]
            for predicted, expected in zip(predicted_slots, expected_slots):
                total_comparisons += 1
                if predicted[-1] == expected:
                    correct_matches += 1
                else:
                    print(f"❌ {timestamp} Expected '{expected}', Got '{predicted}'")
    
    # Calculate accuracy
    accuracy = (correct_matches / total_comparisons) * 100 if total_comparisons > 0 else 0
    
    print(f"\n=== ACCURACY RESULTS ===")
    print(f"Total comparisons: {total_comparisons}")
    print(f"Correct matches: {correct_matches}")
    print(f"Accuracy: {accuracy:.2f}%")
'''
