import cv2
import os
from collections import defaultdict
import pprint

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
        print(f"{icon_path}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_match = icon_name

    return best_match

def match_charge(screenshot, charge_folder, bbox, visualize=False):
    x, y, w, h = bbox
    cropped = screenshot[y:y+h, x:x+w]
    
    if visualize:
        cv2.imshow("Screenshot", cropped)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()  # Close the window after key press

    best_score = float('-inf')
    best_match = None

    for img_name in os.listdir(charge_folder):
        img_path = os.path.join(charge_folder, img_name)
        ref_img = cv2.imread(img_path)

        if ref_img is None or ref_img.shape[:2] != cropped.shape[:2]:
            ref_img = cv2.resize(ref_img, (w, h))

        res = cv2.matchTemplate(cropped, ref_img, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        print(f"{img_path}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_match = img_name

    return best_match

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

def format_timestamp(seconds):
    """Format seconds to M:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


if __name__ == "__main__":
    video_path = "v10.mp4"  # Replace with your video path
    visualize = False
    
    cap = cv2.VideoCapture(video_path) 
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, screenshot = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read frame from video")
        exit()
    
    
    if frame_width == 2560 and frame_height == 1440:
        ability_bbox = (1015, 1300, 75, 75)
        charge_bboxes = [(990, 1383, 126, 25), (1139, 1383, 126, 25), (1289, 1383, 126, 25), (1440, 1383, 126, 25)]
    elif frame_width == 1920 and frame_height == 1080:
        ability_bbox = (759, 974, 60, 60)
        charge_bboxes = [(740, 1036, 100, 23), (852, 1036, 100, 23), (965, 1036, 100, 23), (1078, 1036, 100, 23)]
    elif frame_width == 1280 and frame_height == 720:
        ability_bbox = (506, 650, 40, 40)
        charge_bboxes = [(496, 691, 60, 14), (571, 691, 60, 14), (646, 691, 60, 14), (722, 691, 60, 14)]
    else:
        print("Unsupported video resolution. Please provide a video with 2560x1440, 1920x1080, or 1280x720 resolution.")
    

    ability_matches = defaultdict(int)
    slot_matches = defaultdict(list)
    frames = extract_frames(video_path, interval=1)
    for i, frame in enumerate(frames):
        timestamp_str = format_timestamp(i)
        print(f"Processing frame at {timestamp_str}\n")
        if max(ability_matches.values(), default=0) < 3: 
            ability_icon = match_icon(frame, "assets/abilities", ability_bbox, visualize)
            print(f"Ability icon matched: {ability_icon}")
            ability_matches[ability_icon] += 1
        
        for bbox in charge_bboxes:
            charge_icon = match_charge(frame, "assets/count", bbox, visualize)
            slot_matches[timestamp_str].append(charge_icon[0:3])
            print(f"Ability icon matched: {charge_icon[0:3]}")
        
    max_key = max(ability_matches, key=ability_matches.get)
    print(max_key)
    pprint.pprint(dict((slot_matches)))
    
    
