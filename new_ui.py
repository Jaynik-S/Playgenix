import cv2
import os
from collections import defaultdict
import pprint
from PIL import Image
import imagehash


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
        print(f"{icon_path}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_match = icon_name

    return best_match

#############################
# CHARGES
#############################
def match_charge(screenshot, charge_folder, bbox, visualize=False):
    x, y, w, h = bbox
    cropped = screenshot[y:y+h, x:x+w]
    
    if visualize:
        cv2.imshow("Screenshot", cropped)
        cv2.waitKey(0)  
        cv2.destroyAllWindows() 

    best_score = float('-inf')
    best_match = None

    # Convert to grayscale before histogram equalization
    if len(cropped.shape) == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cropped = cv2.equalizeHist(cropped)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cropped = clahe.apply(cropped)


    for img_name in os.listdir(charge_folder):
        img_path = os.path.join(charge_folder, img_name)
        ref_img = cv2.imread(img_path)

        if ref_img is None or ref_img.shape[:2] != cropped.shape[:2]:
            ref_img = cv2.resize(ref_img, (w, h))

        # Convert to grayscale before histogram equalization
        if len(ref_img.shape) == 3:
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        ref_img = cv2.equalizeHist(ref_img)
        # ref_img = clahe.apply(ref_img)
        

        res = cv2.matchTemplate(cropped, ref_img, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        print(f"{img_path}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_match = img_name

    return best_match



if __name__ == "__main__":
    video_path = "v9.mp4" 
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
            charge_icon = match_charge(frame, "assets/transparent", bbox, visualize) #HERE
            slot_matches[timestamp_str].append(charge_icon[0:3])
            print(f"Ability icon matched: {charge_icon[0:3]}")
        
    max_key = max(ability_matches, key=ability_matches.get)
    print(f'Ability match: {max_key[:len(max_key)-4]}')
    
    #V9
    correct_answer =  {'0:00': ['3-3', '2-2', '0-1', '1-1'],
            '0:01': ['3-3', '2-2', '0-1', '1-1'],
            '0:02': ['2-3', '2-2', '0-1', '1-1'],
            '0:03': ['2-3', '2-2', '0-1', '1-1'],
            '0:04': ['2-3', '2-2', '0-1', '1-1'],
            '0:05': ['2-3', '2-2', '0-1', '1-1'],
            '0:06': ['2-3', '2-2', '1-1', '1-1'],
            '0:07': ['2-3', '2-2', '1-1', '1-1'],
            '0:08': ['2-3', '2-2', '1-1', '1-1'],
            '0:09': ['2-3', '1-2', '1-1', '1-1'],
            '0:10': ['2-3', '1-2', '1-1', '1-1'],
            '0:11': ['2-3', '1-2', '1-1', '1-1'],
            '0:12': ['2-3', '1-2', '1-1', '1-1'],
            '0:13': ['2-3', '1-2', '0-1', '1-1'],
            '0:14': ['2-3', '1-2', '0-1', '1-1'],
            '0:15': ['2-3', '1-2', '0-1', '1-1'],
            '0:16': ['2-3', '1-2', '0-1', '1-1'],
            '0:17': ['2-3', '1-2', '0-1', '1-1'],
            '0:18': ['2-3', '1-2', '0-1', '1-1']}

    # differences = {}
    # for timestamp in slot_matches:
    #     if timestamp in correct_answer:
    #         slot_diffs = []
    #         for i, (predicted, actual) in enumerate(zip(slot_matches[timestamp], correct_answer[timestamp])):
    #             if predicted != actual:
    #                 slot_diffs.append(f'{predicted} vs {actual}')
    #         if slot_diffs:
    #             differences[timestamp] = slot_diffs

    # print("\nDifferences between predictions and correct answers:")
    # pprint.pprint(differences)
    
    total_comparisons, correct_matches = 0, 0
    
    print("\n=== ACCURACY COMPARISON ===")
    
    for timestamp, predicted_slots in slot_matches.items():
        if timestamp in correct_answer:
            expected_slots = correct_answer[timestamp]
            
            # Compare each slot
            for i, (predicted, expected) in enumerate(zip(predicted_slots, expected_slots)):
                total_comparisons += 1
                if predicted == expected:
                    correct_matches += 1
                else:
                    print(f"❌ {timestamp} Slot{i+1}: Expected '{expected}', Got '{predicted}'")
    
    # Calculate accuracy
    accuracy = (correct_matches / total_comparisons) * 100 if total_comparisons > 0 else 0
    
    print(f"\n=== ACCURACY RESULTS ===")
    print(f"Total comparisons: {total_comparisons}")
    print(f"Correct matches: {correct_matches}")
    print(f"Accuracy: {accuracy:.2f}%")

