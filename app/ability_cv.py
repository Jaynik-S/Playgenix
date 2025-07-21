import cv2
import os
from collections import defaultdict
import pprint
from PIL import Image
import imagehash
from img_preprocess import match_charge_cnn
import numpy as np


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
# def match_charge(screenshot, charge_folder, bbox, visualize=False):
#     x, y, w, h = bbox
#     cropped = screenshot[y:y+h, x:x+w]
    
#     if visualize:
#         cv2.imshow("Screenshot", cropped)
#         cv2.waitKey(0)  
#         cv2.destroyAllWindows() 

#     best_score = float('-inf')
#     best_match = None

#     # Convert to grayscale before histogram equalization
#     if len(cropped.shape) == 3:
#         cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#     cropped = cv2.equalizeHist(cropped)
#     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     # cropped = clahe.apply(cropped)


#     for img_name in os.listdir(charge_folder):
#         img_path = os.path.join(charge_folder, img_name)
#         ref_img = cv2.imread(img_path)

#         if ref_img is None or ref_img.shape[:2] != cropped.shape[:2]:
#             ref_img = cv2.resize(ref_img, (w, h))

#         # Convert to grayscale before histogram equalization
#         if len(ref_img.shape) == 3:
#             ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
#         ref_img = cv2.equalizeHist(ref_img)
#         # ref_img = clahe.apply(ref_img)
        

#         res = cv2.matchTemplate(cropped, ref_img, cv2.TM_CCOEFF_NORMED)
#         _, score, _, _ = cv2.minMaxLoc(res)
#         print(f"{img_path}: {score:.4f}")

#         if score > best_score:
#             best_score = score
#             best_match = img_name

#     return best_match

#############################
# AGENT DETECTION
#############################
def detect_agent(screenshot, bbox, agent_folder, visualize=False, threshold=0.3):
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
        
        print(f"{agent_path}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_match = agent_name

    print(f"\n=====\nBest match: {best_match} with score {best_score:.4f}")
    # Determine agent name if above threshold
    if best_score < threshold and (best_match == "Blue.png" and best_match == "Red.png"):
        return None, None
    
    agent = os.path.splitext(best_match)[0]

    # Check for yellow background = ultimate ready
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = cv2.countNonZero(mask) / (h * w)
    ult_ready = yellow_ratio > 0.1

    # Visualization debugging
    if visualize:
        # reload & resize best icon
        icon_path = os.path.join(agent_folder, best_match)
        best_icon = cv2.imread(icon_path)
        best_icon = cv2.resize(best_icon, (w, h))
        # prepare mask for display
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # stack and annotate
        debug_vis = np.hstack([cropped, best_icon, mask_bgr])
        cv2.putText(debug_vis,
                    f"{agent or 'None'} | Score:{best_score:.2f} | Yellow:{yellow_ratio:.2f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1)
        cv2.imshow("Detect Agent Debug", debug_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return agent, ult_ready


def main(file_name: str, visualize: bool = False):
    video_path = f"app/static/uploads/{file_name}"
    
    #temp
    video_path = "v15.mp4"
    visualize = True
    
    cap = cv2.VideoCapture(video_path) 
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, screenshot = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read frame from video")
        exit()
    
    print(f"Video resolution: {frame_width}x{frame_height}")
    if frame_width == 2560 and frame_height == 1440:
        ability_bbox = (1015, 1300, 75, 75)
        charge_bboxes = [(990, 1383, 126, 25), (1139, 1383, 126, 25), (1289, 1383, 126, 25), (1440, 1383, 126, 25)]
        #Sage, Reyna, KJ, Brim, Jett
        team_status = [(591, 37, 60, 60), (678, 37, 60, 60), (768, 37, 60, 60), (856, 37, 60, 60), (944, 37, 60, 60)]  
        # Raze, Sova, Phoenix, Reyna, Omen
        enemy_status = [(1560, 37, 60, 60), (1648, 37, 60, 60), (1736, 37, 60, 60), (1824, 37, 60, 60), (1912, 37, 60, 60)] 
    elif frame_width == 1920 and frame_height == 1080:
        ability_bbox = (759, 974, 60, 60)
        charge_bboxes = [(740, 1036, 100, 23), (852, 1036, 100, 23), (965, 1036, 100, 23), (1078, 1036, 100, 23)]
        # Sage, Reyna, Chamber, Omen, Fade
        team_status = [(444, 27, 45, 44), (509, 27, 45, 44), (573, 27, 45, 44), (641, 27, 45, 44), (708, 27, 45, 44)]
        # Reyna, Fade, Omen, Chamber, Jett
        enemy_status = [(1169, 27, 45, 44), (1235, 27, 45, 44), (1301, 27, 45, 44), (1368, 27, 45, 44), (1434, 27, 45, 44)]
    elif frame_width == 1280 and frame_height == 720:
        ability_bbox = (506, 650, 40, 40)
        charge_bboxes = [(496, 691, 60, 14), (571, 691, 60, 14), (646, 691, 60, 14), (722, 691, 60, 14)]
        # Clove, Jett, Killjoy, Kayo, Reyna
        team_status = [(295, 18, 30, 30), (338, 18, 30, 30), (383, 18, 30, 30), (427, 18, 30, 30), (471, 18, 30, 30)]
        # Omen, Killjoy, Phoenix, Jett, Sova    
        enemy_status = [(779, 18, 30, 30), (823, 18, 30, 30), (867, 18, 30, 30), (912, 18, 30, 30), (956, 18, 30, 30)]
    else:
        print("Unsupported video resolution. Please provide a video with 2560x1440, 1920x1080, or 1280x720 resolution.")
    

    ability_matches = defaultdict(int)
    slot_matches = defaultdict(list)
    team_status_dict = defaultdict(list)
    enemy_status_dict = defaultdict(list)
    frames = extract_frames(video_path, interval=1)
    for i, frame in enumerate(frames):
        timestamp_str = format_timestamp(i)
        print(f"Processing frame at {timestamp_str}\n")

        # if max(ability_matches.values(), default=0) < 3: 
        #     ability_icon = match_icon(frame, "assets/abilities", ability_bbox, visualize)
        #     print(f"Ability icon matched: {ability_icon}")
        #     ability_matches[ability_icon] += 1

        ##########################################
        
        # for bbox in charge_bboxes:
        #     x, y, w, h = bbox
        #     cropped = frame[y:y+h, x:x+w]
            
        #     pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        #     target_size = (128, 32)
        #     new_img = Image.new('RGB', target_size, (0, 0, 0))
        #     offset = ((target_size[0] - pil_img.width) // 2, (target_size[1] - pil_img.height) // 2)
        #     new_img.paste(pil_img, offset)
        #     resized_crop = cv2.cvtColor(np.array(new_img), cv2.COLOR_RGB2BGR)
            
        #     if visualize:
        #         cv2.imshow("Resized Crop", resized_crop)
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()

        #     charge_icon = match_charge_cnn(resized_crop, None, (0, 0, target_size[0], target_size[1]))
        #     slot_matches[timestamp_str].append(charge_icon[0:3])
        #     print(f"Ability icon matched: {charge_icon[0:3]}")
        
        ##########################################
        
        # for bbox in team_status:
        #     agent, ult_ready = detect_agent(frame, bbox, "assets/agents/normal", visualize)
        #     if agent:
        #         print(f"Agent detected: {agent}, Ultimate ready: {ult_ready}")
        #         team_status_dict[timestamp_str].append((agent, ult_ready))
        #     else:
        #         print("No agent detected")
        #         team_status_dict[timestamp_str].append("___")
        
        ##########################################
                
        for bbox in enemy_status:
            agent, ult_ready = detect_agent(frame, bbox, "assets/agents/flipped", visualize)
            if agent:
                print(f"Agent detected: {agent}, Ultimate ready: {ult_ready}")
                enemy_status_dict[timestamp_str].append((agent, ult_ready))
            else:
                print("No agent detected")
                enemy_status_dict[timestamp_str].append("___")
        
    # max_key = max(ability_matches, key=ability_matches.get)
    # print(f'Ability match: {max_key[:len(max_key)-4]}')
    pprint.pprint(enemy_status_dict)

'''
    #V10 (5K JETT)
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
'''

if __name__ == "__main__":
    main("v11.mp4", False)