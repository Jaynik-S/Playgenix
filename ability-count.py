import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash
import os
import json
import numpy as np

THRESHOLD = 12

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

def crop_frame(frame, x, y, width, height):
    """Crop a frame to the specified region."""
    if frame is None:
        raise ValueError("Frame is None")
    
    frame_height, frame_width = frame.shape[:2]
    
    # Validate crop parameters
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError(f"Invalid crop region: {x},{y},{width},{height} for frame size {frame_width}x{frame_height}")
    
    return frame[y:y+height, x:x+width]

def is_same_image(img1, img2, threshold):
    """Return True if the status bar images match based on bar count and colors."""
    # Handle different types of inputs
    if isinstance(img1, str):
        cv_img1 = cv2.imread(img1)
    else:
        cv_img1 = img1
    
    if isinstance(img2, str):
        cv_img2 = cv2.imread(img2)
    else:
        cv_img2 = img2
    
    # Resize img2 to match img1 dimensions
    cv_img2_resized = cv2.resize(cv_img2, (cv_img1.shape[1], cv_img1.shape[0]))
    
    # Analyze both images
    bars1 = analyze_status_bars(cv_img1)
    bars2 = analyze_status_bars(cv_img2_resized)
    
    # Calculate similarity score
    similarity_score = calculate_bar_similarity(bars1, bars2)
    
    # Return True if similarity is above threshold (threshold now represents minimum similarity score)
    return similarity_score >= 0.8  # 80% similarity threshold

def analyze_status_bars(image):
    """Analyze an image to extract bar count and colors."""
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define color ranges for green and grey bars
    # Green range in HSV
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])
    
    # Grey range in HSV (low saturation)
    grey_lower = np.array([0, 0, 50])
    grey_upper = np.array([180, 30, 200])
    
    # Create masks for green and grey regions
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    grey_mask = cv2.inRange(hsv, grey_lower, grey_upper)
    
    # Apply morphological operations to clean up masks
    kernel = np.ones((3,3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    grey_mask = cv2.morphologyEx(grey_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the combined mask to identify individual bars
    combined_mask = cv2.bitwise_or(green_mask, grey_mask)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find bar-like shapes
    bars = []
    min_bar_area = 50  # Minimum area for a bar
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_bar_area:
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if it's bar-like (width should be reasonable compared to height)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 5.0:  # Filter out non-bar shapes
            continue
        
        # Determine if this bar is green or grey
        bar_region = hsv[y:y+h, x:x+w]
        green_pixels = cv2.countNonZero(cv2.inRange(bar_region, green_lower, green_upper))
        grey_pixels = cv2.countNonZero(cv2.inRange(bar_region, grey_lower, grey_upper))
        
        if green_pixels > grey_pixels:
            color = 'green'
        else:
            color = 'grey'
        
        bars.append({
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'color': color,
            'area': area
        })
    
    # Sort bars by x-coordinate (left to right)
    bars.sort(key=lambda b: b['x'])
    
    return {
        'count': len(bars),
        'bars': bars,
        'green_count': sum(1 for b in bars if b['color'] == 'green'),
        'grey_count': sum(1 for b in bars if b['color'] == 'grey')
    }

def calculate_bar_similarity(bars1, bars2):
    """Calculate similarity score between two bar analyses."""
    # Perfect match: same count and same color pattern
    if (bars1['count'] == bars2['count'] and 
        bars1['green_count'] == bars2['green_count'] and 
        bars1['grey_count'] == bars2['grey_count']):
        
        # Check if color pattern matches (same sequence of colors)
        if bars1['count'] > 0 and bars2['count'] > 0:
            pattern1 = [bar['color'] for bar in bars1['bars']]
            pattern2 = [bar['color'] for bar in bars2['bars']]
            if pattern1 == pattern2:
                return 1.0  # Perfect match
            else:
                return 0.9  # Same counts but different pattern
        else:
            return 1.0  # Both have no bars
    
    # Partial match calculations
    score = 0.0
    
    # Count similarity (40% weight)
    if bars1['count'] == bars2['count']:
        score += 0.4
    elif abs(bars1['count'] - bars2['count']) == 1:
        score += 0.2
    
    # Green bar count similarity (30% weight)
    if bars1['green_count'] == bars2['green_count']:
        score += 0.3
    elif abs(bars1['green_count'] - bars2['green_count']) == 1:
        score += 0.15
    
    # Grey bar count similarity (30% weight)
    if bars1['grey_count'] == bars2['grey_count']:
        score += 0.3
    elif abs(bars1['grey_count'] - bars2['grey_count']) == 1:
        score += 0.15
    
    return score

def find_best_match(cropped_frame, reference_images):
    """Find the best matching reference image for a cropped frame."""
    best_match = None
    best_score = -1
    
    # Analyze the cropped frame
    frame_bars = analyze_status_bars(cropped_frame)
    
    for ref_path in reference_images:
        ref_img = cv2.imread(ref_path)
        ref_resized = cv2.resize(ref_img, (cropped_frame.shape[1], cropped_frame.shape[0]))
        
        # Analyze the reference image
        ref_bars = analyze_status_bars(ref_resized)
        
        # Calculate similarity score
        score = calculate_bar_similarity(frame_bars, ref_bars)
        
        if score > best_score:
            best_score = score
            best_match = os.path.splitext(os.path.basename(ref_path))[0]
    
    # Always return the best match found, even if score is low
    return best_match

def get_video_frame_size(video_path):
    """Returns the frame dimensions (width, height) of a video file."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video.release()
    return (width, height)

def get_crop_regions_for_resolution(width, height):
    """Define crop regions for 4 slots based on video resolution."""
    if width == 2560 and height == 1440:
        return {
            "Slot1": (990, 1383, 126, 25),
            "Slot2": (1139, 1383, 126, 25),
            "Slot3": (1289, 1383, 126, 25),
            "Slot4": (1440, 1383, 126, 25)
        }
    elif width == 1920 and height == 1080:
        return {
            "Slot1": (740, 1036, 100, 23),
            "Slot2": (852, 1036, 100, 23),
            "Slot3": (965, 1036, 100, 23),
            "Slot4": (1078, 1036, 100, 23)
        }
    elif width == 1280 and height == 720:
        return {
            "Slot1": (496, 691, 60, 14),
            "Slot2": (571, 691, 60, 14),
            "Slot3": (646, 691, 60, 14),
            "Slot4": (722, 691, 60, 14)
        }
    else:
        raise ValueError(f"Unsupported resolution: {width}x{height}")

def format_timestamp(seconds):
    """Format seconds to M:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

def display_frame_with_crops(frame, crop_regions, slot_matches, timestamp, wait_time=1000):
    """Display frame with crop regions highlighted and matches shown."""
    display_frame = frame.copy()
    
    # Draw rectangles for each slot
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Green, Red, Blue, Yellow
    
    for i, (slot_name, (x, y, w, h)) in enumerate(crop_regions.items()):
        color = colors[i % len(colors)]
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
        
        # Add slot label and match result
        match_text = slot_matches.get(slot_name, "No match")
        cv2.putText(display_frame, f"{slot_name}: {match_text}", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Add timestamp
    cv2.putText(display_frame, f"Time: {timestamp}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Ability Count Analysis", display_frame)
    key = cv2.waitKey(wait_time)
    return key

def process_video_for_ability_counts(video_path, reference_folder, output_json, visualize=False):
    """Process video to identify ability counts in all 4 slots."""
    # Get video properties
    frame_width, frame_height = get_video_frame_size(video_path)
    print(f"Video frame size: {frame_width}x{frame_height}")
    
    # Get crop regions for this resolution
    crop_regions = get_crop_regions_for_resolution(frame_width, frame_height)
    
    # Get reference images
    reference_images = []
    if os.path.exists(reference_folder):
        for filename in os.listdir(reference_folder):
            if filename.lower().endswith('.png'):
                reference_images.append(os.path.join(reference_folder, filename))
    
    if not reference_images:
        raise ValueError(f"No PNG files found in {reference_folder}")
    
    print(f"Found {len(reference_images)} reference images")
    
    # Extract frames
    frames = extract_frames(video_path, interval=1)
    print(f"Extracted {len(frames)} frames")
    
    results = {}
    
    for i, frame in enumerate(frames):
        timestamp_str = format_timestamp(i)
        slot_matches = {}
        
        print(f"Processing frame at {timestamp_str}")
        
        # Process each slot
        for slot_name, (x, y, w, h) in crop_regions.items():
            cropped_frame = crop_frame(frame, x, y, w, h)
            best_match = find_best_match(cropped_frame, reference_images)
            slot_matches[slot_name] = best_match
            
            if best_match:
                print(f"  {slot_name}: {best_match}")
            else:
                print(f"  {slot_name}: No match found")
        
        results[timestamp_str] = slot_matches
        
        # Visualization
        if visualize:
            key = display_frame_with_crops(frame, crop_regions, slot_matches, timestamp_str)
            if key == 27:  # ESC key
                break
    
    if visualize:
        cv2.destroyAllWindows()
    
    # Save results to JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_json}")
    return results

def main():
    VIDEO_PATH = "v9.mp4"
    REFERENCE_FOLDER = "assets/count"
    OUTPUT_JSON = "temp/ability_counts.json"
    VISUALIZE = False  # Set to True for debugging
    
    results = process_video_for_ability_counts(
        VIDEO_PATH, 
        REFERENCE_FOLDER, 
        OUTPUT_JSON, 
        visualize=VISUALIZE
    )
    
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
    
    # Compare results with correct answer
    total_comparisons, correct_matches = 0, 0
    
    print("\n=== ACCURACY COMPARISON ===")
    
    for timestamp, predicted_slots in results.items():
        if timestamp in correct_answer:
            expected_slots = correct_answer[timestamp]
            
            # Convert results dictionary to list in slot order
            predicted_list = [
                predicted_slots.get('Slot1', 'unknown'),
                predicted_slots.get('Slot2', 'unknown'),
                predicted_slots.get('Slot3', 'unknown'),
                predicted_slots.get('Slot4', 'unknown')
            ]
            
            # Compare each slot
            for i, (predicted, expected) in enumerate(zip(predicted_list, expected_slots)):
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

if __name__ == "__main__":
    main()
