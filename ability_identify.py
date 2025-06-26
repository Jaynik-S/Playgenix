import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash
import os
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

def remove_background(image):
    """Remove background from an image, preserving white ability icons."""
    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for white/light colored pixels (ability icons are typically white)
    # Target high brightness pixels
    _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Also check for low saturation (white/gray pixels have low saturation)
    saturation = hsv[:,:,1]
    _, low_sat_mask = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Combine masks to get white/light pixels with low saturation
    icon_mask = cv2.bitwise_and(white_mask, low_sat_mask)
    
    # Clean up the mask with morphological operations
    kernel = np.ones((2,2), np.uint8)
    # Remove small noise
    icon_mask = cv2.morphologyEx(icon_mask, cv2.MORPH_OPEN, kernel)
    # Fill small gaps in the icon
    icon_mask = cv2.morphologyEx(icon_mask, cv2.MORPH_CLOSE, kernel)
    
    # Create a 4-channel image (BGRA) for transparency
    result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    
    # Copy the original image to RGB channels where mask is active
    result[:,:,0] = np.where(icon_mask > 0, image[:,:,0], 0)  # Blue
    result[:,:,1] = np.where(icon_mask > 0, image[:,:,1], 0)  # Green
    result[:,:,2] = np.where(icon_mask > 0, image[:,:,2], 0)  # Red
    result[:,:,3] = icon_mask  # Alpha channel (transparency)
    
    # Convert back to 3-channel for compatibility with existing code
    result_3ch = cv2.bitwise_and(image, image, mask=icon_mask)
    
    return result_3ch, icon_mask

def is_same_image(img1, img2, threshold):
    """Return True if the perceptual hash difference is ≤ threshold."""
    # Handle different types of inputs
    if isinstance(img1, str):
        pil_img1 = Image.open(img1)
        cv_img1 = cv2.imread(img1)
        # Apply background removal to reference image too
        cv_img1, _ = remove_background(cv_img1)
    else:
        cv_img1 = img1
        # Remove background from the cropped frame
        cv_img1, _ = remove_background(cv_img1)
    
    # Convert OpenCV image to PIL format
    cv_rgb1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2RGB)
    pil_img1 = Image.fromarray(cv_rgb1)
    
    if isinstance(img2, str):
        pil_img2 = Image.open(img2)
        cv_img2 = cv2.imread(img2)
    else:
        cv_img2 = img2
    
    # Convert OpenCV image to PIL format
    cv_rgb2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2RGB)
    pil_img2 = Image.fromarray(cv_rgb2)
    
    gray1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2GRAY)
    cv_img2_resized = cv2.resize(cv_img2, (cv_img1.shape[1], cv_img1.shape[0]))
    gray2 = cv2.cvtColor(cv_img2_resized, cv2.COLOR_BGR2GRAY)
    
    # For transparent images, focus on non-zero pixels only
    # Create masks for non-transparent pixels
    mask1 = gray1 > 0
    mask2 = gray2 > 0
    
    # Only normalize areas that contain actual content
    if np.any(mask1):
        gray1_masked = np.where(mask1, gray1, 0)
        gray1 = cv2.equalizeHist(gray1_masked.astype(np.uint8))
    
    if np.any(mask2):
        gray2_masked = np.where(mask2, gray2, 0)
        gray2 = cv2.equalizeHist(gray2_masked.astype(np.uint8))
    
    # 1) Multiple perceptual hashes
    ph1, ph2 = imagehash.phash(pil_img1), imagehash.phash(pil_img2)
    dh1, dh2 = imagehash.dhash(pil_img1), imagehash.dhash(pil_img2)
    ah1, ah2 = imagehash.average_hash(pil_img1), imagehash.average_hash(pil_img2)
    wh1, wh2 = imagehash.whash(pil_img1), imagehash.whash(pil_img2)
    phd, dhd = abs(ph1-ph2), abs(dh1-dh2)
    ahd, whd = abs(ah1-ah2), abs(wh1-wh2)
    
    # 2) SSIM - only on non-transparent regions
    if np.any(mask1) and np.any(mask2):
        s, _ = ssim(gray1, gray2, full=True)
    else:
        s = 0  # No content to compare
    
    # 3) Shape similarity via contours - use masks for thresholding
    _, thresh1 = cv2.threshold(gray1, 50, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY)
    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Compare number of significant contours (adjusted for smaller transparent images)
    sig_contours1 = [c for c in contours1 if cv2.contourArea(c) > 50]
    sig_contours2 = [c for c in contours2 if cv2.contourArea(c) > 50]
    contour_diff = abs(len(sig_contours1) - len(sig_contours2))
    
    # 4) SIFT feature matching (good for shape detection)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    sift_matches = 0
    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        sift_matches = len(good_matches)
    
    # 5) Template matching - only on content areas
    if np.any(mask1) and np.any(mask2):
        res = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
    else:
        max_val = 0
    
    score = 0
    
    # Adjusted scoring for transparent images
    # Hash scores (lower thresholds for transparent images)
    if phd <= 10: score += 2
    if dhd <= 10: score += 1
    
    # SSIM score (higher for similar images)
    if s >= 0.4: score += 3
    
    # Shape similarity (adjusted threshold for transparent images)
    if contour_diff <= 1: score += 2
    
    # SIFT features (lower threshold for smaller images)
    if sift_matches >= 3: score += 3
    
    # Template matching
    if max_val >= 0.3: score += 2
    
    print(f"Scores - PHD: {phd}, DHD: {dhd}, SSIM: {s:.3f}, Contours: {contour_diff}, SIFT: {sift_matches}, Template: {max_val:.3f}")
    print(f"Total similarity score: {score}/13")
    
    # Adjusted threshold for transparent images
    return score >= 6

def display_frame(frame, title="Frame", wait_time=0):
    """Display a frame in a window."""
    cv2.imshow(title, frame)
    key = cv2.waitKey(wait_time)
    return key

def compare_video_frames_to_abilities(video_path, abilities_folder, crop_region=None, interval=1, visualize=False, max_matches=None):
    """Extract frames from a video, optionally crop them, and compare to all ability images in the folder."""
    frames = extract_frames(video_path, interval)
    results = []
    match_count = 0
    
    # Get all PNG files from abilities folder
    ability_images = []
    if os.path.exists(abilities_folder):
        for filename in os.listdir(abilities_folder):
            if filename.lower().endswith('.png'):
                ability_images.append(os.path.join(abilities_folder, filename))
    
    if not ability_images:
        print(f"No PNG files found in {abilities_folder}")
        return []
    
    print(f"Found {len(ability_images)} ability images to compare against")
    
    for i, frame in enumerate(frames):
        timestamp = i * interval
        
        if crop_region is not None:
            x, y, width, height = crop_region
            
            if visualize:
                # Draw rectangle on original frame to show crop region
                marked_frame = frame.copy()
                cv2.rectangle(marked_frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                display_frame(marked_frame, "Full Frame with Crop Region", 500)
            
            cropped_frame = crop_frame(frame, x, y, width, height)
        else:
            cropped_frame = frame
        
        # Remove background from cropped frame
        processed_frame, mask = remove_background(cropped_frame)
            
        if visualize:
            display_frame(cropped_frame, f"Original Cropped Frame at {timestamp}s", 500)
            display_frame(processed_frame, f"Background Removed at {timestamp}s", 1000)
        
        # Compare against all ability images using processed frame
        matched_ability = None
        for ability_path in ability_images:
            if is_same_image(processed_frame, ability_path, THRESHOLD):
                matched_ability = os.path.basename(ability_path)
                break
        
        is_match = matched_ability is not None
        results.append((i, timestamp, is_match, matched_ability))
        
        # Check if we've reached the maximum number of matches
        if is_match:
            match_count += 1
            print(f"Match found at {timestamp}s: {matched_ability}")
            if max_matches is not None and match_count >= max_matches:
                break
        
        if visualize:
            # Use green for matches, red for non-matches
            result_color = (0, 255, 0) if is_match else (0, 0, 255)
            result_frame = processed_frame.copy()
            result_text = f"MATCH: {matched_ability}" if is_match else "NO MATCH"
            # Add text showing the result
            cv2.putText(
                result_frame, 
                result_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                result_color, 
                2
            )
            display_frame(result_frame, f"Result at {timestamp}s", 1500)
            
            # Check if user wants to exit
            key = cv2.waitKey(10)
            if key == 27:  # ESC key
                break
    
    if visualize:
        cv2.destroyAllWindows()
        
    return results

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

def scale_reference_image(image_path, scale_factor):
    """
    Scales a reference image by the given factor and saves a temporary version.
    Returns the path to the scaled image.
    """
    # Create temp directory if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # Get original filename without path
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    scaled_path = f"temp/{name}_scaled{ext}"
    
    # Load and scale the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)
    scaled_img = cv2.resize(img, (new_width, new_height))
    
    # Save the scaled image
    cv2.imwrite(scaled_path, scaled_img)
    print(f"Scaled reference image saved to {scaled_path}")
    
    return scaled_path

def main():
    VIDEO_PATHS = ["v4.mp4", "v6.mp4", "v3.mp4"]
    for VIDEO in VIDEO_PATHS:
        ABILITIES_FOLDER = "assets/abilities"
        VISUALIZE = True
        MAX_MATCHES = 1   
        
        frame_width, frame_height = get_video_frame_size(VIDEO)
        print(f"Video frame size: {frame_width}x{frame_height}")
        
        # Define crop region (x, y, width, height)
        if frame_width == 2560 and frame_height == 1440:
            CROP_REGION = (1015, 1300, 75, 75)
        elif frame_width == 1920 and frame_height == 1080:
            CROP_REGION = (759, 974, 60, 60)
        elif frame_width == 1280 and frame_height == 720:
            CROP_REGION = (506, 650, 40, 40)
        else:
            print("Unsupported video resolution. Please provide a video with 2560x1440, 1920x1080, or 1280x720 resolution.")
            return
        
        print(f"Comparing video frames from {VIDEO} to abilities in {ABILITIES_FOLDER}")
        results = compare_video_frames_to_abilities(
            VIDEO, 
            ABILITIES_FOLDER, 
            crop_region=CROP_REGION,
            interval=1,  
            visualize=VISUALIZE,
            max_matches=MAX_MATCHES 
        )
        
        matching_frames = [r for r in results if r[2]]
        print(f"\nFound {len(matching_frames)} matching frames out of {len(results)} analyzed")
        
        if matching_frames:
            print("Matches found at timestamps (seconds):")
            for _, timestamp, is_match, matched_ability in matching_frames:
                print(f"- {timestamp}s: {matched_ability}")

if __name__ == "__main__":
    main()
