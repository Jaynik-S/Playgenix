import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash
import os

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
    """Return True if the perceptual hash difference is ≤ threshold."""
    # Handle different types of inputs
    if isinstance(img1, str):
        pil_img1 = Image.open(img1)
        cv_img1 = cv2.imread(img1)
    else:
        cv_img1 = img1
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
    
    # Normalize brightness/contrast
    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)
    
    # 1) Multiple perceptual hashes
    ph1, ph2 = imagehash.phash(pil_img1), imagehash.phash(pil_img2)
    dh1, dh2 = imagehash.dhash(pil_img1), imagehash.dhash(pil_img2)
    ah1, ah2 = imagehash.average_hash(pil_img1), imagehash.average_hash(pil_img2)
    wh1, wh2 = imagehash.whash(pil_img1), imagehash.whash(pil_img2)
    phd, dhd = abs(ph1-ph2), abs(dh1-dh2)
    ahd, whd = abs(ah1-ah2), abs(wh1-wh2)
    
    # 2) SSIM
    s, _ = ssim(gray1, gray2, full=True)
    
    # 3) Shape similarity via contours
    _, thresh1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Compare number of significant contours (helps differentiate complex shapes like robot vs triangle)
    sig_contours1 = [c for c in contours1 if cv2.contourArea(c) > 100]
    sig_contours2 = [c for c in contours2 if cv2.contourArea(c) > 100]
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
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        sift_matches = len(good_matches)
    
    # 5) Template matching
    res = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    
    score = 0
    
    # Hash scores (triangle icons have low hash differences)
    if phd <= 15: score += 2
    
    # SSIM score (higher for similar images)
    if s >= 0.5: score += 3
    
    # Shape similarity (triangle vs robot should have different contour counts)
    if contour_diff <= 2: score += 2
    
    # SIFT features (good for distinguishing shapes)
    if sift_matches >= 6: score += 3
    
    # Template matching
    if max_val >= 0.4: score += 2
    
    print(f"Total similarity score: {score}/12")
    
    # Triangle icons should score higher, robot should score lower
    return score >= 6

def display_frame(frame, title="Frame", wait_time=0):
    """Display a frame in a window."""
    cv2.imshow(title, frame)
    key = cv2.waitKey(wait_time)
    return key

def compare_video_frames_to_image(video_path, reference_image_path, crop_region=None, interval=1, visualize=False, max_matches=None):
    """Extract frames from a video, optionally crop them, and compare to a reference image."""
    frames = extract_frames(video_path, interval)
    results = []
    match_count = 0
    
    reference_img = cv2.imread(reference_image_path)
    
    if visualize:
        display_frame(reference_img, "Reference Image", 1000)
    
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
            
        
        if visualize:
            display_frame(cropped_frame, f"Cropped Frame at {timestamp}s", 1000)
        
        is_match = is_same_image(cropped_frame, reference_image_path, THRESHOLD)
        results.append((i, timestamp, is_match))
        
        # Check if we've reached the maximum number of matches
        if is_match:
            match_count += 1
            if max_matches is not None and match_count >= max_matches:
                break
        
        if visualize:
            # Use green for matches, red for non-matches
            result_color = (0, 255, 0) if is_match else (0, 0, 255)
            result_frame = cropped_frame.copy()
            result_text = "MATCH" if is_match else "NO MATCH"
            # Add text showing the result
            cv2.putText(
                result_frame, 
                result_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
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


def main():
    
    VIDEO_PATH = "video.mp4"  
    REFERENCE_IMAGE = "nano.png"  
    VISUALIZE = False
    MAX_MATCHES = 3     
    
    # Define crop region (x, y, width, height)
    # 1015, 1300, 75, 75 -- FIRST ABILITY
    # 1165, 1300, 75, 75 -- SECOND ABILITY
    CROP_REGION = (1015, 1300, 75, 75)  

    
    print(f"Comparing video frames from {VIDEO_PATH} to {REFERENCE_IMAGE}")
    results = compare_video_frames_to_image(
        VIDEO_PATH, 
        REFERENCE_IMAGE, 
        crop_region=CROP_REGION,
        interval=0.25,  
        visualize=VISUALIZE,
        max_matches=MAX_MATCHES 
    )
    
    matching_frames = [r for r in results if r[2]]
    print(f"\nFound {len(matching_frames)} matching frames out of {len(results)} analyzed")
    
    if matching_frames:
        print("Matches found at timestamps (seconds):")
        for _, timestamp, _ in matching_frames:
            print(f"-{timestamp}s")

if __name__ == "__main__":
    main()
