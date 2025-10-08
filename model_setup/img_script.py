import cv2
import os

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

def save_image(frame, bbox):
    x, y, w, h = bbox
    cropped = frame[y:y+h, x:x+w]
    os.makedirs("data", exist_ok=True)
    existing_files = [f for f in os.listdir('data') if f.endswith('.png')]
    next_number = len(existing_files) + 1
    filename = f"data/{next_number:06d}.png"
    cv2.imwrite(filename, cropped)

if __name__ == "__main__":
    videos = [f"v{i}.mp4" for i in range(1, 33)]

    for video_path in videos:
        print(f"Processing video: {video_path}")
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
        frames = extract_frames(video_path, interval=1)
        for i, frame in enumerate(frames):
            for bbox in charge_bboxes:
                save_image(frame, bbox)