import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash


THRESHOLD = 12

def is_same_image(path1: str, path2: str, threshold) -> bool:
    """Return True if the perceptual hash difference is ≤ threshold."""
    # Load images once
    pil_img1, pil_img2 = Image.open(path1), Image.open(path2)
    cv_img1 = cv2.imread(path1)
    cv_img2 = cv2.imread(path2)
    
    # Preprocess: convert grayscale and resize
    gray1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cv2.resize(cv_img2, (cv_img1.shape[1], cv_img1.shape[0])), 
                         cv2.COLOR_BGR2GRAY)
    
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
    print(f"hash diffs↦ p:{phd}, d:{dhd}, a:{ahd}, w:{whd}")
    
    # 2) SSIM
    s, _ = ssim(gray1, gray2, full=True)
    print(f"SSIM: {s:.2f}")
    
    # 3) Shape similarity via contours
    # Threshold and find contours
    _, thresh1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Compare number of significant contours (helps differentiate complex shapes like robot vs triangle)
    sig_contours1 = [c for c in contours1 if cv2.contourArea(c) > 100]
    sig_contours2 = [c for c in contours2 if cv2.contourArea(c) > 100]
    contour_diff = abs(len(sig_contours1) - len(sig_contours2))
    print(f"Contour count diff: {contour_diff}")
    
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
    print(f"SIFT matches: {sift_matches}")
    
    # 5) Template matching
    res = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    print(f"Template match: {max_val:.2f}")
    
    # Decision making - weighted approach
    score = 0
    
    # Hash scores (triangle icons have low hash differences)
    if phd <= 15:
        score += 2
    if dhd <= 15:
        score += 2
    
    # SSIM score (higher for similar images)
    if s >= 0.5:
        score += 3
    
    # Shape similarity (triangle vs robot should have different contour counts)
    if contour_diff <= 2:
        score += 2
    
    # SIFT features (good for distinguishing shapes)
    if sift_matches >= 6:
        score += 3
    
    # Template matching
    if max_val >= 0.4:
        score += 2
    
    print(f"Total similarity score: {score}/12")
    
    # Triangle icons should score higher, robot should score lower
    return score >= 6

def main():
    IMG1, IMG2 = "nano1.png", "nano.png"
    print(f"{IMG1} and {IMG2}")
    print(is_same_image(IMG1, IMG2, THRESHOLD)) # Expected: True -- Actual: True
    print("=======")
    
    IMG1, IMG2 = "nano2.png", "nano.png"
    print(f"{IMG1} and {IMG2}")
    print(is_same_image(IMG1, IMG2, THRESHOLD)) # Expected: True -- Actual: False
    print("=======")
    
    IMG1, IMG2 = "nano3.png", "nano.png"
    print(f"{IMG1} and {IMG2}")
    print(is_same_image(IMG1, IMG2, THRESHOLD)) # Expected: True -- Actual: False
    print("=======")
    
    IMG1, IMG2 = "nano4.png", "nano.png"
    print(f"{IMG1} and {IMG2}")
    print(is_same_image(IMG1, IMG2, THRESHOLD)) # Expected: True -- Actual: False
    print("=======")

    
    IMG1, IMG2 = "bot.png", "nano.png"
    print(f"{IMG1} and {IMG2}")
    print(is_same_image(IMG1, IMG2, THRESHOLD)) # Expected: False -- Actual False 
    print("=======")

if __name__ == "__main__":
    main()
