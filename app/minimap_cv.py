# minimap_cv.py
import os, cv2, json, math, glob
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Optional: reuse extract_frames from your existing module
try:
    from ability_cv import extract_frames  # 1-second stride helper
except Exception:
    def extract_frames(video_path, interval=1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        hop = max(1, int(round(fps * interval)))
        frames, idx = [], 0
        while True:
            ret, f = cap.read()
            if not ret: break
            if idx % hop == 0: frames.append(f)
            idx += 1
        cap.release()
        return frames

# -------------------------------
# Minimap locator (dynamic, safe fallback)
# -------------------------------
PRESET_MINIMAP_BOXES = {
    # x,y,w,h in base resolution 1920x1080 (VALORANT default HUD right-top)
    # These are *fallbacks* if dynamic locator fails.
    "1080p": (1515, 25, 360, 360),
    "1440p": (2020, 35, 480, 480),   # if captured 2560x1440
    "720p" : (1010, 16, 240, 240),
}
BASE = (1920, 1080)

def _scale_rect(rect, w, h, base=BASE):
    x,y,ww,hh = rect
    return (int(x*w/base[0]), int(y*h/base[1]), int(ww*w/base[0]), int(hh*h/base[1]))

def _best_preset_bbox(frame):
    H, W = frame.shape[:2]
    # choose closest aspect to known presets
    cand = ("1080p", PRESET_MINIMAP_BOXES["1080p"])
    if (W, H) == (2560,1440): cand = ("1440p", PRESET_MINIMAP_BOXES["1440p"])
    if (W, H) == (1280, 720): cand = ("720p" , PRESET_MINIMAP_BOXES["720p"])
    return _scale_rect(cand[1], W, H)

def _dynamic_minimap_roi(frame) -> Optional[Tuple[int,int,int,int]]:
    """
    Try to find the HUD minimap automatically.
    Observations:
      - Minimap is a near-square with bright white border on dark HUD.
      - It sits on the top-right quadrant.
    """
    H, W = frame.shape[:2]
    search = frame[0:int(H*0.6), int(W*0.5):W]  # top-right half
    offx = int(W*0.5)

    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 150)

    # Find large squares/rectangles (white border → strong edges)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w < 120 or h < 120: continue
        aspect = w / float(h)
        if 0.8 <= aspect <= 1.25:
            perim = cv2.arcLength(c, True)
            area  = cv2.contourArea(c)
            score = area / (perim+1e-5)
            if score > best_score:
                best_score = score
                best = (x+offx, y, w, h)
    return best  # may be None

def crop_minimap(frame) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """
    Returns (minimap_img, bbox in frame coords).
    Tries dynamic detection first, then preset fallback.
    """
    bbox = _dynamic_minimap_roi(frame)
    if bbox is None:
        bbox = _best_preset_bbox(frame)
    x,y,w,h = bbox
    return frame[y:y+h, x:x+w], bbox

# -------------------------------
# Map identification (scale + rotation invariance)
# -------------------------------
def _rotations(img):
    for ang in [0, 90, 180, 270]:
        if ang == 0:
            yield 0, img
        else:
            rotm = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), ang, 1.0)
            yield ang, cv2.warpAffine(img, rotm, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

def identify_map(minimap_img: np.ndarray, refs_dir: str) -> Tuple[str, float]:
    """
    ORB feature matching + RANSAC homography against all reference maps.
    Tries 0/90/180/270 rotations of the input (rotation-invariant).
    Returns (best_map_name, confidence[0..1]).
    """
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, edgeThreshold=15, patchSize=31)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Precompute features for input image at rotations
    in_rots = []
    for ang, im in _rotations(minimap_img):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kps, des = orb.detectAndCompute(gray, None)
        if des is None or len(kps) < 20: continue
        in_rots.append((ang, im, kps, des))

    best_name, best_score = None, 0.0

    for ref_path in glob.glob(os.path.join(refs_dir, "*.png")) + glob.glob(os.path.join(refs_dir, "*.jpg")):
        ref = cv2.imread(ref_path)
        if ref is None: continue
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        rk, rd = orb.detectAndCompute(ref_gray, None)
        if rd is None or len(rk) < 20: continue

        for ang, im, ik, ides in in_rots:
            matches = bf.knnMatch(ides, rd, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)
            if len(good) < 20:  # not enough to be confident
                continue
            src = np.float32([ik[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst = np.float32([rk[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else 0
            score = inliers / max(len(good), 1)
            if score > best_score:
                best_score = score
                best_name  = os.path.splitext(os.path.basename(ref_path))[0]
    return best_name or "Unknown", float(np.clip(best_score, 0, 1))

# -------------------------------
# Player / spike tracking on minimap
# -------------------------------
def _mask_hsv(img, lo, hi):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))

def _centers_from_mask(mask, min_area=10, max_area=2000):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area: continue
        (x,y), r = cv2.minEnclosingCircle(c)
        centers.append((int(x), int(y)))
    return centers

def track_minimap_objects(minimap_img: np.ndarray) -> Dict[str, List[Tuple[int,int]]]:
    """
    Returns centers for 'ally', 'enemy', 'spike' in minimap pixel coords.
    You might tune HSV ranges for your capture/recording color profile.
    """
    # Typical VALORANT colors (tweak as needed):
    ally_mask  = _mask_hsv(minimap_img, (80, 50, 120), (100, 255, 255))   # teal/cyan (allies)
    enemy1     = _mask_hsv(minimap_img, (0, 80, 120), (10, 255, 255))     # red lows
    enemy2     = _mask_hsv(minimap_img, (170, 80, 120), (180,255, 255))   # red highs
    enemy_mask = cv2.bitwise_or(enemy1, enemy2)
    spike_mask = _mask_hsv(minimap_img, (20, 80, 120), (35, 255, 255))    # yellow/orange spike icon

    # Optional: clean up small noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    ally_mask  = cv2.morphologyEx(ally_mask,  cv2.MORPH_OPEN, k)
    enemy_mask = cv2.morphologyEx(enemy_mask, cv2.MORPH_OPEN, k)
    spike_mask = cv2.morphologyEx(spike_mask, cv2.MORPH_OPEN, k)

    allies = _centers_from_mask(ally_mask)
    enemies= _centers_from_mask(enemy_mask)
    spikes = _centers_from_mask(spike_mask, min_area=8, max_area=3000)

    return {"ally": allies, "enemy": enemies, "spike": spikes}

def normalize_points(points: List[Tuple[int,int]], w: int, h: int) -> List[Tuple[float,float]]:
    return [(round(x/float(w), 6), round(y/float(h), 6)) for (x,y) in points]

# -------------------------------
# Callout lookup (region → name)
# -------------------------------
def load_callout_lookup(map_name: str, mask_dir: str, legend_dir: str):
    mask_path   = os.path.join(mask_dir, f"{map_name}.png")
    legend_path = os.path.join(legend_dir, f"{map_name}.json")
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask is None:
        raise FileNotFoundError(f"Callout mask not found for {map_name}: {mask_path}")
    with open(legend_path, "r") as f:
        legend = json.load(f)  # {"#RRGGBB": "A Heaven", ...}
    # build fast color→name dict
    color_to_name = {}
    for hexc, name in legend.items():
        hexc = hexc.strip().lstrip("#")
        r = int(hexc[0:2],16); g = int(hexc[2:4],16); b = int(hexc[4:6],16)
        color_to_name[(b,g,r)] = name  # OpenCV BGR
    return mask, color_to_name

def callout_at(mask_img: np.ndarray, color_to_name: Dict[Tuple[int,int,int], str], x: int, y: int) -> Optional[str]:
    H, W = mask_img.shape[:2]
    xi = np.clip(int(round(x)), 0, W-1)
    yi = np.clip(int(round(y)), 0, H-1)
    b,g,r = mask_img[yi, xi].tolist()
    return color_to_name.get((b,g,r), None)

# -------------------------------
# ID assignment (simple nearest-neighbor tracker)
# -------------------------------
def assign_ids(prev_pts: Dict[str, Dict[str, Tuple[float,float]]],
               curr_pts: Dict[str, List[Tuple[float,float]]],
               max_dist=0.05) -> Dict[str, Dict[str, Tuple[float,float]]]:
    """
    Greedy association; keep IDs stable over time. Separate pools for ally/enemy.
    prev_pts: {"ally": {"A1":(x,y),...}, "enemy": {"E1":(x,y),...}}
    curr_pts: {"ally":[(x,y),...], "enemy":[(x,y),...]}
    """
    out = {"ally":{}, "enemy":{}}
    for key, prefix in [("ally","P"), ("enemy","E")]:
        used = set()
        # try to match to previous IDs
        for pid, pxy in (prev_pts.get(key, {})).items():
            best, bi = 1e9, -1
            for i,(x,y) in enumerate(curr_pts.get(key, [])):
                if i in used: continue
                d = math.hypot(x-pxy[0], y-pxy[1])
                if d < best:
                    best, bi = d, i
            if bi >= 0 and best <= max_dist:
                out[key][pid] = curr_pts[key][bi]
                used.add(bi)
        # assign new IDs to leftovers
        ctr = 1
        exist = set(out[key].keys())
        for i,(x,y) in enumerate(curr_pts.get(key, [])):
            if i in used: continue
            # find next free increment
            while f"{prefix}{ctr}" in exist: ctr += 1
            out[key][f"{prefix}{ctr}"] = (x,y)
            exist.add(f"{prefix}{ctr}")
            ctr += 1
    return out

# -------------------------------
# Fusion with game status / abilities
# -------------------------------
def fuse_to_events(per_sec_positions: Dict[int, dict],
                   map_name: str,
                   callout_mask: np.ndarray,
                   color_to_name: Dict[Tuple[int,int,int], str],
                   game_status_json: str,
                   ability_json: str) -> List[dict]:
    """
    Build the final per-second records:
    [
      {"timestamp": "0:32", "map": "Ascent", "players":[...], "spike": {...}, "abilities":[...]},
      ...
    ]
    """
    # Load status/abilities keyed by timestamp seconds (your format)
    with open(game_status_json, "r") as f:
        status = json.load(f)  # {sec: {"team": {...}, "enemy": {...}}}
    with open(ability_json, "r") as f:
        abilities = json.load(f)  # {sec: {"Ability #1": "2-2", ...}} or your structure

    def to_mmss(s):
        s = int(s)
        m = max(0, s//60); ss = max(0, s%60)
        return f"{m}:{ss:02d}"

    H, W = callout_mask.shape[:2]
    out = []
    for sec, obj in sorted(per_sec_positions.items(), key=lambda x: x[0]):
        ally = obj["ally_ids"]  # {"P1":(nx,ny),...}
        enemy= obj["enemy_ids"]
        spike= obj["spike_norm"]  # list of (nx,ny)
        # Build players with callouts
        players = []
        for pid,(nx,ny) in ally.items():
            x,y = int(nx*W), int(ny*H)
            loc = callout_at(callout_mask, color_to_name, x, y) or "Unknown"
            players.append({"id": pid, "team":"ally", "agent": None, "pos":[nx,ny], "location": loc})
        for pid,(nx,ny) in enemy.items():
            x,y = int(nx*W), int(ny*H)
            loc = callout_at(callout_mask, color_to_name, x, y) or "Unknown"
            players.append({"id": pid, "team":"enemy","agent": None, "pos":[nx,ny], "location": loc})

        # Spike
        spike_entry = None
        if spike:
            sx, sy = spike[0]
            x,y = int(sx*W), int(sy*H)
            s_loc = callout_at(callout_mask, color_to_name, x, y) or "Unknown"
            spike_entry = {"status": "unknown", "site": None, "pos": [sx, sy], "location": s_loc}

        # Attach ult status/agents if available in your dicts for this second
        if str(sec) in status:
            # try to map agents onto ally IDs in a stable order (if you later learn slot→player mapping)
            pass

        # Abilities at this timestamp → (optional) place them if you later geolocate smokes/flashes
        merged_abilities = []

        out.append({
            "timestamp": to_mmss(sec if sec >= 0 else 0),
            "map": map_name,
            "players": players,
            "spike": spike_entry,
            "abilities": merged_abilities
        })
    return out

# -------------------------------
# High-level driver
# -------------------------------
def run_minimap_pipeline(video_path: str,
                         refs_dir: str,
                         callout_mask_dir: str,
                         callout_legend_dir: str,
                         visualize: bool=False) -> List[dict]:
    """
    Returns a list of merged, per-second dicts.
    """
    frames = extract_frames(video_path, interval=1)
    if not frames: return []

    # 1) Map ID (done on first detected minimap)
    first = frames[0]
    mm0, bbox0 = crop_minimap(first)
    map_name, conf = identify_map(mm0, refs_dir)

    # 2) Load callout mask/legend
    callout_mask, color_to_name = load_callout_lookup(map_name, callout_mask_dir, callout_legend_dir)
    MH, MW = callout_mask.shape[:2]

    # 3) Per-second tracking
    per_sec = {}
    prev_ids = {"ally":{}, "enemy":{}}
    for t, frame in enumerate(frames):
        mm, bb = crop_minimap(frame)
        h, w = mm.shape[:2]
        det = track_minimap_objects(mm)
        ally_norm  = normalize_points(det["ally"], w, h)
        enemy_norm = normalize_points(det["enemy"], w, h)
        spike_norm = normalize_points(det["spike"], w, h)

        assigned = assign_ids(prev_ids, {"ally": ally_norm, "enemy": enemy_norm}, max_dist=0.05)
        prev_ids = assigned

        per_sec[t] = {
            "ally_norm": ally_norm,
            "enemy_norm": enemy_norm,
            "spike_norm": spike_norm,
            "ally_ids": assigned["ally"],
            "enemy_ids": assigned["enemy"],
        }

        if visualize:
            dbg = mm.copy()
            for (x,y) in det["ally"]:
                cv2.circle(dbg,(x,y),5,(0,255,0),-1)
            for (x,y) in det["enemy"]:
                cv2.circle(dbg,(x,y),5,(0,0,255),-1)
            for (x,y) in det["spike"]:
                cv2.circle(dbg,(x,y),6,(0,255,255),2)
            cv2.imshow("minimap-tracking", dbg); cv2.waitKey(1)
    if visualize: cv2.destroyAllWindows()

    # 4) Optional: fuse with existing session JSONs when available (paths shown below)
    # return per_sec to the caller so they can pass the correct JSON paths for this video.
    return per_sec, map_name, callout_mask, color_to_name

# Convenience: end-to-end including fusion, if you know your JSON paths
def run_and_fuse(video_path: str,
                 refs_dir: str,
                 callout_mask_dir: str,
                 callout_legend_dir: str,
                 game_status_json: str,
                 ability_json: str,
                 visualize: bool=False) -> List[dict]:
    per_sec, map_name, callout_mask, color_to_name = run_minimap_pipeline(
        video_path, refs_dir, callout_mask_dir, callout_legend_dir, visualize=visualize
    )
    events = fuse_to_events(per_sec, map_name, callout_mask, color_to_name,
                            game_status_json, ability_json)
    return events

if __name__ == "__main__":
    # Example (edit the paths to your setup):
    events = run_and_fuse(
        video_path="videos/1440-2.mp4",
        refs_dir="assets/minimaps",
        callout_mask_dir="assets/minimap_callout_masks",
        callout_legend_dir="assets/minimap_callout_legend",
        game_status_json="app/json_data/session/v1440_game_status.json",
        ability_json="app/json_data/session/v1440_slot_matches.json",
        visualize=False
    )
    with open("app/json_data/session/v1080_minimap_events.json","w") as f:
        json.dump(events, f, indent=2)