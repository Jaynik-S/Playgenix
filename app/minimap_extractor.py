import os
import cv2
import glob
import json
import math
import numpy as np
from collections import defaultdict

# ===========================
# Utility helpers  https://chatgpt.com/c/6897ecf3-f6e0-8326-b6ce-9bf0d41ffb3c
# ===========================

def imread_any(p):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img


def rotate_bound(image, angle_deg):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR)


def non_max_suppression_fast(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")


# ===========================
# Minimap detection (rotation + scale invariant)
# ===========================
class MapIdentifier:
    def __init__(self, map_ref_dir: str):
        self.map_ref_dir = map_ref_dir
        self.refs = self._load_refs(map_ref_dir)
        self.orb = cv2.ORB_create(nfeatures=4000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _load_refs(self, d):
        items = {}
        for p in glob.glob(os.path.join(d, "*.png")):
            name = os.path.splitext(os.path.basename(p))[0]
            img = imread_any(p)
            items[name] = img
        return items

    def _try_orb_localize(self, frame_gray, ref_gray):
        kp1, des1 = self.orb.detectAndCompute(ref_gray, None)
        kp2, des2 = self.orb.detectAndCompute(frame_gray, None)
        if des1 is None or des2 is None:
            return None
        matches = self.bf.match(des1, des2)
        if len(matches) < 12:
            return None
        matches = sorted(matches, key=lambda x: x.distance)[:100]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None
        h, w = ref_gray.shape[:2]
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        x1, y1 = np.min(dst[:, 0, 0]), np.min(dst[:, 0, 1])
        x2, y2 = np.max(dst[:, 0, 0]), np.max(dst[:, 0, 1])
        return [int(x1), int(y1), int(x2), int(y2)], H

    def detect(self, frame_bgr):
        """Return (map_name, roi_box(x1,y1,x2,y2), H or None).
        Tries ORB + fallback multiscale template matching with rotations.
        """
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        best = None
        best_name = None
        best_H = None

        # 1) ORB + homography (fastest/most robust when it works)
        for name, ref in self.refs.items():
            ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            r = self._try_orb_localize(frame_gray, ref_gray)
            if r is not None:
                box, H = r
                score = (box[2] - box[0]) * (box[3] - box[1])  # area proxy
                if best is None or score > (best[2] - best[0]) * (best[3] - best[1]):
                    best, best_name, best_H = box, name, H
        if best is not None:
            return best_name, best, best_H

        # 2) Fallback: multi-rotation + multi-scale template match on edges
        frame_edges = cv2.Canny(frame_gray, 50, 150)
        for name, ref in self.refs.items():
            ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            for ang in range(0, 360, 15):
                rot = rotate_bound(ref_gray, ang)
                ref_edges = cv2.Canny(rot, 50, 150)
                for scale in [0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]:
                    h, w = ref_edges.shape[:2]
                    th, tw = int(h * scale), int(w * scale)
                    if th < 40 or tw < 40:
                        continue
                    templ = cv2.resize(ref_edges, (tw, th))
                    if templ.shape[0] > frame_edges.shape[0] or templ.shape[1] > frame_edges.shape[1]:
                        continue
                    res = cv2.matchTemplate(frame_edges, templ, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    if max_val < 0.45:
                        continue
                    top_left = max_loc
                    br = (top_left[0] + templ.shape[1], top_left[1] + templ.shape[0])
                    box = [top_left[0], top_left[1], br[0], br[1]]
                    if best is None or max_val > 0.65:
                        best, best_name, best_H = box, name, None
        if best is not None:
            return best_name, best, best_H

        return None, None, None


# ===========================
# Icon segmentation on minimap
# ===========================
class MinimapIconDetector:
    # HSV thresholds tuned for typical VALORANT minimap colors
    ALLY1 = (np.array([80,  40, 120]), np.array([100, 255, 255]))  # cyan/teal
    ENEMY_R1 = (np.array([0,   90, 120]), np.array([10,  255, 255]))
    ENEMY_R2 = (np.array([170, 90, 120]), np.array([180, 255, 255]))
    SPIKE = (np.array([15, 130, 160]), np.array([40, 255, 255]))     # yellow/orange

    def __init__(self):
        pass

    @staticmethod
    def _find_centers(mask, min_area=8, max_area=1200):
        centers = []
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area or a > max_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
        return centers

    def detect(self, minimap_bgr):
        hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)

        ally_mask = cv2.inRange(hsv, self.ALLY1[0], self.ALLY1[1])
        enemy_mask1 = cv2.inRange(hsv, self.ENEMY_R1[0], self.ENEMY_R1[1])
        enemy_mask2 = cv2.inRange(hsv, self.ENEMY_R2[0], self.ENEMY_R2[1])
        enemy_mask = cv2.bitwise_or(enemy_mask1, enemy_mask2)
        spike_mask = cv2.inRange(hsv, self.SPIKE[0], self.SPIKE[1])

        # Clean up
        k = np.ones((3, 3), np.uint8)
        ally_mask = cv2.morphologyEx(ally_mask, cv2.MORPH_OPEN, k)
        enemy_mask = cv2.morphologyEx(enemy_mask, cv2.MORPH_OPEN, k)
        spike_mask = cv2.morphologyEx(spike_mask, cv2.MORPH_OPEN, k)

        allies = self._find_centers(ally_mask)
        enemies = self._find_centers(enemy_mask)
        spikes = self._find_centers(spike_mask, min_area=12, max_area=2500)

        spike_pt = spikes[0] if len(spikes) else None
        return allies, enemies, spike_pt


# ===========================
# Simple tracker (nearest-neighbor)
# ===========================
class SimpleTracker:
    def __init__(self, max_dist=40):
        self.max_dist = max_dist
        self.next_id = 1
        self.tracks = {}  # id -> (x, y)

    def _assign(self, points):
        assigned = {}
        used = set()
        for pid, (px, py) in self.tracks.items():
            best = None
            best_d2 = None
            for i, (x, y) in enumerate(points):
                if i in used:
                    continue
                d2 = (px - x) ** 2 + (py - y) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2, best = d2, i
            if best is not None and best_d2 <= self.max_dist ** 2:
                assigned[pid] = points[best]
                used.add(best)
        # new ids for unassigned
        for i, (x, y) in enumerate(points):
            if i in used:
                continue
            pid = f"P{self.next_id}"
            self.next_id += 1
            assigned[pid] = (x, y)
        self.tracks = assigned
        return assigned


# ===========================
# Callout mapping
# ===========================
class CalloutMapper:
    def __init__(self, map_name: str, callout_dir: str):
        self.map = map_name
        self.dir = callout_dir
        self.mask_img, self.legend = self._load_masks()

    def _load_masks(self):
        # Expect: <map>_mask.png (indexed colors) + <map>_legend.json {"r,g,b":"Label"}
        mask_path = os.path.join(self.dir, f"{self.map}_mask.png")
        legend_path = os.path.join(self.dir, f"{self.map}_legend.json")
        if os.path.exists(mask_path) and os.path.exists(legend_path):
            return imread_any(mask_path), json.load(open(legend_path, "r"))
        # Fallback: try coarse zones from the provided callouts image
        png_path = os.path.join(self.dir, f"{self.map}_callouts.png")
        if os.path.exists(png_path):
            img = imread_any(png_path)
            return img, None
        return None, None

    def label_at(self, x, y):
        if self.mask_img is None:
            return None
        h, w = self.mask_img.shape[:2]
        x = np.clip(int(x), 0, w - 1)
        y = np.clip(int(y), 0, h - 1)
        px = self.mask_img[y, x]
        key = f"{int(px[2])},{int(px[1])},{int(px[0])}"
        if self.legend is not None and key in self.legend:
            return self.legend[key]
        # Heuristic fallback for non-indexed callout images: coarse thirds + A/B sites from yellow overlays
        hsv = cv2.cvtColor(self.mask_img, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array([20, 30, 120]), np.array([45, 255, 255]))
        cnts, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 600]
        boxes = sorted(boxes, key=lambda b: b[0])
        label = None
        if len(boxes) >= 2:
            # Left ~ B Site, Right ~ A Site for most maps like Abyss
            bx, by, bw, bh = boxes[0]
            ax, ay, aw, ah = boxes[-1]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                label = "B Site"
            elif ax <= x <= ax + aw and ay <= y <= ay + ah:
                label = "A Site"
        if label is None:
            # Coarse thirds
            third = self.mask_img.shape[1] // 3
            if x < third:
                label = "B Side"
            elif x < 2 * third:
                label = "Mid"
            else:
                label = "A Side"
        return label


# ===========================
# Main Minimap tracker
# ===========================
class MinimapTracker:
    def __init__(self, map_ref_dir: str, callout_dir: str):
        self.identifier = MapIdentifier(map_ref_dir)
        self.iconer = MinimapIconDetector()
        self.callout_dir = callout_dir
        self.callout_mapper = None
        self.roi = None  # (x1,y1,x2,y2)
        self.map_name = None
        self.ally_tracker = SimpleTracker(max_dist=55)
        self.enemy_tracker = SimpleTracker(max_dist=55)

    def normalize(self, x, y):
        x1, y1, x2, y2 = self.roi
        return (x - x1) / (x2 - x1), (y - y1) / (y2 - y1)

    def _crop_roi(self, frame):
        x1, y1, x2, y2 = self.roi
        return frame[y1:y2, x1:x2]

    def initialize(self, first_frame_bgr):
        name, box, H = self.identifier.detect(first_frame_bgr)
        if name is None:
            # As an ultra-safe fallback, try standard top-left HUD boxes per res
            h, w = first_frame_bgr.shape[:2]
            if (w, h) == (2560, 1440):
                box = (18, 18, 18 + 420, 18 + 420)
                name = "Unknown"
            elif (w, h) == (1920, 1080):
                box = (14, 14, 14 + 330, 14 + 330)
                name = "Unknown"
            elif (w, h) == (1280, 720):
                box = (9, 9, 9 + 220, 9 + 220)
                name = "Unknown"
            else:
                raise RuntimeError("Could not localize minimap ROI and no resolution fallback available.")
        self.roi = tuple(map(int, box))
        self.map_name = name
        self.callout_mapper = CalloutMapper(name, self.callout_dir)
        return self.map_name, self.roi

    def step(self, frame_bgr, timestamp_str, game_status=None, ability_events=None):
        if self.roi is None:
            raise RuntimeError("Call initialize() first with the first frame.")
        roi_img = self._crop_roi(frame_bgr)
        allies, enemies, spike_pt = self.iconer.detect(roi_img)
        # Track
        ally_tracked = self.ally_tracker._assign(allies)
        enemy_tracked = self.enemy_tracker._assign(enemies)

        def pack_players(tracks, team):
            out = []
            for pid, (x, y) in tracks.items():
                nx, ny = self.normalize(self.roi[0] + x, self.roi[1] + y)
                # map to callout pixel in callout space (assume callout image matches ROI aspect)
                label = None
                if self.callout_mapper is not None and self.callout_mapper.mask_img is not None:
                    ch, cw = self.callout_mapper.mask_img.shape[:2]
                    # project normalized coord into callout pixel grid
                    cx, cy = int(nx * cw), int(ny * ch)
                    label = self.callout_mapper.label_at(cx, cy)
                out.append({
                    "id": pid,
                    "team": team,
                    "agent": None,
                    "pos": [round(float(nx), 4), round(float(ny), 4)],
                    "location": label,
                })
            return out

        players = pack_players(ally_tracked, "ally") + pack_players(enemy_tracked, "enemy")

        spike_obj = None
        if spike_pt is not None:
            nx, ny = self.normalize(self.roi[0] + spike_pt[0], self.roi[1] + spike_pt[1])
            site = None
            if self.callout_mapper is not None and self.callout_mapper.mask_img is not None:
                ch, cw = self.callout_mapper.mask_img.shape[:2]
                site = self.callout_mapper.label_at(int(nx * cw), int(ny * ch))
            spike_obj = {"status": "dropped", "site": site, "pos": [round(float(nx), 4), round(float(ny), 4)]}
        elif game_status and game_status.get("spike", {}).get("status") == "planted":
            spike_obj = {"status": "planted", "site": game_status["spike"].get("site")}

        rec = {
            "timestamp": timestamp_str,
            "map": self.map_name,
            "players": players,
            "spike": spike_obj,
            "abilities": ability_events or [],
        }
        return rec


# ===========================
# Public API
# ===========================

def run_minimap(video_path: str,
                map_ref_dir: str,
                callout_dir: str,
                per_second_frames: list = None,
                game_status_by_ts: dict | None = None,
                ability_events_by_ts: dict | None = None,
                visualize: bool = False):
    """Return list of timeline dicts (one per second). If per_second_frames is None,
    frames are sampled at 1Hz internally.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    stamps = []
    if per_second_frames is None:
        # Sample at 1Hz
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % int(max(1, round(fps))) == 0:
                frames.append(frame)
                # naive hh:mm:ss -> we just store seconds index for now; upstream can pass exact timestamp strings
                stamps.append(str(len(stamps)))
            idx += 1
    else:
        frames = per_second_frames
        stamps = [str(i) for i in range(len(per_second_frames))]
    cap.release()

    if not frames:
        return []

    mt = MinimapTracker(map_ref_dir, callout_dir)
    mt.initialize(frames[0])

    out = []
    for i, frame in enumerate(frames):
        ts = stamps[i]
        status = game_status_by_ts.get(ts) if game_status_by_ts else None
        ab = ability_events_by_ts.get(ts) if ability_events_by_ts else None
        rec = mt.step(frame, ts, game_status=status, ability_events=ab)
        out.append(rec)
    return out


if __name__ == "__main__":
    # Quick local sanity (expects Abyss.png and Abyss_callouts.png in the folder provided at runtime)
    demo_video = "videos/v1440.mp4"  # replace with an actual round video
    maps_dir = "assets/minimaps"  # e.g., "assets/maps"
    callouts_dir = "assets/callouts"  # e.g., "assets/callouts"
    if os.path.exists(demo_video):
        results = run_minimap(demo_video, maps_dir, callouts_dir)
        print(json.dumps(results[:2], indent=2))
    else:
        print("[minimap_cv] Demo: place a video at demo.mp4 to smoke-test the module.")
