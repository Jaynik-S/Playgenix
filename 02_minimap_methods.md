## 1) Problem framing

Extracting minimap-derived signals from VALORANT gameplay video is hard because the minimap is a small, stylized, frequently-animated UI element that compresses 3D motion into a tiny 2D representation.

Key challenges:
- **Low effective resolution:** At 720p/1080p the minimap ROI is small; icons can be only a few pixels wide after scaling/compression.
- **Rotation + orientation:** The minimap can rotate (player-facing-up vs north-up settings), and the player arrow rotates continuously.
- **Dynamic clutter:** Pings, revealed enemies, ability indicators, spike markers, and scan effects appear/disappear and overlap.
- **Occlusion + overlays:** Smoke/scan UI effects, map names/callouts, and team pings can partially occlude icons.
- **Compression artifacts:** YouTube/Twitch re-encodes produce ringing/blocking that destroys thin outlines.
- **Color/skin variation:** Different HUD themes, colorblind modes, team colors, and brightness/gamma shifts change icon appearance.
- **HUD scale settings vary:** The minimap size/position can shift with UI scale/aspect ratio; ROI must be detected robustly.
- **Camera transitions:** Spectator swaps, death cam, or replay overlays can change/minimize the minimap.

Assumptions (explicitly define for the extractor):
- Inputs are typical recordings: **1280×720 or 1920×1080**, 30–60fps.
- The minimap is **present and roughly fixed** in one corner during active gameplay (but may disappear during menus/replays).
- **HUD scale may vary**; ROI detection must handle at least a few common scales.
- No game API access; detect only what’s **visible on the minimap** (no inferred wallhack info).
- You have (or can obtain) static reference images of maps (optional but highly beneficial for registration-based methods).

---

## 2) Methods (at least 6 distinct approaches)

### Method 1: Classic CV heuristics + template matching
**Core idea:** Detect minimap ROI, then detect icons via color thresholding + edge/shape cues + template matching for known markers.
**How it works (steps):**
1. Locate minimap ROI (fixed bbox per resolution OR detect circular/square minimap frame with edge/Hough).
2. Normalize ROI (resize to canonical size, denoise, contrast normalize).
3. Build masks for known icon colors (team color, enemy ping red, spike icon color), optionally in HSV.
4. Find connected components/blobs; filter by size/aspect.
5. For each blob, classify by template match (normalized cross-correlation) against a small library (player arrow, ping markers, spike).
6. Convert blob centroid to minimap coordinates; apply temporal smoothing to reduce flicker.
**What it can detect:**
- my position: **yes (limited)** via detecting the **player arrow** centroid; orientation from arrow direction if visible.
- teammates: **maybe** if teammate triangles are distinct and consistent in color/shape.
- enemy pings: **yes** if ping markers have strong color/shape cues (high variance across settings).
- minimap abilities: **limited**; only strong, consistent markers (e.g., scan/reveal pings) if visually distinct.
**Data requirements:** none to small (templates captured from your own clips/settings).
**Engineering effort:** Medium
**Runtime cost:** Low
**Expected accuracy:** Medium for stable settings; drops sharply with HUD/theme changes and low bitrate.
**Pros:**
- Fast to implement; no training pipeline required.
- Easy to debug visually (overlay masks and detections).
**Cons / failure modes:**
- Brittle to UI scale, colorblind mode, stream compression.
- Template matching fails under rotation/scale changes unless you maintain many variants.
**Best use case:**
- MVP prototypes; controlled capture settings (your own recordings, consistent HUD).
**Implementation notes (practical):**
- ROI detection/cropping strategy: start with fixed bbox per resolution + optional auto-detect (frame corner search for minimap border).
- rotation handling: handle icon rotation by matching in edge-space and/or rotating templates at a few angles (coarse).
- coordinate system mapping: output normalized `[0..1]` coords in minimap ROI; later map to world coords via registration (Method 5).
- temporal smoothing / tracking: simple nearest-neighbor association + exponential smoothing; debounce new pings by requiring 2–3 frames.

### Method 2: Object detection on minimap icons (YOLO-style)
**Core idea:** Train an object detector directly on minimap ROI frames to detect icons (player, teammates, pings, spike).
**How it works (steps):**
1. Detect and crop minimap ROI; resize to a fixed input size (e.g., 256×256).
2. Label bounding boxes for classes: `player`, `teammate`, `enemy_ping`, `spike`, `reveal_ping`, etc.
3. Train a small detector (YOLOv8n/YOLOv5n/RT-DETR tiny) on minimap crops with heavy compression/scale augmentations.
4. At inference, run detector per sampled frame; output boxes + confidence.
5. Convert box centers to minimap coordinates; track identities over time (Method 4) for stability.
**What it can detect:**
- my position: **yes** via `player` class (and optionally arrow head direction if you label keypoints or predict angle).
- teammates: **yes** via `teammate` class (IDs optional; you can track by motion).
- enemy pings: **yes** via `enemy_ping` class (including different ping subtypes if labeled).
- minimap abilities: **maybe** if you define classes for consistent indicators and have labels.
**Data requirements:** small to large (practically: **1k–10k labeled minimap crops** depending on class variety).
**Engineering effort:** High
**Runtime cost:** Medium (small model can be real-time on GPU; slower on CPU depending on FPS).
**Expected accuracy:** Medium→High if trained on diverse HUD settings and compression; biggest dependency is label quality/diversity.
**Pros:**
- More robust than templates across scale/rotation/compression.
- Extensible: add new icon classes by labeling more data.
**Cons / failure modes:**
- Requires a labeling workflow; class imbalance is common (enemy pings are rare).
- Detector can hallucinate small icons in compression noise if augmentations are wrong.
**Best use case:**
- Production-ready detection across varied clips once you can invest in labeling.
**Implementation notes (practical):**
- ROI detection/cropping strategy: auto-detect minimap bbox once per video, then keep it fixed unless UI changes.
- rotation handling: the detector learns rotation; still include random rotation augmentation (small angles).
- coordinate system mapping: output normalized coords + per-icon type; optionally predict map orientation setting (rotating vs fixed).
- temporal smoothing / tracking: pair with a tracker (Method 4) for stable positions and to reduce false positives.

### Method 3: Segmentation (pixel-wise minimap parsing)
**Core idea:** Train a segmentation model to classify each pixel in the minimap ROI into classes (background, player marker, teammate marker, ping markers, ability indicators).
**How it works (steps):**
1. Crop minimap ROI; resize to fixed size (e.g., 256×256).
2. Create pixel masks for classes (semi-automatic labeling can help: initial thresholds + manual cleanup).
3. Train a lightweight U-Net/DeepLabv3+ segmentation model.
4. Post-process masks: connected components → centroids → per-instance detections.
5. Track centroids over time; compute confidences from mask probabilities + component stability.
**What it can detect:**
- my position: **yes** if player marker is segmentable (often easiest).
- teammates: **yes** if teammate markers are consistent.
- enemy pings: **yes** if ping markers segment cleanly.
- minimap abilities: **best chance** among methods to capture varied indicators if labeled as separate classes.
**Data requirements:** medium to large (pixel masks are costly; can be reduced with weak supervision).
**Engineering effort:** High
**Runtime cost:** Medium→High (segmentation per frame is heavier than detection at same input size).
**Expected accuracy:** High for “is something present here” tasks; instance separation can be tricky when markers overlap.
**Pros:**
- Strong for tiny/thin icons where boxes are unstable; uses full pixel evidence.
- Can handle overlapping/irregular shapes better than box detectors.
**Cons / failure modes:**
- Labeling is expensive; post-processing to instances can be brittle.
- Over-segmentation under compression artifacts unless trained with realistic noise.
**Best use case:**
- “Best in class” parsing when you want abilities/pings beyond a few templates.
**Implementation notes (practical):**
- ROI detection/cropping strategy: stable ROI is critical; mis-crop ruins masks.
- rotation handling: include rotation augmentation; optionally predict a rotation field (advanced).
- coordinate system mapping: centroids in normalized minimap coords; later map to world coords with Method 5.
- temporal smoothing / tracking: component-level tracking; merge/split handling when icons overlap.

### Method 4: Tracking-first (Kalman / optical flow + periodic detections)
**Core idea:** Use a tracker to maintain stable icon trajectories; run heavier detection intermittently and track in-between frames.
**How it works (steps):**
1. Run an initial detector (Method 1/2/3) to get icon positions at time `t0`.
2. For each icon, initialize a Kalman filter state `(x, y, vx, vy)` in minimap coordinates.
3. Between detection frames, update state using constant-velocity motion model; optionally refine with optical flow (Lucas-Kanade) on small patches.
4. Every N frames, re-detect and use detections as measurements; do association with Hungarian/greedy matching.
5. Output smoothed tracks with confidence based on innovation/error and detection confidence.
**What it can detect:**
- my position: **yes** (track a single target is easier and reliable).
- teammates: **yes** (if detections are good enough to initialize/refresh).
- enemy pings: **yes** (track pings as short-lived “events”; even 1–2 frames can be stabilized).
- minimap abilities: **maybe** (track only if indicators persist long enough and are detectable).
**Data requirements:** depends on detector (none if Method 1; labeled data if Method 2/3).
**Engineering effort:** Medium→High
**Runtime cost:** Low→Medium (tracking is cheap; detection cost amortized).
**Expected accuracy:** Improves stability and reduces flicker; cannot fix systematic detector bias.
**Pros:**
- Greatly reduces jitter and false-positive flicker.
- Lets you sample at lower detection FPS while maintaining smooth output.
**Cons / failure modes:**
- If the detector misses for too long, tracker drifts.
- Identity switches can happen for clustered teammate icons.
**Best use case:**
- Production pipeline where stability matters more than per-frame perfect detection.
**Implementation notes (practical):**
- ROI detection/cropping strategy: ROI must be stable; otherwise all tracks drift.
- rotation handling: tracking works in ROI coordinates; if minimap rotates, apparent motion includes rotation—either stabilize rotation (Method 5) or accept additional noise.
- coordinate system mapping: track in normalized ROI coords; map later.
- temporal smoothing / tracking: Kalman + gating thresholds; “track confidence” decays when not updated by detections.

### Method 5: Map registration (align minimap to known static map image)
**Core idea:** Estimate a transform from the minimap background to a known static map image so icon coordinates can be mapped to consistent “world map” coordinates.
**How it works (steps):**
1. Maintain a library of static map reference images (one per VALORANT map, in a canonical orientation).
2. Crop minimap ROI and preprocess to emphasize structure (edges, walls, corridors).
3. Estimate alignment:
   - Option A: feature matching (ORB/SIFT-like) + RANSAC homography (often hard due to small ROI).
   - Option B: phase correlation / ECC alignment on edge maps (works better if minimap is stable).
   - Option C: learn a small “registration network” that predicts map_id + rotation + scale + translation (supervised).
4. Once aligned, convert detected icon positions from ROI coords → canonical map coords.
5. Use canonical map coords for tracking, heatmaps, and callout mapping.
**What it can detect:**
- my position: **yes** (after you detect player marker by any method, you can place it on canonical map).
- teammates: **yes** (same as above).
- enemy pings: **yes** (maps to canonical coords).
- minimap abilities: **yes** if detected; registration improves interpretability (where on map).
**Data requirements:** none→medium (classic alignment needs reference maps; learned registration needs labeled alignments).
**Engineering effort:** High
**Runtime cost:** Medium (alignment per second or per round; cache transform and update occasionally).
**Expected accuracy:** Medium→High when alignment locks; biggest dependencies: minimap style, presence of a clear background, rotation setting.
**Pros:**
- Converts noisy ROI coordinates into meaningful map space (enables coaching: rotations, site control, timings).
- Helps tracking by removing global rotation/scale drift.
**Cons / failure modes:**
- Alignment can fail under heavy overlays (smokes/scan) or when minimap is rotated with little stable texture.
- Different minimap “skins” or colorblind settings may require multiple reference styles or normalization.
**Best use case:**
- Mid/late-stage system when you want map-aware coaching and location-based rules.
**Implementation notes (practical):**
- ROI detection/cropping strategy: strict cropping; include a small border so registration can use frame edges.
- rotation handling: explicitly estimate rotation (player-up vs north-up) and compensate.
- coordinate system mapping: define canonical map coordinate system (e.g., pixels in reference map, or normalized [0..1]).
- temporal smoothing / tracking: smooth the estimated transform over time (don’t let alignment jump).

### Method 6: Hybrid “detect + register + track” (recommended architecture)
**Core idea:** Use a lightweight detector for icons, a periodic map registration to canonical coords, and a tracker to stabilize trajectories and handle missed detections.
**How it works (steps):**
1. Find minimap ROI once (or re-detect on scene changes).
2. Run a small icon detector (Method 2) or segmentation-derived centroids (Method 3) at 2–5 Hz.
3. Run map registration (Method 5) at low rate (e.g., once per round / when confidence drops) to get transform `T`.
4. Convert detections into canonical map coords with `T`.
5. Track entities in canonical coords with a Kalman filter (Method 4) and output per-timestamp smoothed positions + confidence.
**What it can detect:**
- my position: **yes** (detector + track; highest priority target).
- teammates: **yes** (detector + track; ID may be “track_id” unless you add extra cues).
- enemy pings: **yes** (event detection + short track; include type if classified).
- minimap abilities: **maybe→yes** depending on detector/segmentation classes and labels.
**Data requirements:** small→large (depends on whether detector/registration are learned).
**Engineering effort:** High
**Runtime cost:** Medium
**Expected accuracy:** Best overall when tuned; robust to flicker and gives map-aware outputs; depends on registration confidence.
**Pros:**
- Balances robustness, interpretability, and stability.
- Degrades gracefully: if registration fails, you can still output ROI-normalized coords.
**Cons / failure modes:**
- More moving parts; requires careful confidence propagation and fallback logic.
- Labeling effort if you go learned detector/registration.
**Best use case:**
- Production system where you want reliable trajectories + map context for coaching.
**Implementation notes (practical):**
- ROI detection/cropping strategy: detect once, lock bbox; re-detect only on UI/layout change.
- rotation handling: detect minimap mode; if rotating minimap, treat the transform as time-varying and smooth it.
- coordinate system mapping: always store both ROI coords and canonical coords; canonical only when registration confidence high.
- temporal smoothing / tracking: maintain per-entity track confidence; decay confidence when detector misses.

Optional additional approach (useful when you can’t label much):

### Method 7: Self-supervised “player marker” localization + weak supervision
**Core idea:** Learn to localize the player marker by exploiting its temporal consistency (it’s almost always present) and using weak labels from heuristics.
**How it works (steps):**
1. Use classic heuristics to propose noisy player-marker locations on many frames.
2. Train a small CNN to predict a heatmap for player location (teacher-student / bootstrapping).
3. Refine with tracking and reject outliers using temporal consistency loss.
**What it can detect:**
- my position: **yes (primary target)**.
- teammates: **no** unless extended similarly.
- enemy pings: **no** unless extended similarly.
- minimap abilities: **no** unless extended similarly.
**Data requirements:** small labeled set + large unlabeled set (weak labels from heuristics).
**Engineering effort:** Medium→High
**Runtime cost:** Low→Medium
**Expected accuracy:** Medium→High for player marker only; depends on quality of weak labels and UI consistency.
**Pros:**
- Reduces manual labeling; focuses on the highest-value, always-present signal (your position).
**Cons / failure modes:**
- Can learn heuristic bias; struggles when UI differs from the weak-label source domain.
**Best use case:**
- Fast path to reliable “my position” without full multi-class labeling.
**Implementation notes (practical):**
- ROI detection/cropping strategy: stable ROI required.
- rotation handling: include rotation augmentation; output in ROI coords then map if available.
- coordinate system mapping: same as others; can be combined with Method 5 later.
- temporal smoothing / tracking: integrate with Kalman filter to stabilize.

---

## 3) Recommended path

Staged plan (MVP → improved → best in class):

**MVP (1–2 weeks, low risk)**
- Start with **Method 1** to detect minimap ROI + player marker + enemy ping blobs.
- Add **Method 4** tracking for stability (player marker first; pings as events).
- Output only **ROI-normalized coordinates** with confidence; no map registration yet.

**Improved (2–6 weeks, higher value)**
- Train a small **YOLO-style detector (Method 2)** for: player, teammate marker, enemy ping, spike marker (if visible).
- Keep **Method 4** tracking to reduce flicker and enable lower detection FPS.
- Begin **Method 5** registration prototype for 1–2 maps you have reference images for; only enable when confidence is high.

**Best in class (6+ weeks, scalable)**
- Move to **Hybrid (Method 6)**: detector + periodic registration + tracking, with robust fallbacks.
- If you need rich minimap ability indicators, consider **Segmentation (Method 3)** for a broader set of classes.
- Build a dataset strategy: collect diverse HUD settings (colorblind, scale, brightness), plus re-encoded videos.

What to prototype first and why:
- Prototype **player marker localization + tracking** first: it’s the most valuable and most consistently visible signal.
- Next, add **enemy ping detection** (often visually distinct and directly coaching-relevant: “enemy revealed here → rotate/hold”).
- Only then invest in map registration and multi-entity teammate tracking, which are higher complexity but unlock map-aware coaching.

---

## 4) Output schema suggestion

Per-timestamp minimap outputs (store both ROI coords and (optional) canonical map coords when registration succeeds):

```json
{
  "video_id": "string",
  "minimap": {
    "roi": { "x": 0, "y": 0, "w": 0, "h": 0, "confidence": 0.0 },
    "map_id": "string|null",
    "map_transform": {
      "type": "homography|affine|none",
      "matrix": [[0,0,0],[0,0,0],[0,0,0]],
      "confidence": 0.0
    }
  },
  "timestamps": [
    {
      "t": 83,
      "my_pos": {
        "roi_xy": [0.0, 0.0],
        "map_xy": [0.0, 0.0],
        "heading_deg": 0.0,
        "confidence": 0.0
      },
      "teammates": [
        {
          "track_id": "t1",
          "roi_xy": [0.0, 0.0],
          "map_xy": [0.0, 0.0],
          "confidence": 0.0
        }
      ],
      "enemy_pings": [
        {
          "roi_xy": [0.0, 0.0],
          "map_xy": [0.0, 0.0],
          "type": "revealed|ping|scan|unknown",
          "confidence": 0.0
        }
      ],
      "minimap_abilities": [
        {
          "roi_xy": [0.0, 0.0],
          "map_xy": [0.0, 0.0],
          "ability_type": "recon|smoke|ult_zone|unknown",
          "confidence": 0.0
        }
      ],
      "spike": {
        "visible": false,
        "roi_xy": [0.0, 0.0],
        "map_xy": [0.0, 0.0],
        "state": "carried|planted|dropped|unknown",
        "confidence": 0.0
      }
    }
  ]
}
```

Notes:
- `roi_xy` are normalized to `[0..1]` within the minimap ROI (stable even without map registration).
- `map_xy` are normalized to `[0..1]` in canonical map space (only valid when `map_transform.confidence` is high).
- `track_id` is a stable identifier from the tracker; if you later add identity cues, you can attach agent/name.
*** End PatchResponse file: 02_minimap_methods.md
