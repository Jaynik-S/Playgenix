# Minimap Pipeline Implementation Plan

This plan defines how to integrate a YOLO minimap detector (`best.pt`) into the existing CV pipeline and produce timestamp-keyed minimap outputs that match the repo's current JSON style.

## 1) Repository integration plan

### Existing integration pattern (cv_pipeline.py)
- `app/cv_files/cv_pipeline.py` is a single orchestrator function (`main`) that:
  - Loads the video and determines resolution.
  - Computes UI ROI boxes using a base 1920x1080 template + scaling.
  - Extracts frames via `extract_frames` (1s interval).
  - Computes per-frame timestamp integers (OCR-based, then decremented each frame).
  - Calls helper functions for detection, builds dicts keyed by timestamps, then writes JSON via `output_utils.save_to_json`.
- Integration is function-based (no classes), with helpers living in `app/cv_files`.

### Proposed minimap integration pattern
- Add a new minimap module with a single entry function, called from `cv_pipeline.main` inside the same frame loop and timestamp flow:
  - `minimap_pipeline.run_minimap(frames, timestamps, config, visualize=False, debug_dump_dir=None) -> dict`
  - The output dict is keyed by the same timestamp strings used in `_game_status` and `_slot_matches`.
- `cv_pipeline.main` should pass the already-decoded frames and timestamps to avoid re-reading the video.
- Add a new JSON writer in `app/cv_files/output_utils.py`:
  - `save_minimap_json(minimap_dict, file_name)` writes `app/session_data/<base_name>_minimap.json`.

### Configuration (paths, thresholds, presets)
- New config file: `app/cv_files/minimap_config.py` or a config dict in `app/cv_files/minimap_pipeline.py`.
- Config keys should include:
  - `MODEL_PATH`: default `assets/models/best.pt` (or `assets/models/best.pt` to avoid collisions).
  - `MINIMAP_ASSETS_DIR`: `assets/minimaps`.
  - `CALLOUT_ASSETS_DIR`: `assets/minimaps_callouts`.
  - `ROI_PRESETS`: base ROI coords for 1920x1080 plus derived presets for 1280x720 and 2560x1440.
  - `YOLO_IMGSZ`, `CONF_THRES`, `IOU_THRES`, `CLASSES_MAP`.
  - `REGISTER_INTERVAL_SEC`, `REGISTER_CONF_MIN`, `REGISTER_MAX_AGE`.
  - `CALL_OUT_SOFT_RADIUS_PX`, `CALL_OUT_DEBOUNCE_FRAMES`.
  - `ENABLE_ROTATION_NORM`, `ENABLE_ALIGNMENT`, `ENABLE_CALLOUTS`.

### Enable/disable flags in cv_pipeline
- Add keyword flags similar to existing debug options:
  - `enable_minimap: bool = False`
  - `minimap_config: dict | None = None`
  - `minimap_visualize: bool = False`
- If `enable_minimap` is False, skip all minimap work and avoid the dependency on `ultralytics`.

## 2) Architecture overview

### High-level flow (text diagram)
```
Video -> extract_frames + timestamps
     -> minimap_roi.detect_roi
     -> minimap_preprocess.preprocess_roi
     -> minimap_detect.yolo_infer
     -> minimap_normalize.normalize_rotation + normalize_scale
     -> minimap_align.align_to_static_map (cached transform)
     -> minimap_coords.roi_to_map_coords
     -> callout_map.assign_callouts
     -> minimap_export.build_timestamp_dict -> output_utils.save_minimap_json
```

### Suggested module structure
- `app/cv_files/minimap_roi.py`        - ROI detection + validation
- `app/cv_files/minimap_preprocess.py` - resize, normalize, optional CLAHE
- `app/cv_files/minimap_detect.py`     - YOLO wrapper
- `app/cv_files/minimap_normalize.py`  - rotation + scale normalization
- `app/cv_files/minimap_align.py`      - static map registration + caching
- `app/cv_files/minimap_coords.py`     - ROI coords -> map coords
- `app/cv_files/callout_map.py`        - callout polygons + point-in-polygon
- `app/cv_files/minimap_export.py`     - output formatting helpers
- `app/cv_files/minimap_pipeline.py`   - entrypoint used by cv_pipeline

## 3) Step-by-step implementation plan

### A) Minimap ROI detection
**Objective**
- Locate the minimap region robustly across 720p/1080p/1440p and per-user HUD customization.

**Recommended algorithm + fallback**
- Primary: per-video auto-detect the minimap ring in the expected corner, then lock it for the rest of the clip.
- Validation: check circular mask coverage and edge intensity along the minimap border.
- Fallback: try ROI presets if detection fails for the first N frames, then retry auto-detect periodically.

**Implementation details**
- Auto-detect strategy:
  - Search the likely corner region (top-left by default, but allow config override).
  - Run edge detection and Hough circle to find a candidate ring.
  - Score candidates by ring edge density and interior texture variance.
  - Select best candidate and cache `roi_bbox` for the video.
- Validation metrics:
  - `edge_ratio` along a thin annulus around the ROI.
  - mean intensity change between inside vs outside the circle.
  - stability of the ROI center across adjacent frames (should not drift).
- Confidence = weighted sum of validation metrics.
- Re-detect trigger: ROI confidence drops below `ROI_CONF_MIN` for K consecutive frames.

**Where to implement**
- `app/cv_files/minimap_roi.py`:
  - `detect_roi(frame, config) -> (roi_bbox, confidence, method)`
  - `detect_roi_video_init(frames, config) -> (roi_bbox, confidence)`
  - `validate_roi(frame, roi_bbox, config) -> confidence`

**Acceptance criteria + test plan**
- ROI detected correctly in 90%+ of frames for sample 720p and 1080p videos.
- Visual overlays saved to debug dir show correct ROI alignment.

**Failure modes + debugging guidance**
- UI scale mismatch: update ROI presets or add additional scale buckets.
- Circular detection false positives: increase Hough constraints, apply edge masking to corner only.
- Missing minimap in menus: allow `roi_bbox=None` and skip that frame.

### B) ROI preprocessing
**Objective**
- Normalize the ROI for stable detection and alignment.

**Recommended algorithm + fallback**
- Resize ROI to `YOLO_IMGSZ` (e.g., 256x256).
- Apply mild denoise + CLAHE on luminance to stabilize contrast.
- Fallback: raw resize only (for speed or if preprocessing hurts).

**Implementation details**  
- Use LAB color space; apply CLAHE on L channel.
- Cache resized ROI for the same frame to avoid repeat work.

**Where to implement**
- `app/cv_files/minimap_preprocess.py`:
  - `preprocess_roi(roi_bgr, config) -> roi_pre`

**Acceptance criteria + test plan**
- YOLO confidence improves or stays stable vs raw resize on a small validation clip.

**Failure modes + debugging guidance**
- Over-contrast causing false positives: reduce CLAHE clipLimit and tileGridSize.

### C) Run YOLO best.pt inference on ROI
**Objective**
- Detect minimap entities with the trained model.

**Recommended algorithm + fallback**
- Use `ultralytics.YOLO` with `model.predict` on preprocessed ROI.
- Fallback: if model fails to load, skip minimap output and log error.

**Implementation details**
- Dependencies: add `ultralytics` and a compatible `torch` build to `requirements.txt`.
- Load model once per run; reuse the model object.
- Map class ids to canonical names using a config mapping:
  - Example: `{"user": "player", "teammate": "teammate", "enemy": "enemy", "enemy_dead": "enemy_dead", "teammate_dead": "teammate_dead"}`
- Output detection list: `[{class, conf, bbox_xyxy, centroid_xy, roi_norm_xy}]`.

**Where to implement**
- `app/cv_files/minimap_detect.py`:
  - `load_model(model_path)`
  - `infer(roi_pre, model, config) -> detections`

**Acceptance criteria + test plan**
- On 20 sampled frames, the player marker is detected in 80%+ frames.
- No crashes when model path missing or invalid.

**Failure modes + debugging guidance**
- Small icons missed: raise `imgsz` or lower `CONF_THRES`.
- Over-detections on noise: increase `CONF_THRES` or filter by box size.

### D) Rotation normalization
**Objective**
- Normalize orientation so detections align to static map assets.

**Recommended algorithm + fallback**
- Primary: estimate heading from the player marker:
  - Crop around "user" detection, threshold in HSV, fit `cv2.minAreaRect`, use angle.
- Detect north-up mode by measuring stability of background edges across consecutive frames.
- Fallback: if heading cannot be estimated, assume north-up (0 deg).

**Implementation details**
- Keep a moving average of heading to smooth jitter.
- If `heading_std` over N frames is near zero, assume north-up.
- Apply rotation to ROI and detections with `cv2.getRotationMatrix2D`.

**Where to implement**
- `app/cv_files/minimap_normalize.py`:
  - `estimate_heading(roi_pre, detections, config) -> heading_deg`
  - `rotate_roi_and_points(roi_pre, detections, heading_deg) -> (roi_rot, detections_rot)`

**Acceptance criteria + test plan**
- Rotation-corrected detections align to static map orientation visually.
- Heading angle is stable across short sequences.

**Failure modes + debugging guidance**
- Arrow too small for orientation: use temporal motion vector as a proxy.
- False heading spikes: clamp angle deltas per frame.

### E) Scale normalization
**Objective**
- Bring the minimap to a canonical scale for alignment.

**Recommended algorithm + fallback**
- Use a fixed target size (e.g., 256x256) after rotation normalization.
- Fallback: use ROI size-derived scale factor if resize is disabled.

**Implementation details**
- Track scale factor from raw ROI to canonical size and apply it to detection coordinates.

**Where to implement**
- `app/cv_files/minimap_normalize.py`:
  - `normalize_scale(roi_rot, detections_rot, target_size) -> (roi_norm, detections_norm)`

**Acceptance criteria + test plan**
- Detection centroids remain stable and proportional after resizing.

**Failure modes + debugging guidance**
- Aspect ratio mismatch: enforce square crop based on circle radius.

### F) Static minimap matching / alignment to static reference
**Objective**
- Align the live minimap background to a static map asset and cache the transform.

**Map ID selection strategy**
- If map id known from user/session metadata, use it directly.
- Otherwise, classify by template matching:
  - Compare edge maps of ROI to each `assets/minimaps/<Map>.png`.
  - Choose map with highest normalized correlation above threshold.

**Registration strategy**
- Primary: edge-based ECC alignment (`cv2.findTransformECC`) on the ROI edge map vs static map edge map.
- Fallback: ORB/AKAZE keypoints + RANSAC homography.
- If both fail, keep last valid transform or skip canonical mapping for that frame.

**Implementation details**
- Compute transform only every `REGISTER_INTERVAL_SEC` or when confidence drops.
- Cache: `transform`, `confidence`, `map_id`, `last_update_ts`.
- Confidence measures:
  - ECC correlation coefficient.
  - Inlier ratio for RANSAC fallback.

**Where to implement**
- `app/cv_files/minimap_align.py`:
  - `select_map_id(roi_edge, map_assets, config) -> (map_id, score)`
  - `register(roi_edge, map_edge, prev_transform, config) -> (transform, confidence)`
  - `should_refresh(prev_conf, age, config) -> bool`

**Acceptance criteria + test plan**
- Transform consistency: consecutive frames should produce <5 px drift on static map.
- Map id selected correctly for known clips.

**Failure modes + debugging guidance**
- Overlays (smokes/pings) hide background: downweight masked areas or use temporal median background.
- Rotation mismatch: ensure rotation normalization is applied before registration.

### G) Convert YOLO detections to canonical map coordinates
**Objective**
- Convert detections from ROI pixel space to static map pixel or normalized coordinates.

**Recommended algorithm + fallback**
- Use cached homography/affine transform to map points via `cv2.perspectiveTransform`.
- Fallback: output ROI-normalized coordinates if transform confidence is low.

**Implementation details**
- Store both ROI normalized coords and map coords in the output.
- Map coords can be normalized to [0..1] based on static map image size.

**Where to implement**
- `app/cv_files/minimap_coords.py`:
  - `roi_to_map(points_roi, transform, map_shape) -> points_map`
  - `normalize_points(points, width, height) -> points_norm`

**Acceptance criteria + test plan**
- Known reference points (e.g., map corners) map consistently across frames.

**Failure modes + debugging guidance**
- Bad transform causes points outside map bounds: clamp or mark as invalid.

### H) Callout assignment
**Objective**
- Assign each detection to a named callout region.

**Recommended algorithm + fallback**
- Primary: point-in-polygon on manually defined callout polygons per map.
- Soft assignment near boundaries: compute distance to polygon edge and choose nearest if within a soft radius.
- Fallback: if callouts unavailable for map, return `callout=None`.

**Implementation details**
- Callout assets (recommended manual authoring):
  - `assets/minimaps_callouts/<Map>.json` with polygon vertices per callout, authored once per map.
  - Create polygons by manually tracing callouts in a simple annotation tool (e.g., a small OpenCV click-to-vertex script or an external polygon editor).
  - Keep vertices in the static map image coordinate system to match registration output.
- Optional alternative:
  - `assets/minimaps_callouts/<Map>.png` color-coded regions + `assets/minimaps_callouts/<Map>_colors.json` mapping colors to callout names.
- Use `cv2.pointPolygonTest` to determine inclusion and distance.
- Debounce: require callout to persist for N frames or use majority vote over a short window.

**Where to implement**
- `app/cv_files/callout_map.py`:
  - `load_callouts(map_id) -> list[CalloutPolygon]`
  - `assign_callout(point, callouts, config) -> (callout, confidence)`
  - `smooth_callouts(track_id, callout) -> callout`

**Acceptance criteria + test plan**
- Callout labels match expected regions on a few manual test points.
- Flicker rate below target threshold (see metrics below).

**Failure modes + debugging guidance**
- Overlapping regions: ensure consistent priority or nearest-boundary logic.
- Wrong map id: callouts will be off; add debug overlay to verify.

### I) Export + save JSON output
**Objective**
- Create a timestamp-keyed JSON output consistent with `_game_status` and `_slot_matches`.

**Recommended format**
- Top-level object keyed by timestamp strings.
- Each value includes:
  - `roi`: bbox + confidence
  - `map_id`, `map_confidence`, `transform_confidence`
  - `entities`: list of detections with class, roi_xy, map_xy, callout, confidence

**Implementation details**
- Use the same timestamp values that `cv_pipeline` already computes.
- Output path: `app/session_data/<video_id>_minimap.json` (matches existing naming).

**Where to implement**
- `app/cv_files/minimap_export.py`:
  - `build_entry(ts, roi, map_info, detections) -> dict`
- `app/cv_files/output_utils.py`:
  - `save_minimap_json(minimap_dict, file_name)`

**Acceptance criteria + test plan**
- JSON keys are strings and match timestamps in existing outputs.
- Output writes without changing existing files.

**Failure modes + debugging guidance**
- Missing timestamps: ensure `timestamps` list aligns with frame sampling.
- Empty detections: still emit an empty list for that timestamp.

## 4) Concrete pseudo-code

### Minimap module entrypoint
```python
def run_minimap(frames, timestamps, config, visualize=False, debug_dump_dir=None):
    model = load_model(config["MODEL_PATH"])
    cache = init_alignment_cache()
    output = {}

    for frame, ts in zip(frames, timestamps):
        roi_bbox, roi_conf, roi_method = detect_roi(frame, config)
        if roi_bbox is None:
            output[str(ts)] = build_entry(ts, None, None, [])
            continue

        roi = crop(frame, roi_bbox)
        roi_pre = preprocess_roi(roi, config)
        dets = infer(roi_pre, model, config)

        heading = estimate_heading(roi_pre, dets, config)
        roi_rot, dets_rot = rotate_roi_and_points(roi_pre, dets, heading)
        roi_norm, dets_norm = normalize_scale(roi_rot, dets_rot, config["YOLO_IMGSZ"])

        map_id = select_map_id_if_needed(roi_norm, cache, config)
        transform, t_conf = align_if_needed(roi_norm, map_id, cache, config)

        dets_map = apply_transform(dets_norm, transform, map_id, t_conf, config)
        dets_callout = assign_callouts(dets_map, map_id, config)

        output[str(ts)] = build_entry(ts, roi_bbox, map_id, dets_callout, roi_conf, t_conf)

    return output
```

### Per-frame loop outline
```python
for frame_idx, frame in enumerate(frames):
    ts = timestamps[frame_idx]

    roi_bbox, roi_conf = detect_roi(frame, config)
    if not roi_bbox:
        output[str(ts)] = build_entry(ts, None, None, [])
        continue

    roi = crop(frame, roi_bbox)
    roi_pre = preprocess_roi(roi, config)
    dets = yolo_infer(roi_pre)
    heading = estimate_heading(roi_pre, dets)
    roi_rot, dets_rot = rotate_roi_and_points(roi_pre, dets, heading)
    roi_norm, dets_norm = normalize_scale(roi_rot, dets_rot)

    if should_refresh_alignment(cache, ts):
        map_id = select_map_id(roi_norm)
        transform, conf = register(roi_norm, map_id)
        cache.update(map_id, transform, conf, ts)

    dets_map = roi_to_map_coords(dets_norm, cache.transform)
    dets_callout = assign_callouts(dets_map, cache.map_id)

    output[str(ts)] = build_entry(ts, roi_bbox, cache.map_id, dets_callout, roi_conf, cache.conf)
```

### cv_pipeline.py integration
```python
from .minimap_pipeline import run_minimap
from .output_utils import save_minimap_json

def main(file_name, visualize=False, debug_dump_dir=None, use_edge_matching=True, enable_minimap=False):
    ...
    frames = extract_frames(video_path, interval=1)
    timestamps = build_timestamp_list(frames, clock_bbox, spike_bbox, visualize)

    if enable_minimap:
        minimap_dict = run_minimap(frames, timestamps, minimap_config, visualize, debug_dump_dir)
        save_minimap_json(minimap_dict, file_name)

    # existing ability + game status pipeline continues unchanged
```

## 5) Testing & evaluation

### Metrics
- ROI detection success rate: `% frames with valid ROI` (target > 95%).
- Alignment confidence stability: std-dev of transform confidence across frames (target low).
- Callout flicker rate: `% of detections that change callout within N frames` (target < 10%).
- Detection precision/recall: measured on a labeled minimap eval set.

### Golden-video regression
- Add `tests/expected_outputs/<video_id>_minimap.json` for a few clips.
- Comparison script:
  - Match timestamps.
  - Compare per-class detection counts with tolerance.
  - Compare callout assignment consistency (majority within a window).
- Report summary metrics:
  - Per-class precision/recall.
  - Mean callout stability.
  - Percent frames with valid map transform.

### Dependencies to add
- `ultralytics` (YOLO inference)
- `torch` and `torchvision` compatible with the environment
- Optional: `shapely` (only if you prefer polygon ops outside OpenCV)

### Required asset formats
- `assets/models/best.pt` (YOLO weights)
- `assets/minimaps/<Map>.png` (static clean reference)
- Callouts:
  - Preferred: `assets/minimaps_callouts/<Map>.json` with polygon vertices
  - Alternate: `assets/minimaps_callouts/<Map>.png` with unique colors + a color map file
