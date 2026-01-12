# Minimap Pipeline Implementation Plan (v2)

## 0) Goals, constraints, and non-goals

### Goals
- Produce `app/session_data/<video_id>_minimap.json` keyed by the same timestamp keys as existing outputs (`*_game_status.json`, `*_slot_matches.json`).
- Robustly extract minimap entities from video-only input using `ultralytics` YOLO weights (`assets/models/best.pt`) + per-map static assets (`assets/minimaps/*.png`, `assets/minimaps_callouts/*`).
- Be resilient to UI scale/aspect ratio differences, overlays (pings/smokes), and gameplay states where the minimap disappears/moves (death, spectator, menu).
- Meet throughput targets via multi-rate processing + batch inference (avoid running full pipeline on every video frame).

### Constraints
- Python + OpenCV + `ultralytics` (YOLO). No Riot APIs.
- Must integrate into `app/cv_files/cv_pipeline.py` as an optional module and write JSON alongside other outputs.

### Non-goals (explicit)
- Perfect world-coordinate reconstruction (only minimap-space + static-map-space).
- Full multi-object identity tracking for all entities (basic temporal smoothing is sufficient for stability).

## 1) Repository integration and module layout

### Current orchestration pattern
- `app/cv_files/cv_pipeline.py` extracts frames at 1 Hz (`extract_frames(interval=1)`) and creates timestamp keys by decrementing game-clock seconds.
- JSON persistence is centralized in `app/cv_files/output_utils.py`.

### Integration approach (minimally invasive)
- Add a minimap module that can run at internal multi-rate sampling without changing existing 1 Hz ability/status extraction.
- `cv_pipeline.main(...)` responsibilities:
  - Keep existing 1 Hz extraction and timestamp generation.
  - Compute and pass a `timebase` (timestamp mapping) into the minimap pipeline so minimap output keys match existing outputs even if minimap sampling differs.
  - Gate minimap logic behind `enable_minimap`.

### Proposed entrypoint (function-based, repo-consistent)
- `app/cv_files/minimap_pipeline.py`
  - `run_minimap(video_path: str, timebase: dict, config: dict, visualize: bool = False, debug_dump_dir: str | None = None) -> dict`

### Proposed file layout (all under `app/cv_files/`)
- `minimap_pipeline.py`         Orchestrator (sampling, batching, state machines, JSON assembly)
- `minimap_timebase.py`         Timestamp mapping helpers (reuses cv_pipeline’s clock + spike decisions)
- `minimap_roi.py`              ROI presets + validator + relock logic
- `minimap_preproc.py`          Lightweight preprocessing + CLAHE gating + A/B selection
- `minimap_yolo.py`             Model load + batched inference + per-class thresholding/logging
- `minimap_alignment.py`        Map-id selection + registration + transform cache + overlay masking
- `minimap_callouts.py`         Callout lookup (mask/polygon), soft assignment, adjacency constraints, smoothing
- `minimap_export.py`           Schema builders + debug summaries

### Output persistence
- Extend `app/cv_files/output_utils.py` with:
  - `save_minimap_json(minimap_dict: dict, file_name: str) -> None`
  - Output path: `app/session_data/<base_name>_minimap.json`

## 2) Timestamp strategy (must match existing outputs)

### Problem
- Existing pipeline uses a *game clock* timestamp key that decrements once per extracted frame (1 Hz), and switches to negative values after spike plant.
- Minimap should support internal sampling >1 Hz without changing the key style.

### Timebase contract (computed in `cv_pipeline.py`, reused by minimap)
- `timebase = {`
  - `"clock_at_video_start_sec"`: int, game-clock seconds at video t=0 (this equals the timestamp key of the first 1 Hz frame)
  - `"spike_planted_at_video_sec"`: int | None, wall-clock second when spike plant is first detected in the 1 Hz loop
  - `"frame_interval_sec"`: int, currently `1` (cv_pipeline sampling interval)
- `}`

### Mapping from video time to timestamp key
- Define `second_bin = floor(video_time_sec)` (all samples within the same wall-clock second share the same output key).
- If `spike_planted_at_video_sec is None` or `video_time_sec < spike_planted_at_video_sec`:
  - `ts_key = clock_at_video_start_sec - second_bin`
- Else:
  - `ts_key = -1 - (second_bin - spike_planted_at_video_sec)`

### Output cadence
- Emit exactly one JSON entry per integer timestamp key that exists in the 1 Hz baseline timeline.
- Internally, aggregate multiple minimap samples into that timestamp entry (median/majority vote + stability scoring).

## 3) Performance / throughput plan (multi-rate + batching)

### Rates (defaults; tune with benchmarking)
- **Output rate:** 1 Hz (timestamp keys)
- **ROI validation:** 5 Hz (cheap; downsampled corner crop)
- **ROI relock search:** 0.5 Hz when unlocked, plus immediate relock attempt on validator failures
- **YOLO detection (base):** 2 Hz (every 0.5s) while ROI is locked
- **YOLO detection (burst):** 10 Hz for 2 seconds after a “ping-like” event trigger (see below)
- **Map-id voting:** 0.2 Hz (every 5s) until confident; then only on invalidation
- **Registration refresh:** 0.2 Hz (every 5s) + immediate refresh on ROI relock or confidence drop
- **Callout smoothing:** per output timestamp (1 Hz) using a window of the internal samples

### Event-driven FPS increases (ping trigger)
- Trigger burst detection if **any** of:
  - YOLO detects a `ping/*` class at base rate with `conf >= 0.35` (if ping classes exist).
  - Lightweight pixel-change detector on ROI (downsampled abs-diff) exceeds threshold:
    - `mean_absdiff(gray_64x64) >= 12` for 2 consecutive samples.

### YOLO batch inference (mandatory)
- Batch ROI crops for inference:
  - Accumulate up to `BATCH_SIZE` crops before calling `model.predict(...)`.
  - `BATCH_SIZE` defaults:
    - GPU: `16` (or `32` if VRAM allows)
    - CPU: `4`
- Expected speedup estimate (vs per-crop calls):
  - GPU: **~3–6×** throughput improvement (amortizes Python/IO overhead + improves kernel occupancy on small 256–320px inputs).
  - CPU: **~1.2–2×** improvement (less overhead, some vectorization; dominated by model compute).

### CLAHE cost bounding
- Default: **CLAHE off** (avoid distribution shift for YOLO).
- If enabled, run CLAHE only when needed:
  - Compute luminance contrast on resized ROI (`256x256`): `std(L)`
  - Apply CLAHE only if `std(L) < 18` (low contrast) and only once per output second (cache the CLAHE result for all internal samples that map to the same `ts_key`).
- Cheap alternatives (prefer before CLAHE):
  - Gamma correction (`gamma=0.8` if too dark, `gamma=1.2` if too bright based on mean L)
  - `cv2.normalize` on luminance (linear)

### Memory strategy (avoid storing full-frame buffers)
- Minimap pipeline reads video via `cv2.VideoCapture` and immediately crops ROI (or corner search region).
- Only retain:
  - ROI crops for the current inference batch
  - Per-timestamp aggregation buffers (small dicts of summary stats)

## 4) ROI strategy (presets + validator + relock)

### Requirements
- No single point of failure: ROI detection must be recoverable, validated, and tolerant of minimap disappearance.

### ROI presets (multi-bucket)
- Store ROI presets as **normalized** boxes (fractions of frame width/height) to generalize across resolutions.
- Buckets:
  - `resolution_bucket`: `(aspect_ratio ~ 16:9)` plus explicit `(w,h)` overrides for common modes
  - `ui_scale_bucket`: inferred from best-matching preset size (small/medium/large)
  - `corner`: top-left / top-right (configurable; allow search across both if unknown)

### ROI lock state machine
- States: `UNLOCKED`, `LOCKING`, `LOCKED`, `MISSING`
- Per-sample validator outputs: `roi_conf in [0,1]`, `roi_status in {valid, invalid, missing}`
- Transition rules (defaults):
  - `LOCKING -> LOCKED` after `5` consecutive `valid` samples (`roi_conf >= 0.70`)
  - `LOCKED -> MISSING` after `8` consecutive `missing` samples (`roi_conf < 0.25`)
  - `LOCKED -> UNLOCKED` after `8` consecutive `invalid` samples (`0.25 <= roi_conf < 0.70`)
  - `MISSING -> UNLOCKED` after `2` seconds (continue relock attempts at 0.5 Hz)

### ROI validator (lightweight, per-sample)
- Input: candidate ROI crop resized to `128x128` (cheap)
- Metrics (combine into `roi_conf`):
  - `ring_edge_energy`: mean Sobel magnitude on a thin border band
  - `inside_texture_std`: stddev of grayscale inside the border band (menus tend to be flat)
  - `corner_anchor`: ROI center proximity to expected corner (prevents drift)
- Default thresholds:
  - `ring_edge_energy >= 18` (valid ring)
  - `inside_texture_std >= 8` (not a flat overlay/menu)
  - `corner_anchor <= 0.12` in normalized distance to expected corner center

### Relock strategy (fallback detector)
- When `UNLOCKED`:
  - Evaluate all presets for the current `(w,h)` across `ui_scale_bucket` × `corner` on 5 warmup frames.
  - Pick the best preset by average `roi_conf`.
- If presets do not reach `roi_conf >= 0.70`:
  - Run a constrained search in the expected corner region:
    - downsample corner region to `256px` max side
    - use edge density + circularity proxy (fast Hough only as last step)
  - Cache the discovered ROI as a new preset for that `(w,h)` bucket (optional; gated behind `debug_dump_dir`).

## 5) Preprocessing strategy (defensive, A/Bable)

### Default behavior
- `preproc.mode = "raw"`:
  - Resize ROI to `YOLO_IMGSZ` (`320` default; use `256` if CPU-bound).
  - Convert BGR->RGB.
  - No CLAHE by default.

### A/B selection to avoid harming YOLO
- During warmup (first `N_WARMUP_SEC=10` seconds where ROI is `LOCKED`):
  - Run YOLO on a small subset of samples with `raw` and `clahe_gated`.
  - Log per-class:
    - mean confidence
    - detections-per-second
    - duplicate rate (same class spamming within a second)
  - Select the mode with higher “quality score”:
    - `score = mean_conf(player) - 0.5 * fp_surrogate_rate`
  - Persist selection for the remainder of the video (per-video cache).

### Per-class confidence logging (required)
- Maintain histograms per class for:
  - raw conf
  - post-filter conf
- Dump to `debug_dump_dir/minimap_conf_<video_id>.json` when enabled.

## 6) YOLO inference strategy (batching + thresholds + CPU/GPU)

### Model loading
- Load once per process:
  - `YOLO_MODEL_PATH = assets/models/best.pt`
  - Use `device="cuda:0"` if available else `"cpu"`.
  - On GPU: `half=True` when supported.

### Inference call pattern (batched)
- `model.predict(list_of_roi_rgb, imgsz=YOLO_IMGSZ, conf=CONF_MIN, iou=IOU_THRES, max_det=MAX_DET, verbose=False)`
- Post-filter detections by class-specific thresholds:
  - `player >= 0.25`
  - `teammate >= 0.25`
  - `enemy >= 0.35`
  - `ping/* >= 0.35` (if applicable)
  - `spike >= 0.30` (if applicable)

### Aggregation per timestamp key (1 Hz output)
- For each `ts_key`, combine internal samples:
  - Player position: median of detected centroids (robust to occasional misses)
  - Counts (teammates/enemies/pings): majority vote or median count
  - Per-entity points: keep top-K by confidence per class per second (`K=10`)

## 7) Rotation + scale strategy (robust heading with safe fallback)

### Rotation mode handling (never assume)
- `rotation_mode in {unknown, north_up, rotating}`.
- Default: `unknown` until proven by evidence.

### Evidence-driven mode selection
- During warmup (first `N_WARMUP_SEC=10` seconds with known `map_id` and stable ROI):
  - Compute alignment confidence under two hypotheses:
    - H1: `north_up` (no per-sample rotation compensation)
    - H2: `rotating` (apply heading-based unrotation; see below)
  - Choose the mode with higher median alignment confidence margin:
    - Require `median(conf_Hbest) - median(conf_Hother) >= 0.10` and `median(conf_Hbest) >= 0.60`
  - If not met: keep `unknown` and do not emit map-space callouts (emit ROI coords only).

### Heading estimation (if mode is rotating)
- Input: crop around `player` detection box expanded by `+40%` (clamped to ROI bounds).
- Primary estimator: discrete template matching over rotated arrow templates
  - Precompute `N_ANGLES=24` templates (every 15°) from a canonical arrow patch (one-time authoring; store in `assets/minimap_templates/`).
  - Score with normalized cross-correlation on edge map (more robust to color shifts).
  - Output: `heading_deg`, `heading_conf` (peak score margin vs runner-up).
- Temporal smoothing (mandatory):
  - Maintain circular EMA with `alpha=0.35`.
  - Delta clamp per second-bin: `abs(delta) <= 120°` (if exceeded, treat as outlier unless `heading_conf >= 0.75`).
- Fallback:
  - If player not detected or `heading_conf < 0.40`, reuse last heading for up to `2` seconds.
  - After `2` seconds without a confident heading: set `rotation_mode=unknown` for that interval and suppress map-space output.

### Scale normalization
- Prefer ROI-derived scale using the validated ROI box (stable once locked).
- Normalize ROI to a canonical square input; always record:
  - `roi_px_bbox` (frame-space)
  - `roi_norm` transform (ROI->canonical)
  - so downstream mapping is consistent even if ROI slightly changes.

## 8) Map ID + alignment strategy (robust voting + masking + caching)

### Map ID selection priority
1. **Explicit config override**: `config["map_id"]` (most reliable; preferred when available).
2. **Multi-frame voting** using low-cost features on overlay-masked ROI background:
   - perceptual hash (pHash) distance against each static map
   - edge histogram similarity (Canny + HOG-lite)
3. **Optional classifier** (recommended if scaling to many maps/themes):
   - small CNN trained on minimap crops (offline), exported and loaded locally
   - only used if available; otherwise skip

### Overlay masking (required for map-id and registration)
- Build a mask to downweight dynamic overlays:
  - union of YOLO detections dilated by `3px` (icons/pings)
  - high-saturation blobs mask (HSV `S > 140` and small connected components)
- Use masked regions as “don’t care” for similarity and alignment.

### Map-id voting window
- While `map_id` is unknown:
  - Evaluate candidates every 5 seconds.
  - Maintain vote counts over the last `30` seconds.
  - Lock map_id if:
    - top1 vote share `>= 0.70` and
    - top1-top2 margin `>= 0.20`.
- Invalidate map_id if:
  - alignment confidence drops below `0.45` for `3` consecutive refresh attempts, or
  - ROI is relocked (new ROI implies potential UI/scale shift).

### Registration (ROI -> static map)
- Target transform: similarity (rotation + scale + translation) first; homography only if needed.
- Primary: ECC alignment on edge map (`cv2.findTransformECC`) with mask:
  - Run at 0.2 Hz (every 5 seconds) when locked and stable.
  - Optimization stops early if `cc >= 0.80` or max iters reached.
- Fallback: ORB/AKAZE + RANSAC if ECC fails
  - Accept if inlier ratio `>= 0.35` and reprojection error `<= 3.0px` (on canonical scale).
- Cache:
  - `transform`, `transform_conf`, `map_id`, `last_refresh_video_sec`
- Invalidation:
  - ROI relock
  - map_id change
  - `transform_conf < 0.50` for `>=2` refreshes

## 9) Callout mapping strategy (masks/polygons + soft assignment + constraints)

### Asset formats (support both)
- Preferred (fastest runtime): per-map callout **label mask**
  - `assets/minimaps_callouts/<Map>_labels.png` where each callout has a unique RGB color or integer id
  - `assets/minimaps_callouts/<Map>_labels.json` mapping color/id -> callout name + hierarchy (coarse/fine)
- Alternate: polygon JSON
  - `assets/minimaps_callouts/<Map>.json` list of polygons per callout name in static-map pixel coordinates

### Coarse-to-fine callouts
- Maintain two levels:
  - `callout_coarse` (sites/spawns/mid/lanes) – low authoring burden, high stability
  - `callout_fine` (specific named rooms) – optional, added over time
- Always output coarse if available; output fine only if alignment confidence is high.

### Soft assignment near boundaries
- Compute a per-callout distance-to-boundary score:
  - For masks: precompute distance transform per label boundary
  - For polygons: use `cv2.pointPolygonTest` distance
- If within `BOUNDARY_SOFT_PX=6` of a boundary:
  - mark `callout_conf <= 0.5`
  - apply stronger temporal smoothing (see below)

### Adjacency + transition constraints (reduce flicker)
- Define `assets/minimaps_callouts/<Map>_adjacency.json`:
  - graph of allowed transitions between callouts (coarse and optionally fine)
- Transition rule (per tracked entity, per output second):
  - if new callout is not adjacent to previous callout:
    - require `2` consecutive seconds of evidence OR `callout_conf >= 0.85`
  - else accept immediately

### Stability scoring + smoothing
- For each entity class, maintain a short window per timestamp key:
  - `WINDOW_SEC=3` (current + previous 2 seconds)
- Output callout = weighted majority vote:
  - weight = `transform_conf * detection_conf * callout_conf`
- Emit `callout_stability` = vote share of chosen callout in the window.

## 10) Output JSON schema (timestamp-keyed, repo-consistent)

### File naming
- `app/session_data/<video_id>_minimap.json` (same `<video_id>` base name convention as other outputs).

### Schema (per timestamp key)
- Top-level: `{ "<ts_key>": <snapshot>, ... }`
- `snapshot`:
  - `roi`: `{ "bbox_xywh": [x,y,w,h], "conf": float, "status": "locked|unlocked|missing", "method": "preset|search" }`
  - `map`: `{ "map_id": str | null, "map_conf": float | null, "transform_conf": float | null }`
  - `rotation`: `{ "mode": "unknown|north_up|rotating", "heading_deg": float | null, "heading_conf": float | null }`
  - `entities`: list of:
    - `{ "cls": str, "conf": float, "roi_xy": [float,float], "roi_bbox": [x1,y1,x2,y2], "map_xy": [float,float] | null, "callout": str | null, "callout_coarse": str | null, "callout_conf": float | null, "callout_stability": float | null, "track_id": int | null }`
  - `debug` (optional, only if enabled): per-class counts, selected preproc mode, sampling rates used in this second

### Required invariants
- Timestamp keys are strings in JSON output (Python ints are acceptable pre-serialization).
- Always emit an entry for each timestamp key present in the 1 Hz baseline timeline:
  - If ROI missing: `entities=[]`, map fields null, and `roi.status="missing"`.

## 11) Acceptance criteria

### Throughput (target on representative 1080p clips)
- GPU: `>= 10×` realtime for minimap-only pass at base rates (2 Hz YOLO, batch=16).
- CPU: `>= 1×` realtime for minimap-only pass at base rates (1–2 Hz YOLO, batch=4, imgsz=256).
- Hard cap: alignment refresh must not exceed `1` ECC attempt per 5 seconds when stable.

### Reliability
- ROI locked during active gameplay: `>= 98%` of seconds.
- ROI relock time after missing/invalid: `<= 2` seconds median.
- Map-id stability once locked: `<= 1` change per 5 minutes unless invalidated by ROI relock.
- Callout flicker (player coarse callout): `<= 5%` of seconds change without adjacency transition.

## 12) Tests and debug tooling

### Tests (additions)
- [ ] Unit test: timestamp mapping (pre-spike and post-spike behavior) using synthetic `timebase`.
- [ ] Unit test: ROI validator scoring (valid vs missing overlays using saved crops).
- [ ] Unit test: callout lookup + adjacency constraint behavior on synthetic points.
- [ ] Golden-video regression:
  - [ ] Add `tests/expected_outputs/<video_id>_minimap.json` for 2–3 short clips.
  - [ ] Comparator that tolerates small coordinate noise (median error tolerance) and focuses on stability metrics.

### Debug tooling (required)
- [ ] Overlay renderer:
  - draw ROI bbox + roi_conf
  - draw detections + class/conf
  - draw map_id + transform_conf + rotation_mode
  - write to `debug_dump_dir/minimap_overlay_<video_id>.mp4` (or image sequence)
- [ ] Dump intermediate crops:
  - ROI crop per second (not per sample) to bound disk usage
  - masked edge image used for alignment
- [ ] Confidence logs:
  - `minimap_conf_<video_id>.json` (per-class conf histograms + rates)

## 13) Implementation checklist (ordered)

- [ ] Add `timebase` computation in `app/cv_files/cv_pipeline.py` (clock_at_video_start_sec, spike_planted_at_video_sec).
- [ ] Add `save_minimap_json(...)` in `app/cv_files/output_utils.py`.
- [ ] Implement `app/cv_files/minimap_pipeline.py` (sampler + batcher + per-second aggregator + JSON builder).
- [ ] Implement `app/cv_files/minimap_roi.py` (presets, validator, lock state machine, relock search).
- [ ] Implement `app/cv_files/minimap_preproc.py` (raw path, CLAHE gating, warmup A/B mode, logging hooks).
- [ ] Implement `app/cv_files/minimap_yolo.py` (model singleton, batched predict, per-class thresholds, ping trigger).
- [ ] Implement `app/cv_files/minimap_alignment.py` (overlay masking, map voting, ECC/ORB registration, cache + invalidation).
- [ ] Implement `app/cv_files/minimap_callouts.py` (mask/polygon loaders, coarse-to-fine mapping, soft assignment, adjacency constraints, smoothing).
- [ ] Add tests + golden outputs + debug overlay CLI entrypoint.

