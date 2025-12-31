# CV Extraction Accuracy Plan (9 Self-Contained Steps)

Scope: Improve VALORANT HUD-based extraction accuracy in the current pipeline (`app/ability_cv.py`, `app/img_preprocess.py`, `model_setup/img_train.py`) while keeping each step independently implementable, testable, and shippable.

Accuracy-driven priority note: Current metrics show `slot_matches.ability_name_accuracy` (46%) and `slot_matches.ability_count_accuracy` (31%) dominate overall error. Timestamp accuracy is already 100%, so it is excluded from priority work. Steps are re-ranked to front-load fixes that directly improve ability name and ability count detection.

## Steps

1. Step 1 - Task-specific ROI preprocessing (template match + charge crops)
2. Step 2 - Add confidences + "unknown" instead of forcing a label
3. Step 3 - Temporal smoothing + state constraints
4. Step 4 - Make charge decoding ability-aware (constrain impossible classes)
5. Step 5 - Fix training split + augmentations (reduce leakage, mimic video)
6. Step 6 - Sampling upgrade (pick sharpest frame per second)
7. Step 7 - Enforce team uniqueness constraint (agent slots)
8. Step 8 - Add alive/dead explicitly (separate from ult-ready)
9. Step 9 - Make clock OCR stable (confidence + monotonic countdown)

## Step 1 - Task-specific ROI preprocessing (template match + charge crops)

### Objective
Increase `ability_name_accuracy` and `ability_count_accuracy` by making both template matching and CNN inputs robust to scale variance, color shifts, cooldown overlays, partial occlusion, and compression artifacts. Better looks like higher match margins and cleaner charge glyphs before classification.

### Where to implement (files + functions)
- `app/ability_cv.py`: `match_icon()` (ability name), charge crop path in `main()` before calling `match_charge_cnn()`.
- `app/img_preprocess.py`: add optional preprocessing entrypoints for CNN inputs (or accept preprocessed crops).
- To locate: search for `cropped = screenshot[y:y+h, x:x+w]` (ability icon and charge crops).

### Inputs / Outputs
- Inputs: raw BGR ROI crops for ability icons and charge count strips.
- Outputs: preprocessed crops and/or feature images used by template matching and CNN inference; no schema change, but improved confidence and accuracy downstream.

### Implementation checklist
- [ ] Diagnose template matching failures (scale variance from HUD scale, brightness/gamma shifts, cooldown overlays, partial occlusion, compression blur) using a small ROI debug dataset.
- [ ] Add a template pyramid (multi-scale matching) in `match_icon()` to handle 720p/1080p and HUD scale variance.
- [ ] Normalize color for ability icon matching: convert to HSV or LAB, equalize luminance (CLAHE), and match on normalized channels.
- [ ] Add an edge-based template path (Canny or Sobel) to reduce impact of color shifts and overlays.
- [ ] Optional upgrade: add an embedding matcher (e.g., MobileNet or small CNN features) and use cosine similarity to re-rank top-N template matches.
- [ ] For charge crops: apply denoise + contrast normalization; resize with a consistent interpolation method (e.g., `INTER_AREA` down, `INTER_CUBIC` up).
- [ ] Apply a light morphological open/close to remove compression speckles on charge glyphs before CNN inference.
- [ ] Add a debug flag to save preprocessed crops for manual inspection and for building a misclassification set.

### Acceptance criteria
- `slot_matches.ability_name_accuracy` improves by ~20-30% relative on the test set (template matching benefit).
- `slot_matches.ability_count_accuracy` improves by ~10-20% relative due to cleaner charge crops.
- Template matching margin (best - second best score) increases on average across sampled frames.

### Test plan
- Unit: verify preprocessing functions preserve shape/dtype and do not crash on empty crops.
- Golden-video: compare ability name accuracy and charge count accuracy before/after preprocessing.
- Metrics: log template match score distributions and CNN `p_max` distributions pre/post.

### Risks / gotchas
- Aggressive normalization can remove discriminative color cues; keep a fallback to raw RGB matching.
- Multi-scale matching increases compute; restrict to 2-3 scales or precompute scaled templates.
- Over-sharpening can distort charge glyphs; tune denoise/sharpen strength conservatively.

### "Prompt to implement this step"
Implement Step 1 (ROI preprocessing). Update `app/ability_cv.py` to add multi-scale + color-normalized template matching in `match_icon()` and to preprocess charge crops before `match_charge_cnn()`. Add optional edge-based matching and a debug dump for preprocessed ROIs. Verify `slot_matches.ability_name_accuracy` and `slot_matches.ability_count_accuracy` improve on the test set.

## Step 2 - Add confidences + "unknown" instead of forcing a label

### Objective
Reduce false positives for ability names and counts by explicitly modeling confidence and suppressing low-confidence outputs. Better looks like fewer wrong labels and a higher ratio of "unknown" when evidence is weak.

### Where to implement (files + functions)
- `app/ability_cv.py`: `match_icon()` (ability name), `main()` loop where slot matches are appended.
- `app/img_preprocess.py`: `match_charge_cnn()` should return class probabilities and top-k.
- To locate: search for `cv2.matchTemplate`, `model.predict`, and where `slot_matches[timestamp_str].append(...)` occurs.

### Inputs / Outputs
- Inputs: template matching scores and CNN probabilities.
- Outputs: `(label, confidence, margin)` for ability name; `(label, p_max, topk)` for ability count. Persist confidence in output JSON.

### Implementation checklist
- [ ] Diagnose template matching false positives caused by similar icons (e.g., overlapping silhouettes) and cooldown overlays.
- [ ] Return top-2 template scores from `match_icon()` and compute a margin; gate output to `unknown` if score/margin below threshold.
- [ ] Add per-agent template sets or per-ability template variants (normal vs cooldown) and use confidence to choose.
- [ ] In `match_charge_cnn()`, return `p_max` and top-k labels; add temperature scaling or simple calibration if `p_max` is overconfident.
- [ ] Add a fallback rule: if `p_max` is below threshold, keep last stable count for that ability (or mark unknown).
- [ ] Store confidence for both ability name and count in JSON (or parallel fields) for downstream smoothing.

### Acceptance criteria
- `slot_matches.ability_name_accuracy` improves by ~10-20% relative due to reduced false positives.
- `slot_matches.ability_count_accuracy` improves by ~10-15% relative from confidence gating and fallback.
- The percentage of `unknown` outputs is bounded (target <20% on stable HUD sections).

### Test plan
- Unit: ensure low-confidence template matches produce `unknown`.
- Golden-video: measure accuracy vs unknown rate tradeoff and choose thresholds.
- Metrics: track confusion matrix for ability names and counts with and without confidence gating.

### Risks / gotchas
- Overly strict thresholds can reduce recall too far; tune with ROC-style curves.
- CNN probabilities may be poorly calibrated; verify `p_max` against true correctness.

### "Prompt to implement this step"
Implement Step 2 (confidence gating). Update `match_icon()` in `app/ability_cv.py` to return top-2 scores and gate low-margin matches to `unknown`. Update `match_charge_cnn()` in `app/img_preprocess.py` to return `p_max` and top-k; add a low-confidence fallback to last stable count. Persist confidence fields and re-measure `ability_name_accuracy` and `ability_count_accuracy`.

## Step 3 - Temporal smoothing + state constraints

### Objective
Improve stability for ability name and ability count outputs by enforcing temporal consistency and plausible transitions. Better looks like fewer 1-2 frame glitches and smoother count trajectories.

### Where to implement (files + functions)
- `app/ability_cv.py`: main loop in `main()` after raw detections and before writing JSON.
- `app/img_preprocess.py`: ensure `match_charge_cnn()` returns `p_max`/top-k so smoothing can be confidence-weighted.
- To locate: search for `slot_matches[timestamp_str]` and any per-frame state assembly.

### Inputs / Outputs
- Inputs: per-timestamp ability name predictions + confidences; per-timestamp ability count predictions + probabilities.
- Outputs: smoothed ability name and count sequences with optional anomaly flags.

### Implementation checklist
- [ ] Diagnose flicker sources: HUD animation, compression flicker, cooldown overlays causing transient template mismatches.
- [ ] Apply majority vote / hysteresis for ability name labels across a 3-5 frame window.
- [ ] Apply count smoothing with constraints (counts should not increase unexpectedly unless a refill occurs).
- [ ] Add debouncing: require N consecutive frames before accepting a new ability name or count.
- [ ] Add a "plausible transition" table for counts (e.g., 2->0 is plausible only if enough time passed).
- [ ] Propagate confidence through smoothing (min/mean confidence of the accepted window) using `p_max` from `img_preprocess.py`.

### Acceptance criteria
- `slot_matches.ability_name_accuracy` improves by ~10-15% relative from reduced flicker.
- `slot_matches.ability_count_accuracy` improves by ~15-25% relative from smoothed counts.
- Flip rate (label changes per minute) drops by at least 30% on sample clips.

### Test plan
- Unit: verify debouncing logic on synthetic time series.
- Golden-video: compare pre/post flicker rate and count jump rate.
- Metrics: report accuracy and stability over multiple clips with different bitrate settings.

### Risks / gotchas
- Over-smoothing can hide real rapid events; keep window sizes configurable.
- Countdown timestamps decrease; ensure smoothing operates on ordered time series, not raw JSON key order.

### "Prompt to implement this step"
Implement Step 3 (temporal smoothing). Add majority/hysteresis smoothing for ability name and count signals in `app/ability_cv.py`, with debouncing and plausible transition constraints. Persist smoothed outputs and compare `ability_name_accuracy`, `ability_count_accuracy`, and flip rate before/after.

## Step 4 - Make charge decoding ability-aware (constrain impossible classes)

### Objective
Improve `ability_count_accuracy` by rejecting impossible CNN outputs using ability-specific max charges and fallback logic. Better looks like no invalid counts and fewer misreads from similar classes.

### Where to implement (files + functions)
- `app/img_preprocess.py`: `match_charge_cnn()` should expose softmax probabilities (not only argmax).
- `app/ability_cv.py`: apply constraints after mapping `Ability #n` to actual ability name.
- To locate: search for `agent_to_ability.json` usage and charge string assembly.

### Inputs / Outputs
- Inputs: CNN probabilities, ability name, and ability max charges mapping.
- Outputs: constrained charge label + confidence; optionally both raw and constrained labels for debugging.

### Implementation checklist
- [ ] Diagnose CNN errors: confusion between visually similar counts (e.g., `1-2` vs `2-2`), low-res aliasing, missing negatives.
- [ ] Add `ability_max_charges.json` keyed by canonical ability name.
- [ ] Implement `constrain_charge_prediction(ability_name, topk_preds)` to filter out invalid classes.
- [ ] Add fallback: if all valid classes are low confidence, keep last stable count or mark unknown.
- [ ] Log constraint overrides and analyze if they reduce known confusions.
- [ ] Tie back to `img_preprocess.py` by returning calibrated `p_max` for filtering.

### Acceptance criteria
- `slot_matches.ability_count_accuracy` improves by ~10-20% relative on the test set.
- Zero outputs violate `max_charges` constraints.
- Override rate is reasonable (<25% of frames) and correlates with previously misclassified cases.

### Test plan
- Unit: constraint function rejects invalid labels and selects best valid alternative.
- Golden-video: compare accuracy and invalid-label rate before/after constraints.

### Risks / gotchas
- Ability max charges change with patches; keep mapping versioned and easy to update.
- Naming mismatches across JSON and templates can break constraints; canonicalize IDs first.

### "Prompt to implement this step"
Implement Step 4 (ability-aware constraints). Update `app/img_preprocess.py` to return top-k with probabilities. Add `ability_max_charges.json` and apply constraints in `app/ability_cv.py` after ability name mapping. Add fallback to last stable count when confidence is low and verify `ability_count_accuracy` improves.

## Step 5 - Fix training split + augmentations (reduce leakage, mimic video)

### Objective
Improve `ability_count_accuracy` by retraining the CNN with proper splits and realistic augmentations to handle font variation, UI animation, low resolution, and compression. Better looks like stronger validation accuracy and fewer confusions in production clips.

### Where to implement (files + functions)
- `model_setup/img_train.py`: dataset creation, augmentations, training loop.
- `app/img_preprocess.py`: ensure model and class names remain compatible after retraining.
- To locate: search for `image_dataset_from_directory` and augmentation layers.

### Inputs / Outputs
- Inputs: charge crop dataset grouped by video/source.
- Outputs: updated `assets/models/best_slot_classifier.h5` and `assets/models/class_names.json`.

### Implementation checklist
- [ ] Diagnose CNN failure modes: class imbalance, font/aliasing variation, animated HUD backgrounds, and lack of "hard negative" crops.
- [ ] Split train/val by video, not by frame, to avoid leakage.
- [ ] Rebalance classes or use class weights / focal loss to handle rare counts.
- [ ] Add realistic augmentations: JPEG compression, motion blur, downscale-then-upscale, gamma shifts, slight ROI jitter.
- [ ] Consider a lighter architecture with explicit digit features (e.g., small CNN + global average pool) and compare to MobileNetV2.
- [ ] Calibrate confidence (temperature scaling) and export calibration params used by `img_preprocess.py`.
- [ ] Include "negative" examples: crops with overlays, partial occlusion, or corrupted frames.

### Acceptance criteria
- `slot_matches.ability_count_accuracy` improves by ~20-30% relative on the test set.
- Confusion matrix shows reduced swaps among the top-3 confused classes.
- The exported model remains compatible with `app/img_preprocess.py` input size and labels.

### Test plan
- Training: run `model_setup/img_train.py` and evaluate with `model_setup/img_test.py`.
- Inference: run end-to-end extraction and compare count accuracy vs baseline.

### Risks / gotchas
- Aggressive augmentation can hurt if it deviates from actual capture artifacts; tune gradually.
- Changing class names or ordering can silently break inference; keep class_names.json in sync.

### "Prompt to implement this step"
Implement Step 5 (CNN retraining). Update `model_setup/img_train.py` to split by video, add realistic augmentations, handle class imbalance, and optionally test a smaller CNN. Export updated models and update `app/img_preprocess.py` to load the new artifacts. Verify `ability_count_accuracy` improves on the test set.

## Step 6 - Sampling upgrade (pick sharpest frame per second)

### Objective
Improve both template matching and CNN accuracy by sampling the sharpest frame around each second, reducing motion blur and transitional HUD frames. Better looks like clearer ability icons and charge glyphs.

### Where to implement (files + functions)
- `app/ability_cv.py`: `extract_frames()` or a new "burst sampling" function; main loop in `main()`.
- To locate: search for `extract_frames(video_path, interval=1)` and `frame_count % frame_interval == 0`.

### Inputs / Outputs
- Inputs: video stream frames.
- Outputs: a list of sampled frames per second selected by a sharpness metric (variance of Laplacian).

### Implementation checklist
- [ ] Diagnose blur-related failures for both template matching and CNN (e.g., low `p_max`, low match margins).
- [ ] Sample a small window of frames around each 1-second tick and compute sharpness for each.
- [ ] Select the sharpest frame; optionally keep a runner-up for debugging.
- [ ] Log sharpness and detection confidence (template match margin and `p_max` from `img_preprocess.py`) to correlate improvements.
- [ ] Keep sampling deterministic and bounded (small window size).

### Acceptance criteria
- `slot_matches.ability_name_accuracy` improves by ~5-10% relative on motion-heavy clips.
- `slot_matches.ability_count_accuracy` improves by ~10-15% relative on motion-heavy clips.
- Average template match margin and CNN `p_max` increase vs baseline sampling.

### Test plan
- Unit: validate sharpness ranking on synthetic blurred vs sharp frames.
- Golden-video: compare accuracy and confidence metrics pre/post sampling upgrade.

### Risks / gotchas
- Increased decode cost; keep window size small (3-5 frames).
- The sharpest frame might still be mid-animation; pair with smoothing (Step 3).

### "Prompt to implement this step"
Implement Step 6 (sharpest-frame sampling). Update `app/ability_cv.py` to sample a small frame burst per second and select the sharpest frame via variance of Laplacian. Verify `ability_name_accuracy` and `ability_count_accuracy` improve on motion-heavy clips and that confidence metrics increase.

## Step 7 - Enforce team uniqueness constraint (agent slots)

### Objective
Reduce agent misclassification in the 5-slot HUD by enforcing the constraint that each team has unique agents. Better looks like fewer duplicates and fewer `unknown` oscillations when multiple templates are similar.

### Where to implement (files + functions)
- `app/ability_cv.py`: `detect_agent()` (return top-k candidates), main loop where `team_status_dict` / `enemy_status_dict` are populated.
- To locate: search for `team_status_dict[timestamp_str][agent] =` and `enemy_status_dict`.

### Inputs / Outputs
- Inputs: For each team and timestamp, per-slot candidate scores (top-k agent labels with template-match scores).
- Outputs: A resolved per-slot assignment with unique agent names, plus an assignment confidence (e.g., min of chosen scores or assignment margin).

### Implementation checklist
- [ ] Modify `detect_agent()` to return top-k candidates (e.g., top 3 labels + scores) instead of only best match.
- [ ] Create a `resolve_unique_agents(slot_candidates)` function:
  - input: list length=5, each element is list of `(agent, score)` candidates.
  - output: list length=5 unique agents (or `unknown`) maximizing total score.
- [ ] Start with a simple greedy resolver (highest score first) and upgrade to Hungarian assignment if needed.
- [ ] Preserve `unknown` when all candidates are below threshold (do not force a unique-but-wrong assignment).
- [ ] Ensure resolver operates separately for `team` and `enemy` and per timestamp.
- [ ] Add debug output for duplicate resolution decisions (only when `visualize` or `debug` flag set).
- [ ] Add a metric: `% timestamps with duplicates before/after`.

### Acceptance criteria
- Duplicate agents per team per timestamp drop to near-zero on sample clips (target: <1% of timestamps).
- When duplicates occur, resolver outputs `unknown` rather than inventing a low-confidence label.
- No `"null"` keys appear in `v*_game_status.json`.

### Test plan
- Unit: feed synthetic candidate lists with duplicates and verify resolver picks the global optimum (or expected greedy choice).
- Golden-video: compare duplicate-rate metric before/after on a short clip.

### Risks / gotchas
- If the underlying detector is very noisy, uniqueness constraints can "spread" errors to other slots; keep `unknown` as a valid outcome.
- Agents with similar portraits may still confuse template matching; Step 1 preprocessing helps.

### "Prompt to implement this step"
Implement Step 7 (unique agent assignment). Update `app/ability_cv.py` so `detect_agent()` returns top-k candidates. Add a resolver that enforces unique agents across the 5 slots for each team at each timestamp, preferring high scores and allowing `unknown`. Update JSON outputs and add a small metric script that reports duplicate rate before/after.

## Step 8 - Add alive/dead explicitly (separate from ult-ready)

### Objective
Extract whether each agent slot is alive/dead as its own signal instead of conflating status with ult readiness. Better looks like a reliable `alive` boolean per agent per timestamp, enabling coaching rules that depend on numbers advantage.

### Where to implement (files + functions)
- `app/ability_cv.py`: agent-slot processing loop inside `main()`, plus a new detector function (e.g., `detect_alive_dead()`).
- To locate: search for `team_status` / `enemy_status` loops and where `team_status_dict` is written.

### Inputs / Outputs
- Inputs: per-slot agent portrait crop (same crop used for `detect_agent()`).
- Outputs: Extend game status output to include `alive` separate from `ult_ready`.
  - Example: `team: { agent: { "alive": bool, "ult_ready": bool, ... } }`.
  - If agent is `unknown`, still record slot-level `alive` (keyed by slot index) or skip safely.

### Implementation checklist
- [ ] Define the output structure update for `v*_game_status.json` (version or new keys).
- [ ] Implement an alive/dead heuristic detector:
  - [ ] Option A (fast): saturation/brightness stats + edge density + a "dead overlay" template on a sub-ROI.
  - [ ] Option B (more robust): train a tiny binary classifier on slot crops (alive vs dead).
- [ ] Keep the alive detector independent of agent ID (works even when agent is `unknown`).
- [ ] Add confidence for alive/dead (ratio-based confidence or classifier probability).
- [ ] Ensure no `"null"` keys; if agent unknown, store by `slot_index` or omit agent key but keep alive signal elsewhere.
- [ ] Update JSON assembly accordingly and update any downstream readers/scripts.

### Acceptance criteria
- Adds `alive` (and optionally `alive_conf`) per slot per timestamp in `v*_game_status.json`.
- On a labeled mini-set (manually labeled ~200 frames), achieves >=90% alive/dead accuracy.

### Test plan
- Unit: synthetic crops (blank, dark, bright) do not crash; detector returns a boolean + confidence.
- Golden-video: manually label 2 short segments and compute alive/dead accuracy; log false positives/negatives.

### Risks / gotchas
- HUD styles vary (spectator/recording overlays); heuristics may not generalize without training data.
- Agent portraits can be partially occluded during UI animation; prefer temporal smoothing (Step 3).

### "Prompt to implement this step"
Implement Step 8 (alive/dead extraction). Add an `alive` signal per agent slot in `app/ability_cv.py` using a heuristic detector (or a small binary classifier) and update `app/session_data/*_game_status.json` schema to include `alive` separately from `ult_ready`, with confidence. Verify on a small labeled frame set that alive/dead accuracy is >=90%.

## Step 9 - Make clock OCR stable (confidence + monotonic countdown)

### Objective
Reduce OCR errors in the round clock by using OCR confidence and enforcing the clock's monotonic countdown. Better looks like a clean seconds-remaining timeline with rare, corrected OCR glitches.

### Where to implement (files + functions)
- `app/ability_cv.py`: `time_capture()`, the initial time discovery loop in `main()` (search for `while not (len(m) == 1 and len(s) == 2)`), and timestamp progression logic.
- To locate: search for `pytesseract.image_to_string` and `clock_bbox`.

### Inputs / Outputs
- Inputs: clock ROI crop per sampled frame.
- Outputs: a stable `round_time_s` per frame (and/or a stable mapping from frame index to round seconds), plus an OCR confidence score for each read.

### Implementation checklist
- [ ] Switch from `image_to_string` to `image_to_data` to capture confidence per detection.
- [ ] Parse OCR output into `mm:ss`; reject low-confidence or malformed reads.
- [ ] Implement a monotonic constraint solver:
  - [ ] expected `t_i = t_0 - i*interval`
  - [ ] accept reads close to expected; otherwise ignore and interpolate
- [ ] Add a fallback when OCR fails early (use frame index as elapsed time until first good read).
- [ ] Log OCR acceptance/rejection rates and the final derived timebase.
- [ ] Store a `time_conf` per timestamp (optional) for downstream uncertainty.

### Acceptance criteria
- OCR produces a valid `mm:ss` (or seconds) for >=95% of sampled timestamps on a sample clip (with interpolation allowed).
- Time series is strictly monotonic in the correct direction (countdown).
- Initial timestamp alignment is consistent run-to-run (deterministic given same video).

### Test plan
- Unit: feed known synthetic `mm:ss` strings and verify parsing and monotonic filter behavior.
- Golden-video: log raw OCR reads + accepted reads; compute `% accepted`, `% interpolated`, and max deviation from expected.

### Risks / gotchas
- Some recordings hide the clock in certain phases; handle missing clock gracefully.
- Countdown vs elapsed can vary by mode/overlay; detect direction from first few reads.

### "Prompt to implement this step"
Implement Step 9 (stable clock OCR). Update `app/ability_cv.py` `time_capture()` to use `pytesseract.image_to_data` and return both parsed time and OCR confidence. Add monotonic countdown enforcement in `main()` so bad reads are rejected and missing reads are interpolated. Verify >=95% valid timestamps and a strictly monotonic time series on a sample clip.
