# CV Extraction Accuracy Plan (9 Self-Contained Steps)

Scope: Improve VALORANT HUD-based extraction accuracy in the current pipeline (`app/ability_cv.py`, `app/img_preprocess.py`, `model_setup/img_train.py`) while keeping each step independently implementable, testable, and shippable.

## Steps

1. Step 1 — Add confidences + “unknown” instead of forcing a label
2. Step 2 — Enforce team uniqueness constraint (agent slots)
3. Step 3 — Task-specific ROI preprocessing (template match + ult color)
4. Step 4 — Add alive/dead explicitly (separate from ult-ready)
5. Step 5 — Temporal smoothing + state constraints
6. Step 6 — Make charge decoding ability-aware (constrain impossible classes)
7. Step 7 — Fix training split + augmentations (reduce leakage, mimic video)
8. Step 8 — Make clock OCR stable (confidence + monotonic countdown)
9. Step 9 — Sampling upgrade (pick sharpest frame per second)

## Step 1 — Add confidences + “unknown” instead of forcing a label

### Objective
Expose per-detection confidence (and optionally a runner-up margin) so downstream logic can reject low-quality reads and reduce flicker. “Better” looks like fewer incorrect label flips and explicit `unknown` outputs rather than wrong labels.

### Where to implement (files + functions)
- `app/ability_cv.py`: `match_icon()`, `detect_agent()`, `detect_ultimate()`, main loop inside `main()` where outputs are assembled.
- `app/img_preprocess.py`: `match_charge_cnn()` (return top-k probabilities or at least `p_max`).
- To locate: search for `cv2.matchTemplate`, `model.predict`, and the places where JSON dicts are built (`combined_status`, `converted`).

### Inputs / Outputs
- Inputs: ROI crops (agent slot, ult bar, ability icon, charge crop) from each sampled frame.
- Outputs:
  - For each detection type, return `(label, confidence[, margin])`.
  - JSON outputs should include confidences, either by extending current values or by adding parallel `*_confidence` fields.
    - Example (one option): `team: { agent_name: { "ult_ready": bool, "ult_conf": float, "agent_conf": float } }`.

### Implementation checklist
- [ ] Define a small, shared “detection result” shape (tuple or dict) for all detectors: `label`, `confidence`, `margin`.
- [ ] Update `match_icon()` to return best score and (optionally) second-best score margin.
- [ ] Update `detect_agent()` to return agent label + score(s) and gate low-confidence results to `unknown`.
- [ ] Update `detect_ultimate()` to return a probability-like confidence (e.g., normalized yellow ratio) alongside the boolean.
- [ ] Update `match_charge_cnn()` to optionally return `{"label": str, "p_max": float, "topk": [...]}` (requires reading softmax output).
- [ ] Add thresholds in one place (constants/config): `MIN_TM_SCORE`, `MIN_TM_MARGIN`, `MIN_CNN_PMAX`, `MIN_ULT_RATIO`.
- [ ] Update JSON assembly to persist confidence values for every timestamp.
- [ ] Add a per-timestamp debug log option (guarded by a flag) that prints low-confidence detections.
- [ ] Ensure `unknown`/`None` keys never become JSON keys (avoid `"null": false` artifacts).

### Acceptance criteria
- Low-confidence detections are emitted as `unknown` (not a wrong label) for at least agent and ability icon matching.
- Adds confidence output for: agent label, ult-ready, and charge class (`p_max`).
- On 2–3 sample clips, reduces “impossible flips” (label changes that revert within 1–2 seconds) by ≥30% vs baseline.

### Test plan
- Unit: add a small test that `match_icon()` returns `(label, score)` and that `unknown` triggers when score below threshold.
- Golden-video: run extraction on a short clip twice; confirm deterministic outputs and record flip counts.
- Metrics/logging: write a script to compute per-signal flip rate and `% unknown` over time.

### Risks / gotchas
- Thresholds can be resolution/encoding dependent; keep them configurable per run.
- Adding confidences may require a JSON schema bump; version outputs or add new keys instead of breaking old readers.
- Template matching scores can be misleading under brightness shifts; margins help more than raw score.

### “Prompt to implement this step”
Implement Step 1 (confidences + unknown gating). Update `app/ability_cv.py` (`match_icon`, `detect_agent`, `detect_ultimate`, JSON assembly) and `app/img_preprocess.py` (`match_charge_cnn`) to return confidences and output `unknown` when below thresholds. Update the saved JSON in `app/session_data/` to include confidence fields. Add a small script/test to compute flip rate and confirm it drops on a sample clip.

## Step 2 — Enforce team uniqueness constraint (agent slots)

### Objective
Reduce agent misclassification in the 5-slot HUD by enforcing the constraint that each team has unique agents. “Better” looks like fewer duplicates and fewer `null/unknown` oscillations when multiple templates are similar.

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
- [ ] Preserve `unknown` when all candidates are below threshold (don’t force a unique-but-wrong assignment).
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
- If the underlying detector is very noisy, uniqueness constraints can “spread” errors to other slots; keep `unknown` as a valid outcome.
- Agents with similar portraits may still confuse template matching; Step 3 preprocessing helps.

### “Prompt to implement this step”
Implement Step 2 (unique agent assignment). Update `app/ability_cv.py` so `detect_agent()` returns top-k candidates. Add a resolver that enforces unique agents across the 5 slots for each team at each timestamp, preferring high scores and allowing `unknown`. Update JSON outputs and add a small metric script that reports duplicate rate before/after.

## Step 3 — Task-specific ROI preprocessing (template match + ult color)

### Objective
Improve robustness to compression, gamma shifts, and HUD animation by preprocessing ROIs differently per task (icons vs colors). “Better” looks like higher template-match margins and fewer ult-ready false positives/negatives.

### Where to implement (files + functions)
- `app/ability_cv.py`: inside `match_icon()`, `detect_agent()`, `detect_ultimate()`.
- To locate: search for ROI crop lines like `cropped = screenshot[y:y+h, x:x+w]`.

### Inputs / Outputs
- Inputs: raw BGR ROI crops.
- Outputs: preprocessed crops (or feature images) used for matching; no required output schema change, but confidence improvements should be measurable.

### Implementation checklist
- [ ] Add preprocessing helpers:
  - [ ] `prep_for_template(crop_bgr) -> img`: grayscale → CLAHE → (optional) blur → Canny edges or normalized gray.
  - [ ] `prep_for_color_ratio(crop_bgr) -> mask`: HSV mask + morphology + inner ROI mask.
- [ ] Convert template matching to use preprocessed representations consistently for both `crop` and `template`.
- [ ] For ult detection, restrict measurement to a central sub-rectangle (ignore borders).
- [ ] Calibrate HSV thresholds by sampling a few frames and logging yellow ratio distribution.
- [ ] Add an optional “debug save” mode to write preprocessed ROIs to disk for inspection.
- [ ] Add/adjust thresholds after preprocessing (scores will shift).

### Acceptance criteria
- Template matching score margin (best-second) increases on average vs baseline on a sample set of frames.
- Ult-ready flicker (true↔false toggles within 2 seconds) reduced by ≥25% on sample clips.

### Test plan
- Unit: ensure preprocessing functions preserve expected shapes/dtypes and don’t crash on empty/invalid crops.
- Golden-video: run extraction with debug logging of score margins and ult ratios; compare distributions pre/post change.

### Risks / gotchas
- Edge-based matching can fail if HUD uses low-contrast art; keep a fallback to normalized grayscale if needed.
- CLAHE parameters can over-amplify compression blocks; tune clip limit conservatively.

### “Prompt to implement this step”
Implement Step 3 (ROI preprocessing). In `app/ability_cv.py`, add task-specific preprocessing helpers and use them in `match_icon`, `detect_agent`, and `detect_ultimate` (edge/normalized matching for icons; masked HSV ratio for ult). Add debug outputs to inspect preprocessed crops and verify improved match margins and reduced ult flicker on a sample clip.

## Step 4 — Add alive/dead explicitly (separate from ult-ready)

### Objective
Extract whether each agent slot is alive/dead as its own signal instead of conflating status with ult readiness. “Better” looks like a reliable `alive` boolean per agent per timestamp, enabling coaching rules that depend on numbers advantage.

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
  - [ ] Option A (fast): saturation/brightness stats + edge density + a “dead overlay” template on a sub-ROI.
  - [ ] Option B (more robust): train a tiny binary classifier on slot crops (alive vs dead).
- [ ] Keep the alive detector independent of agent ID (works even when agent is `unknown`).
- [ ] Add confidence for alive/dead (ratio-based confidence or classifier probability).
- [ ] Ensure no `"null"` keys; if agent unknown, store by `slot_index` or omit agent key but keep alive signal elsewhere.
- [ ] Update JSON assembly accordingly and update any downstream readers/scripts.

### Acceptance criteria
- Adds `alive` (and optionally `alive_conf`) per slot per timestamp in `v*_game_status.json`.
- On a labeled mini-set (manually labeled ~200 frames), achieves ≥90% alive/dead accuracy.

### Test plan
- Unit: synthetic crops (blank, dark, bright) don’t crash; detector returns a boolean + confidence.
- Golden-video: manually label 2 short segments and compute alive/dead accuracy; log false positives/negatives.

### Risks / gotchas
- HUD styles vary (spectator/recording overlays); heuristics may not generalize without training data.
- Agent portraits can be partially occluded during UI animation; prefer temporal smoothing (Step 5).

### “Prompt to implement this step”
Implement Step 4 (alive/dead extraction). Add an `alive` signal per agent slot in `app/ability_cv.py` using a heuristic detector (or a small binary classifier) and update `app/session_data/*_game_status.json` schema to include `alive` separately from `ult_ready`, with confidence. Verify on a small labeled frame set that alive/dead accuracy is ≥90%.

## Step 5 — Temporal smoothing + state constraints

### Objective
Convert noisy per-frame detections into stable time series using smoothing and debouncing, while enforcing basic constraints (no rapid flips, no impossible jumps). “Better” looks like fewer 1-frame glitches and smoother coaching-relevant events.

### Where to implement (files + functions)
- `app/ability_cv.py`: main per-frame loop in `main()`, right after raw detections and before writing to dicts.
- To locate: search for where `slot_matches[timestamp_str]` and `team_status_dict[timestamp_str]` are populated.

### Inputs / Outputs
- Inputs: per-timestamp raw detections + confidences (from Steps 1–4).
- Outputs: smoothed detections written to JSON (plus optional anomaly flags).

### Implementation checklist
- [ ] Implement a small temporal filter utility:
  - [ ] majority vote over a window (e.g., 3–5 samples) for categorical labels.
  - [ ] median / hysteresis for boolean ult-ready / alive.
  - [ ] debouncing: require `N` consecutive samples before changing a state.
- [ ] Apply smoothing separately per signal:
  - [ ] agent label per slot
  - [ ] ult_ready per slot
  - [ ] alive per slot
  - [ ] ability charge class per ability
- [ ] Add “impossible change” detection (e.g., charge jumps by >1 level in 1 second) and either clamp or mark as anomaly.
- [ ] Propagate confidence through smoothing (e.g., average confidence for chosen label, or min over window).
- [ ] Add a debug mode to print all state changes with timestamps and confidence.

### Acceptance criteria
- Reduces label flicker: count of state changes per minute drops by ≥30% on sample clips.
- Charge time series contains ≤1% “impossible jumps” after smoothing (define clearly).
- Adds optional anomaly flags without crashing downstream JSON consumers.

### Test plan
- Unit: feed synthetic time series and verify debouncing behavior (no change until N confirmations).
- Golden-video: compute flip rates and impossible-jump rates pre/post.

### Risks / gotchas
- Over-smoothing can hide real quick events; keep window sizes configurable.
- If your timebase is countdown and timestamps decrease, smoothing logic must handle reversed time order.

### “Prompt to implement this step”
Implement Step 5 (temporal smoothing). In `app/ability_cv.py`, add debouncing/majority filters for agent labels, alive, ult-ready, and ability charges using a configurable window. Add anomaly detection for impossible jumps and log/flag them. Verify flip rate and impossible-jump rate decrease on a sample clip.

## Step 6 — Make charge decoding ability-aware (constrain impossible classes)

### Objective
Use known ability max charges to reject impossible CNN outputs (e.g., predicting `*-3` for a 1-charge ability). “Better” looks like higher charge classification accuracy without retraining, and fewer nonsensical charge states.

### Where to implement (files + functions)
- `app/img_preprocess.py`: `match_charge_cnn()` should expose softmax probabilities (not only argmax).
- `app/ability_cv.py`: after mapping slot → ability name (see `save_to_json()` and slot conversion), apply constraints before finalizing charge outputs.
- To locate: search for `agent_to_ability.json` usage and where charge strings like `2-2` are written.

### Inputs / Outputs
- Inputs: charge classifier probabilities; slot → ability mapping; a config mapping `ability_name -> max_charges`.
- Outputs: constrained charge labels + confidence; optionally store both raw and constrained predictions for debugging.

### Implementation checklist
- [ ] Create `ability_max_charges.json` (or python dict) keyed by canonical ability name.
- [ ] Update `match_charge_cnn()` to return `preds` or top-k labels with probabilities.
- [ ] Implement `constrain_charge_prediction(ability_name, topk_preds)`:
  - filters out labels incompatible with `max_charges`.
  - chooses best remaining label; falls back to previous stable label if confidence low.
- [ ] Integrate constraint step right before writing `slot_matches` / converted JSON.
- [ ] Add logging for when constraints override the raw argmax.
- [ ] Add a metric: override rate and accuracy change on a labeled validation set (if available).

### Acceptance criteria
- No constrained output violates `max_charges` constraints.
- On a small labeled set, charge accuracy improves vs baseline (target: +3–5% absolute) or flicker decreases if labels are sparse.

### Test plan
- Unit: constraint function rejects impossible labels and picks best valid alternative.
- Golden-video: track how often constraints override model predictions; inspect a few overrides visually.

### Risks / gotchas
- Ability max charges can change with patches; keep the mapping easy to update.
- Naming mismatches between `agent_to_ability.json`, `ability_description.json`, and model class labels must be normalized (canonicalization).

### “Prompt to implement this step”
Implement Step 6 (ability-aware charge constraints). Update `app/img_preprocess.py` so `match_charge_cnn` returns probabilities/top-k. Add an `ability_max_charges` mapping and apply it in `app/ability_cv.py` when converting slot outputs to ability names, rejecting impossible charge classes and emitting constrained labels + confidence. Verify no outputs violate max charges and measure override rate on a sample clip.

## Step 7 — Fix training split + augmentations (reduce leakage, mimic video)

### Objective
Improve generalization of the charge classifier by splitting train/val by video (not by frame) and using augmentations that match real capture artifacts. “Better” looks like stable validation accuracy that transfers to unseen videos and fewer misreads under compression/blur.

### Where to implement (files + functions)
- `model_setup/img_train.py`: dataset creation, augmentations, training loop.
- `model_setup/img_script.py` / `model_setup/img_crop.py`: data collection pipeline (if you use them to generate crops).
- To locate: search for `image_dataset_from_directory` and augmentation layers.

### Inputs / Outputs
- Inputs: cropped charge images organized by source (ideally per video ID).
- Outputs: updated `assets/models/best_slot_classifier.h5` and `assets/models/class_names.json` (same consumer interface).

### Implementation checklist
- [ ] Restructure `data_crop/` to preserve video identity (e.g., `data_crop/by_video/<video_id>/<label>/*.png`) or store an index file with `video_id`.
- [ ] Build train/val splits by `video_id` (no shared videos across splits).
- [ ] Remove/avoid augmentations that break realism (e.g., `RandomFlip("horizontal")` for charge crops).
- [ ] Add realistic augmentations:
  - [ ] JPEG compression simulation (encode/decode in pipeline or offline aug)
  - [ ] motion blur / gaussian blur
  - [ ] downscale then upscale
  - [ ] gamma/brightness shifts within capture-like bounds
  - [ ] small ROI jitter (translation)
- [ ] Add metrics: per-class accuracy, confusion matrix (reuse `model_setup/img_test.py` approach).
- [ ] Export the trained model to `assets/models/` and update any paths used by `app/img_preprocess.py`.

### Acceptance criteria
- Validation set shares zero videos with training set (verify by listing split membership).
- Confusion matrix improves on the most common confusions (define top-3 confusions and show reduction).
- Model inference in `app/img_preprocess.py` remains compatible (same input size and class name mapping).

### Test plan
- Training: run `model_setup/img_train.py` and save artifacts; run `model_setup/img_test.py` to print accuracy/confusion matrix.
- Inference: run extraction on a short video and confirm charge predictions look plausible (no widespread collapse).

### Risks / gotchas
- Data refactor can be time-consuming; do it incrementally and keep a backward-compatible loader if needed.
- Augmentations that are too strong can hurt; add one at a time and measure.

### “Prompt to implement this step”
Implement Step 7 (training split + augmentations). Update `model_setup/img_train.py` to split train/val by video ID (no leakage) and replace unrealistic augmentations with capture-like ones (compression, blur, downscale/upscale, gamma, slight jitter). Retrain and export the best model to `assets/models/`, then verify with `model_setup/img_test.py` and a quick end-to-end extraction run.

## Step 8 — Make clock OCR stable (confidence + monotonic countdown)

### Objective
Reduce OCR errors in the round clock by using OCR confidence and enforcing the clock’s monotonic countdown. “Better” looks like a clean seconds-remaining timeline with rare, corrected OCR glitches.

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
- OCR produces a valid `mm:ss` (or seconds) for ≥95% of sampled timestamps on a sample clip (with interpolation allowed).
- Time series is strictly monotonic in the correct direction (countdown).
- Initial timestamp alignment is consistent run-to-run (deterministic given same video).

### Test plan
- Unit: feed known synthetic `mm:ss` strings and verify parsing and monotonic filter behavior.
- Golden-video: log raw OCR reads + accepted reads; compute `% accepted`, `% interpolated`, and max deviation from expected.

### Risks / gotchas
- Some recordings hide the clock in certain phases; handle missing clock gracefully.
- Countdown vs elapsed can vary by mode/overlay; detect direction from first few reads.

### “Prompt to implement this step”
Implement Step 8 (stable clock OCR). Update `app/ability_cv.py` `time_capture()` to use `pytesseract.image_to_data` and return both parsed time and OCR confidence. Add monotonic countdown enforcement in `main()` so bad reads are rejected and missing reads are interpolated. Verify ≥95% valid timestamps and a strictly monotonic time series on a sample clip.

## Step 9 — Sampling upgrade (pick sharpest frame per second)

### Objective
Reduce motion-blur and transitional HUD frames by sampling multiple frames around each second and selecting the sharpest. “Better” looks like higher-confidence detections without changing the overall 1Hz time resolution.

### Where to implement (files + functions)
- `app/ability_cv.py`: `extract_frames()` (or replace it), and the main loop that processes `frames`.
- To locate: search for `extract_frames(video_path, interval=1)` and `frame_count % frame_interval == 0`.

### Inputs / Outputs
- Inputs: video stream frames (via OpenCV capture).
- Outputs: a list of sampled frames per second (same length as before), but chosen by a sharpness metric; optionally also return frame indices/timestamps.

### Implementation checklist
- [ ] Change extraction to grab a short burst around each target second (e.g., ±2 frames or a 5-frame window).
- [ ] Compute sharpness per candidate frame (variance of Laplacian on grayscale).
- [ ] Choose the sharpest frame; keep the chosen frame’s original index for debugging.
- [ ] Ensure sampling is deterministic (no randomness).
- [ ] Handle edge cases: start/end of video, variable FPS, dropped frames.
- [ ] Optionally: expose a knob for compute vs quality (window size).

### Acceptance criteria
- Average detection confidence increases vs baseline on a sample clip (define: mean agent template score margin and mean charge `p_max`).
- The chosen-frame sharpness is higher than baseline sampled frame sharpness for ≥70% of seconds.
- Runtime overhead stays acceptable (e.g., <2× extraction time at 1Hz sampling).

### Test plan
- Unit: verify the sharpness scorer and frame selection logic chooses the expected frame in a synthetic blurred-vs-sharp pair.
- Golden-video: log sharpness per second for baseline vs upgraded sampling; compare confidence distributions.

### Risks / gotchas
- More decoding work increases CPU time; keep window small by default.
- Some HUD elements animate; “sharpest” may not always be the most semantically stable—pair with smoothing (Step 5).

### “Prompt to implement this step”
Implement Step 9 (sharpest-frame sampling). Update `app/ability_cv.py` to sample a small window of frames around each 1-second tick and select the sharpest frame using variance-of-Laplacian. Keep output cadence the same, log chosen indices and sharpness, and verify higher average detection confidence without excessive runtime overhead.

