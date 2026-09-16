## 1) Shared requirements

### Desired coaching output format (JSON schema)

Use a strict, machine-validated schema so coaching can be rendered consistently and audited back to the extracted signals.

**Schema (JSON Schema-style, simplified):**

```json
{
  "schema_version": "1.0",
  "video_id": "string",
  "timebase": { "units": "s", "direction": "elapsed|countdown", "t0_reference": "string" },
  "player": { "agent": "string|null" },
  "events": [
    {
      "event_id": "string",
      "timestamp_start": 0,
      "timestamp_end": 0,
      "event_type": "string",
      "severity": "low|medium|high",
      "confidence": 0.0,
      "entities": {
        "player_involved": true,
        "agents": ["string"],
        "teams": ["team|enemy"]
      },
      "evidence": [
        {
          "source": "game_status|slot_matches|ability_kb",
          "ref": "string",
          "timestamp": 0,
          "value": "any",
          "note": "string|null"
        }
      ],
      "coaching_message": "string",
      "suggested_alternative": "string|null",
      "debug": {
        "detector": "llm|rules",
        "rule_id": "string|null",
        "model": "string|null"
      }
    }
  ]
}
```

**Evidence citation requirements**
- Every event MUST include 2+ evidence items (except purely informational summaries).
- Each evidence item MUST cite:
  - `source`: which input file it comes from
  - `ref`: a stable pointer (JSON pointer-like string) to the field used (example: `game_status#/83/team/KAYO/alive`)
  - `timestamp`: the timestamp of the evidence snapshot
  - `value`: the observed value at that timestamp

### Requirements
- **Avoid hallucinations:** Only describe what can be grounded in evidence; if something is an inference, label it as such and lower confidence.
- **Handle noisy inputs:** Missing agents, misreads, flicker, and timestamp jitter are expected; pipeline must be robust to `unknown`/`null`.
- **Confidence scoring:** Provide a numeric confidence per event; incorporate input detector confidences if present, otherwise derive heuristically.
- **Temporal coherence:** Avoid repeated or contradictory advice; merge adjacent events of the same type when evidence is continuous.
- **Scalable cost:** Minimize LLM tokens by sending diffs/state-changes, not full per-frame JSON; chunk intelligently (time windows + event density).
- **Deterministic validation:** Always validate LLM output against schema and postprocess (dedupe, clamp timestamps, drop events with missing evidence).
- **Ability grounding:** Any ability-specific advice should be tied to `ability_description.json` entries used for that event.

### Inputs (current)
- `session/v720_game_status.json`: expected to contain (at least) `alive` and `ult_ready` per agent per timestamp (may be noisy).
- `session/v720_slot_matches.json`: ability charge counts for the player agent’s 4 abilities per timestamp (strings like `"2-2"`).
- `ability_description.json`: long-form tactical knowledge per agent ability.

### Core normalization step (shared)
Before any option:
- Parse timestamps to ints and sort in chronological order (handle countdown vs elapsed).
- Canonicalize agent/ability names (consistent IDs across JSON + ability KB).
- Convert charge strings like `"2-2"` into `{current: 2, max: 2}`.
- Compute derived series:
  - `team_alive_count(t)`, `enemy_alive_count(t)`
  - `player_alive(t)`, `player_ult_ready(t)`
  - `ability_current(t, ability_name)` for each player ability
  - `delta_ability(t) = ability_current(t) - ability_current(t+1)` (for countdown) or reverse for elapsed

---

## 2) Option A: Pure LLM (2–3 variants)

### Variant A1: Single-pass LLM per time chunk (diff-based input)
**Architecture diagram (text):**
- JSON ingest/normalize → chunk builder (diffs) → LLM (JSON output) → schema validation → merge/dedupe → final JSON

**Step-by-step pipeline:**
1. **Ingest:** load `game_status`, `slot_matches`, `ability_description`.
2. **Normalize:** canonical names; parse charges; compute derived counts (team/enemy alive).
3. **Chunk:** split into windows (e.g., 20–40s) with overlap (2–3s); represent as state-change stream:
   - changes in alive counts, player alive, ult_ready flips, ability charge drops/refills.
4. **Prompt:** send chunk + relevant ability KB snippets for the player’s agent.
5. **Parse:** parse the assistant output as JSON.
6. **Validate:** JSON schema validation; drop invalid events.
7. **Postprocess:** merge adjacent duplicate events; clamp timestamps to chunk range; compute final confidence caps.

**Prompt template (system + user)**

**System:**
```text
You are a VALORANT coaching engine. Output ONLY valid JSON that matches the provided schema.
Rules:
- Every event must cite evidence with source+ref+timestamp+value.
- Do not invent minimap/map/aim details; only use provided inputs.
- If evidence is weak/noisy, output fewer events with lower confidence.
```

**User:**
```text
Schema:
{OUTPUT_SCHEMA_JSON}

Inputs:
- game_status (diff stream): {GAME_STATUS_DIFF_JSON}
- slot_matches (diff stream): {SLOT_MATCHES_DIFF_JSON}
- ability knowledge (player agent only): {ABILITY_KB_SNIPPETS_JSON}

Task:
Generate timestamped coaching events for this chunk.
Constraints:
- Use event_type from this set: {EVENT_TYPE_ENUM}
- Provide 2+ evidence items per event.
- confidence in [0,1].
Return ONLY JSON.
```

**Pros/cons**
- Pros: simplest to ship; minimal code; adapts to new coaching styles quickly.
- Cons: higher hallucination risk; harder to guarantee coverage/consistency; token cost can grow with long videos.

**Expected accuracy**
- Medium: depends heavily on prompt quality, chunk representation (diffs), and strict post-validation.
Improves with: better diff encoding, few-shot examples, ability KB snippet retrieval, and tighter schema enforcement.

**Implementation difficulty:** Medium

**Failure modes**
- Hallucinated game context (“you rotated late”) without evidence.
- Missed events due to noisy/ambiguous diffs.
- Inconsistent event types/severities between chunks.

**Guardrails**
- Strict schema validation + drop events with missing evidence.
- Enforce `event_type` enum and timestamp bounds.
- Add a second pass: “self-check” JSON-only validation prompt when schema fails (no new content allowed).

---

### Variant A2: Two-pass LLM (extract events → coach)
**Architecture diagram (text):**
- ingest/normalize → chunk diffs → LLM#1 (event candidates, no prose) → validate → LLM#2 (write coaching text) → validate → final JSON

**Step-by-step pipeline:**
1. Build diff-stream chunks as in A1.
2. **LLM pass 1:** produce minimal event objects (type, timestamps, evidence refs, confidence), no coaching text.
3. Validate + dedupe event candidates deterministically.
4. **LLM pass 2:** for each validated event, provide ability KB snippets relevant to the event and ask for `coaching_message` + `suggested_alternative` only.
5. Validate final schema and postprocess.

**Prompt template**

**System (pass 1):**
```text
Extract coaching-relevant events from the provided diff stream.
Output ONLY JSON. No prose. No advice text. Evidence is mandatory.
```

**User (pass 1):**
```text
Schema (event-only subset):
{EVENT_ONLY_SCHEMA_JSON}

Diff stream:
{DIFF_STREAM_JSON}

Return only events with citations and confidence.
```

**System (pass 2):**
```text
You are a VALORANT coach. You will be given an event with evidence and ability KB snippets.
Write coaching_message and suggested_alternative grounded in the evidence.
Output ONLY JSON for the updated event.
```

**User (pass 2):**
```text
Event:
{EVENT_JSON}

Ability KB snippets:
{ABILITY_KB_SNIPPETS_JSON}

Fill in coaching_message and suggested_alternative. Do not change timestamps/type/evidence.
Return ONLY JSON.
```

**Pros/cons**
- Pros: reduces hallucination (pass 1 constrained); easier to test coverage; better maintainability.
- Cons: 2× LLM calls; needs careful dedupe/merge logic.

**Expected accuracy**
- Medium→High: pass 1 is structured extraction; pass 2 focuses on wording grounded in KB.
Improves with: better event-only schema and deterministic merge rules.

**Implementation difficulty:** High

**Failure modes**
- Pass 1 under-extracts (missed events) if diffs are too compressed.
- Pass 2 adds advice not supported by evidence (must be constrained).

**Guardrails**
- Pass 2 prompt forbids changing evidence and forbids adding new claims.
- Automatic rejection if coaching text introduces non-evidenced claims (optional: regex/LLM self-critique pass).

---

### Variant A3: Pure LLM with retrieval over ability_description.json (RAG-lite)
**Architecture diagram (text):**
- normalize → chunk diffs → ability KB retriever (by ability names + keywords) → LLM (JSON events) → validate → postprocess

**Step-by-step pipeline:**
1. Extract player agent abilities present in `slot_matches` keys.
2. Retrieve only the top relevant KB paragraphs for those abilities (and optionally “economy/ult timing” generic snippets).
3. Send chunk diffs + retrieved snippets to the LLM with A1 or A2 structure.

**Prompt template (delta from A1)**
- Replace `{ABILITY_KB_SNIPPETS_JSON}` with retrieved snippets only (e.g., 1–3 paragraphs per mentioned ability).

**Pros/cons**
- Pros: lower token cost than sending full KB; reduces generic advice; improves grounding.
- Cons: retrieval mistakes can omit key guidance; still LLM-driven event detection.

**Expected accuracy**
- Medium: improves ability-specific coaching quality more than event detection quality.

**Implementation difficulty:** Medium (simple keyword-based retrieval is sufficient to start)

**Failure modes**
- Retrieved snippet mismatch due to naming inconsistencies.
- Over-reliance on KB causes repetitive advice across events.

**Guardrails**
- Canonicalize ability names before retrieval.
- Cache retrieved snippets per agent per video.
- Add “don’t repeat yourself” postprocess: de-duplicate near-identical coaching messages.

---

## 3) Option B: Hybrid (at least 3 variants)

### Variant B1: Deterministic rule engine → LLM writes explanations
**Event detector design**
- Represent state at time `t`:
  - `player_alive(t)`, `player_ult_ready(t)`, `team_alive_count(t)`, `enemy_alive_count(t)`
  - `ability_current(t, a)` for each player ability `a`
- Rules run on a sliding window (e.g., 5–15s) and detect discrete events:
  - charge drops (ability used), ult flips, death moments, “held resources” windows.
- Each rule outputs an **event candidate**: `{event_type, ts_start, ts_end, evidence[], confidence_base}`.

**Uncertainty representation**
- Base confidence from:
  - stability: how many consecutive timestamps support the condition
  - magnitude: size of charge drop (e.g., 2→0 is stronger than 1→0)
  - consistency: no contradictory reads in the window
  - optional detector confidences if present in JSON (future)
- Final event confidence = `confidence_base * evidence_quality_factor`.

**How the LLM is used**
- Input: one event candidate + a small set of KB snippets for the ability/ult involved.
- Output: fill `coaching_message`, `suggested_alternative`, and optionally adjust `severity` (bounded).
- LLM is NOT allowed to create new events or change evidence.

**Pros/cons**
- Pros: low hallucination; highly testable; cost scales with number of detected events (not video length).
- Cons: coverage limited to implemented rules; initial rule set needs iteration.

**Expected accuracy and why**
- High for the events you encode (precision); moderate recall initially.

**Implementation difficulty:** Medium

**Failure modes + mitigations**
- Missed subtle situations → add more rules gradually; log “near-miss” diagnostics.
- Noisy charge flicker creates false “ability used” → require N-frame confirmation and use debouncing.

---

### Variant B2: Hybrid with scoring + prioritization + de-duplication (productized)
**Event detector design**
- Same rule engine as B1, but:
  - Each rule emits `impact_score` (0–100) derived from:
    - numbers state (clutch vs 5v5)
    - resource importance (ult > key utility)
    - timing (late-round > early if it’s the last chance)
  - Add a de-dup layer that merges overlapping events of the same type and chooses the strongest evidence.

**Uncertainty representation**
- Keep both:
  - `confidence` (how sure we are it happened)
  - `impact_score` (how much it matters if true)
- Use conservative messaging when confidence is low (“Consider…” vs “You should have…”).

**How the LLM is used**
- LLM writes explanations AND short “next time” action steps.
- LLM also re-ranks within a small candidate set (top K) but cannot invent new candidates.

**Pros/cons**
- Pros: better UX (fewer, higher-value insights); controlled cost; consistent structure.
- Cons: more engineering (scoring tuning, dedupe rules, thresholds).

**Expected accuracy and why**
- High perceived quality because you only show top, well-supported events.

**Implementation difficulty:** High

**Failure modes + mitigations**
- Over-aggressive pruning hides useful advice → keep a “show more” tier with lower-impact events.
- Repeated coaching across rounds → add cross-round memory to avoid repeating identical tips too often.

---

### Variant B3: Hybrid with pattern library (retrieval) + LLM personalization
**Event detector design**
- Same as B1/B2 to produce structured events with evidence.
- Add a pattern library keyed by `(event_type, agent, ability)` with:
  - a short title
  - coaching template text
  - “why it matters”
  - recommended alternatives
  - optional examples

**Uncertainty representation**
- Rule confidence determines how strongly to assert the coaching.
- Pattern library provides safe, non-hallucinated defaults when LLM is skipped or fails.

**How the LLM is used**
- Retrieve 1–3 best patterns for the event.
- LLM personalizes tone and references the specific evidence timestamps.
- If LLM fails validation, fall back to the pattern template (deterministic).

**Pros/cons**
- Pros: most consistent; lowest hallucination; easy to keep brand voice; graceful fallback.
- Cons: requires curating/maintaining a pattern library; can feel repetitive without enough variants.

**Expected accuracy and why**
- High precision and consistency; limited by event detector recall and pattern coverage.

**Implementation difficulty:** High

**Failure modes + mitigations**
- Pattern mismatch due to naming/ID issues → canonicalization + strict keys.
- Stale patterns after patches/meta changes → version patterns and update periodically.

---

## 4) Recommendation

**Best MVP:** Variant **B1** (rules → LLM writes explanations)
- Fast to ship with strong guardrails and low hallucination risk.
- Token cost scales with events, not video length.
- Easy to test rule-by-rule with unit tests and golden JSON fixtures.
- Lets you iterate on extraction noise handling without changing prompts constantly.
- Ability KB can be injected per-event, keeping prompts short and relevant.

**Best production-grade:** Variant **B2 + B3** (scored hybrid + pattern fallback, optional LLM personalization)
- Deterministic event coverage + prioritization yields stable UX.
- Pattern library provides safety and consistency; LLM becomes an enhancer, not a dependency.
- Supports strict SLAs (latency/cost) by skipping LLM for low-priority events.
- Easier to monitor quality regressions via rule metrics and pattern coverage.
- De-duplication and “don’t repeat yourself” can be enforced centrally.

---

## 5) Concrete “event types” doable with current signals

Below are event types detectable using ONLY:
- `session/v720_game_status.json` (alive/dead + ult status)
- `session/v720_slot_matches.json` (player ability charge counts)
- `ability_description.json` (ability guidance)

For each, assume a consistent time direction (convert countdown/elapsed first) and use debouncing (e.g., require 2 consecutive timestamps).

### 1) Ability used (charge drop)
- **Trigger logic:** An ability’s `current_charges` decreases between adjacent timestamps (e.g., 2→1 or 1→0).
- **Evidence fields used:** `slot_matches#/t/<AbilityName>` at `t` and `t+1`.
- **Confidence heuristic:** High if the drop is stable for ≥2 samples and not immediately reverted; lower if flickering.

### 2) Multi-utility burst (“utility dump”)
- **Trigger logic:** Two or more abilities decrease within a short window (e.g., ≤5s), or one ability drops by ≥2 across the window.
- **Evidence fields used:** multiple `slot_matches` deltas across timestamps in the window.
- **Confidence heuristic:** Medium→High if charge drops are large and stable; medium if only small drops and noisy series.

### 3) Utility hoard (stays max too long)
- **Trigger logic:** One or more abilities remain at max charges for ≥X seconds while the round progresses.
- **Evidence fields used:** `slot_matches` for that ability across `[t_start..t_end]`.
- **Confidence heuristic:** Medium; raise confidence if the player is alive and multiple abilities remain capped.

### 4) Died with utility available
- **Trigger logic:** Player transitions `alive: true → false` while having ≥1 ability with `current_charges > 0`.
- **Evidence fields used:** `game_status#/t/.../alive` (player agent) and `slot_matches#/t/...`.
- **Confidence heuristic:** High if death transition is stable and charge values are stable in the ±2s window.

### 5) Died with ultimate ready (unused ult)
- **Trigger logic:** Player transitions `alive: true → false` while `ult_ready == true` (and no `true→false` ult flip prior).
- **Evidence fields used:** `game_status#/t/.../alive`, `game_status#/t/.../ult_ready`.
- **Confidence heuristic:** Medium→High; lower if ult_ready flickers near death.

### 6) Ultimate became ready (plan opportunity)
- **Trigger logic:** Player `ult_ready` flips `false → true`.
- **Evidence fields used:** `game_status#/t/.../ult_ready` at `t` and `t+1`.
- **Confidence heuristic:** High if it stays true for ≥N seconds.

### 7) Ultimate used (timing marker)
- **Trigger logic:** Player `ult_ready` flips `true → false` while player is alive.
- **Evidence fields used:** `game_status#/t/.../ult_ready`, `game_status#/t/.../alive`.
- **Confidence heuristic:** Medium; higher if the flip persists and doesn’t flicker back.

### 8) Held ultimate too long
- **Trigger logic:** Player `ult_ready == true` continuously for ≥X seconds without a `true→false` flip.
- **Evidence fields used:** `game_status` ult_ready across window.
- **Confidence heuristic:** Medium; increase when team/enemy alive counts suggest high-leverage phase (e.g., 3v3 or worse).

### 9) Team ult stack (coordination opportunity)
- **Trigger logic:** At timestamp `t`, count of teammates with `ult_ready == true` is ≥K (e.g., 2 or 3).
- **Evidence fields used:** `game_status#/t/team/<Agent>/ult_ready` for all team agents.
- **Confidence heuristic:** Medium; higher if the condition persists for several seconds.

### 10) Enemy ult threat window
- **Trigger logic:** At timestamp `t`, count of enemies with `ult_ready == true` is ≥K.
- **Evidence fields used:** `game_status#/t/enemy/<Agent>/ult_ready` for all enemy agents.
- **Confidence heuristic:** Medium; higher if sustained.

### 11) Clutch state detected (player in 1vX or Xv1)
- **Trigger logic:** Based on alive counts: `team_alive_count == 1` and `enemy_alive_count >= 2` (or inverse).
- **Evidence fields used:** `game_status` alive flags for all agents at `t`.
- **Confidence heuristic:** High if alive flags are stable across ±2s.

### 12) Resource mismatch in disadvantage (had util but down numbers)
- **Trigger logic:** `team_alive_count < enemy_alive_count` AND player has multiple abilities available (e.g., ≥2 abilities with charges > 0) for a sustained window.
- **Evidence fields used:** `game_status` alive counts + `slot_matches` charges across the window.
- **Confidence heuristic:** Medium; higher if window is long and charge series stable.

### 13) Low-impact ability usage (spent when already dead/down)
- **Trigger logic:** Ability charge drops occur in the same window where player is dead or the player dies immediately after (≤2s).
- **Evidence fields used:** `slot_matches` deltas + `game_status` player alive transition.
- **Confidence heuristic:** Low→Medium (since “impact” is inferred); only emit when timing is tight and signals are stable.

### 14) Ability-specific missed opportunity (KB-grounded)
- **Trigger logic:** An ability remains unused (at/near max) during a high-leverage state (clutch, or team down numbers) for ≥X seconds.
- **Evidence fields used:** `slot_matches` charges + `game_status` alive counts + `ability_kb` entry for that ability.
- **Confidence heuristic:** Medium; raise if the state persists and charge readings are stable.

Notes on evidence refs:
- Use refs like `slot_matches#/83/FLASH/drive` and `game_status#/83/team/KAYO/alive` (adapt to your actual schema).
- Always include evidence snapshots for both sides of a transition (e.g., before/after for a flip).
