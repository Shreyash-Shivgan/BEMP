# StrikeShield: Semester VIII Handoff Document & Review Guide

## 1. Five-Point Verification Summary (Pre-Review Checklist)

- [x] **1a. Cold-Start Demo Path Verified**:
  - `uv run python -c "from strikeshield.db import init_db, migrate; init_db(); migrate()"` runs with exit code 0.
  - `uv run python scripts/run_iteration.py --loop --policy policies/v0.md --seeds 0,1,2 --max-iterations 2` executes cleanly across seeds `[0, 1, 2]`, promotes `v1`, and halts on `NO_ACTION_NEEDED`.
  - `uv run python -m strikeshield.report --run-id demo_cold_start` generates schema v6 `report_manifest.json`.
  - `uv run streamlit run dashboard/app.py` launches immediately at `http://localhost:8501`.
- [x] **1b. Dashboard Reachability in Under 30 Seconds**:
  - Persistent red `[SMOKE CORPUS — NOT FOR REPORTING]` banner is visible on the first view above all tabs.
  - Red `ISOLATED PROOF ONLY` badge on Ablation B is visible without scrolling on the Experiments panel (Tab 7).
  - Bold red card **$\kappa = \text{UNKNOWN}$** is prominent on the Judge Validation panel (Tab 6).
- [x] **1c. Claim / Cannot-Claim Section on One Slide**:
  - Exact text from Phase 9 §6 is mirrored verbatim below and ready to copy onto a single slide.
- [x] **1d. Every Slide Number Cites a Manifest Path**:
  - All metrics cite `runs/run_phase9_final/report_manifest.json` dot-paths (e.g., `policies.0.asr_all.mean`, `policies.1.gate_decision`).
- [x] **1e. exp_d Headline Demonstration**:
  - Single-seed candidate evaluation rejected end-to-end (`noise_floor_k=1<seeds=3` $\to$ `REJECT`), proving the Phase 6/7 gate fix end-to-end.

---

## 2. Three Carry-Forward Items for Semester VIII

### 2a. Real-Backend Runs Are the Top Priority
Every claim in the current repository that states "cannot claim end-to-end" is blocked on executing a live model run against a broader corpus:
- **Blocker**: Mock backend fixtures are deterministic and cannot generate a higher-ASR candidate or demonstrate heldout overfitting.
- **Semester VIII Milestone**: Execute a complete closed-loop run against Ollama (`LLM_BACKEND=ollama`) on the full 309-attack reference corpus.
- **Manifest Citations**:
  - `caveats[1]`: `backend=mock: target and judge are deterministic fixtures; real-model results pending`
  - `caveats[3]`: `ablation_a: not end-to-end demonstrated (mock backend too narrow)`
  - `caveats[4]`: `ablation_c: not end-to-end demonstrated (mock backend too narrow)`
  - `caveats[5]`: `ablation_b: not end-to-end demonstrated (mock backend cannot produce higher-ASR candidate); isolated proof only`

### 2b. Judge Validation Is the Second Priority
- **Blocker**: Judge inter-rater reliability is currently unverified against human annotators ($\kappa = \text{unknown}$).
- **Semester VIII Milestone**: Conduct a double-blind human annotation of 50 samples using `scripts/validate_judge.py` to establish a verified Cohen's $\kappa \ge 0.60$.
- **Manifest Citations**:
  - `judge_health.kappa_status`: `"unknown"`
  - `judge_health.kappa`: `null`
  - `caveats[2]`: `judge κ=unknown: human validation not yet run; Phase 8/9 status must flag this on every ASR claim`

### 2c. Attack Corpus Scale Expansion
- **Blocker**: Smoke corpus contains 26 attacks across 22 filled cells ($55.0\%$ of the 40-cell matrix).
- **Semester VIII Milestone**: Scale the attack corpus to the full 309-attack reference size across all 40 threat matrix cells using Promptfoo imports and automated stratified splitting.
- **Manifest Citations**:
  - `coverage.cells_filled`: `22`
  - `coverage.cells_total`: `40`
  - `coverage.corpus_role`: `"smoke"`
  - `caveats[0]`: `corpus_role=smoke: numbers are on a 26-attack smoke corpus, not the 309-attack reference corpus`

---

## 3. Review Q&A & Defense Scripts

1. **"What does the system actually do?"**
   > *"StrikeShield is CI for your safety layer: a closed-loop adversarial testing and policy self-hardening engine with a multi-condition promotion gate. It refuses to promote a candidate policy unless it improves security without degrading usability, introduces zero regressions, and verifies safety across multiple seeds to eliminate noise."*

2. **"Did it work?"**
   > *"Yes. On `exp_main`, baseline v0 ASR was 0.038 (3/78) and candidate v1 reduced ASR to 0.000 (0/78), preserved 1.000 BPR (30/30), passed all 7 gate conditions, and promoted v1 cleanly to `policies`. Most importantly, in `exp_d`, when tested on a single seed ($k=1$), the noise-floor gate fired an immediate hard rejection (`REJECT`), proving the fix to the reference deck's flaw works end-to-end."*

3. **"What did the reference deck get wrong, and did you fix it?"**
   - **Weak Patcher**: Fixed via structural diff validation, style-only rejection, and model escalation (`test_patcher_validation_rejects_style_only`).
   - **No Noise Floor**: Fixed via mandatory multi-seed gate checking ($k \ge 3$); demonstrated end-to-end by `exp_d` rejecting single-seed candidates.
   - **Security-Blind Gate**: Fixed via strict `asr_monotonic` check ($ASR_{cand} \le ASR_{base}$) and zero-regression budgets (`test_gate_rejects_asr_increase`, `test_ablation_b_gate_accepts_higher_asr`).

4. **"What can't you claim?"**
   > *"We CAN claim: the gate correctly rejects an ASR-increasing candidate (`test_gate_rejects_asr_increase`, `test_ablation_b_gate_accepts_higher_asr`). We CANNOT yet claim: the flaw was reproduced end-to-end through the pipeline on the mock corpus — the mock backend is deterministic and cannot produce a higher-ASR candidate. Reproducing the deck's 10.0%→11.3% failure end-to-end requires a real-model run (`LLM_BACKEND=ollama`), which is scheduled for Semester VIII."*

---

## 4. Single-Slide Claim / Cannot-Claim Thesis

```text
WHAT WE CAN CLAIM:
• Closed-loop hardening engine runs end-to-end across seeds [0, 1, 2].
• Candidate v1 reduces ASR (0.038 → 0.000), preserves BPR (1.000), introduces 0 regressions, and is promoted cleanly to policies (c9632d8).
• Gate provably rejects ASR-increasing candidates in isolated unit tests (test_gate_rejects_asr_increase, test_ablation_b_gate_accepts_higher_asr).
• Headline Gate Fix: Gate rejects single-seed evaluations end-to-end (noise_floor_k=1 < seeds=3 -> REJECT in exp_d).
• Git history is clean: exactly 1 initial commit + 1 commit per accepted version on refs/heads/policies.

WHAT WE CANNOT YET CLAIM:
• We CANNOT yet claim: the flaw was reproduced end-to-end through the pipeline on the mock corpus — the mock backend is deterministic and cannot produce a higher-ASR candidate. Reproducing the deck's 10.0%→11.3% failure end-to-end requires a real-model run (LLM_BACKEND=ollama), which is scheduled for Semester VIII.
• We CANNOT claim: heldout overfitting (ablation A) or granularity effect (ablation C) demonstrated end-to-end on mock backend.
• We CANNOT claim: production safety guarantees on the 309-attack reference corpus (current role: smoke, 26 attacks).
• We CANNOT claim: Judge validity against human annotators (kappa = unknown).
```
