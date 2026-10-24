-- Migration 0006_phase7.sql: Phase 7 closed loop and multi-seed noise floor schema updates

-- 1. v0 is baseline, not a candidate
UPDATE policies SET candidate_status='baseline' WHERE version='v0';

-- 2. Multi-seed metrics columns
ALTER TABLE metrics ADD COLUMN seeds_run_json TEXT;
ALTER TABLE metrics ADD COLUMN asr_all_range_low REAL;
ALTER TABLE metrics ADD COLUMN asr_all_range_high REAL;
ALTER TABLE metrics ADD COLUMN asr_patch_range_low REAL;
ALTER TABLE metrics ADD COLUMN asr_patch_range_high REAL;
ALTER TABLE metrics ADD COLUMN asr_heldout_range_low REAL;
ALTER TABLE metrics ADD COLUMN asr_heldout_range_high REAL;
ALTER TABLE metrics ADD COLUMN bpr_range_low REAL;
ALTER TABLE metrics ADD COLUMN bpr_range_high REAL;
ALTER TABLE metrics ADD COLUMN noise_overlap_asr INTEGER;

-- 3. Policy tracking columns
ALTER TABLE policies ADD COLUMN seeds_run_json TEXT;
ALTER TABLE policies ADD COLUMN promotion_commit TEXT;
