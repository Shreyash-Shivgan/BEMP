-- Migration 0003: Phase 4 Evaluator, Regressions, and Judge Health Tracking

ALTER TABLE metrics ADD COLUMN judge_health_json TEXT;
ALTER TABLE metrics ADD COLUMN noise_floor_k INTEGER DEFAULT 1;
ALTER TABLE metrics ADD COLUMN regression_count INTEGER DEFAULT 0;
