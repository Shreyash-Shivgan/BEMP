-- Migration 0004: Phase 5 Patcher Candidates, Escalation, and Regressed IDs Tracking

ALTER TABLE metrics ADD COLUMN regressed_attack_ids_json TEXT;
ALTER TABLE policies ADD COLUMN candidate_status TEXT DEFAULT 'candidate';
ALTER TABLE policies ADD COLUMN patcher_model TEXT;
ALTER TABLE policies ADD COLUMN patcher_fallback_used INTEGER DEFAULT 0;
