-- Migration 0002: Phase 3 Target and Judge Audit Trail

ALTER TABLE metrics ADD COLUMN backend TEXT;
ALTER TABLE metrics ADD COLUMN corpus_role TEXT;

ALTER TABLE responses ADD COLUMN seed INTEGER;
ALTER TABLE responses ADD COLUMN temperature REAL;

CREATE TABLE IF NOT EXISTS judge_calls (
    id TEXT PRIMARY KEY,
    response_id TEXT NOT NULL,
    raw_output TEXT NOT NULL,
    parsed_json TEXT,
    schema_error TEXT,
    evidence_check_passed INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(response_id) REFERENCES responses(id)
);

CREATE TABLE IF NOT EXISTS evidence_mismatches (
    id TEXT PRIMARY KEY,
    verdict_id TEXT NOT NULL,
    quoted TEXT NOT NULL,
    response_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(verdict_id) REFERENCES verdicts(id)
);

CREATE INDEX IF NOT EXISTS idx_judge_calls_resp ON judge_calls(response_id);
CREATE INDEX IF NOT EXISTS idx_evidence_mismatches_verdict ON evidence_mismatches(verdict_id);
