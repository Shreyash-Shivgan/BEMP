-- Migration 0005: Phase 6 Gate Decisions, Conditions Audit, and Advisories

ALTER TABLE policies ADD COLUMN advisory TEXT;
ALTER TABLE policies ADD COLUMN gate_decision TEXT
    CHECK(gate_decision IN ('ACCEPT', 'REJECT', 'NO_ACTION_NEEDED'));
ALTER TABLE policies ADD COLUMN gate_conditions_json TEXT;

ALTER TABLE metrics ADD COLUMN gate_decision TEXT
    CHECK(gate_decision IN ('ACCEPT', 'REJECT', 'NO_ACTION_NEEDED'));
ALTER TABLE metrics ADD COLUMN gate_conditions_json TEXT;
