-- Migration 0001: Initial StrikeShield Schema

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    seed INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    git_commit TEXT,
    parent_version TEXT,
    diff_text TEXT,
    status TEXT DEFAULT 'ACCEPTED', -- 'ACCEPTED' or 'REJECTED'
    rejection_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS attacks (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    risk_category TEXT NOT NULL,
    technique_family TEXT NOT NULL,
    target_behaviour TEXT NOT NULL,
    embedding BLOB,
    cell_id TEXT NOT NULL,
    split TEXT NOT NULL CHECK(split IN ('patch', 'heldout')),
    source TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS responses (
    id TEXT PRIMARY KEY,
    attack_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    response_text TEXT NOT NULL,
    run_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(attack_id) REFERENCES attacks(id),
    FOREIGN KEY(policy_version) REFERENCES policies(version),
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS verdicts (
    id TEXT PRIMARY KEY,
    response_id TEXT NOT NULL,
    verdict_class TEXT NOT NULL CHECK(verdict_class IN (
        'full_compliance',
        'partial_leak',
        'refused_but_leaked',
        'clean_refusal',
        'off_target'
    )),
    evidence_quote TEXT,
    confidence REAL NOT NULL,
    judge_model TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(response_id) REFERENCES responses(id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_version TEXT NOT NULL,
    attack_id TEXT NOT NULL,
    blocked INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(policy_version) REFERENCES policies(version),
    FOREIGN KEY(attack_id) REFERENCES attacks(id),
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    asr_all REAL NOT NULL,
    asr_patch REAL NOT NULL,
    asr_heldout REAL NOT NULL,
    bpr REAL NOT NULL,
    regressions INTEGER NOT NULL,
    grid_coverage REAL NOT NULL,
    mean_pairwise_dist REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES runs(id),
    FOREIGN KEY(policy_version) REFERENCES policies(version)
);

CREATE TABLE IF NOT EXISTS judge_validation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id TEXT NOT NULL,
    human_label TEXT NOT NULL CHECK(human_label IN (
        'full_compliance',
        'partial_leak',
        'refused_but_leaked',
        'clean_refusal',
        'off_target'
    )),
    annotator TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(response_id) REFERENCES responses(id)
);

-- Indices for rapid audit and replay queries
CREATE INDEX IF NOT EXISTS idx_attacks_cell ON attacks(cell_id);
CREATE INDEX IF NOT EXISTS idx_attacks_split ON attacks(split);
CREATE INDEX IF NOT EXISTS idx_responses_attack ON responses(attack_id);
CREATE INDEX IF NOT EXISTS idx_responses_policy ON responses(policy_version);
CREATE INDEX IF NOT EXISTS idx_evaluations_run_policy ON evaluations(run_id, policy_version);
