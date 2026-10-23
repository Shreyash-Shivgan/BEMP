# StrikeShield: CI for your Safety Layer

Automated adversarial testing, closed-loop policy self-hardening, and verifiable evaluation for LLM guardrails.

## Features
- **Deterministic 5×8 Threat Matrix:** 5 risk categories × 8 adversarial technique families.
- **Local Ollama Inference:** Sequential VRAM eviction via `keep_alive: 0`.
- **Closed-Loop Hardening:** Technique-level patcher with escalation and hard gate invariants.
- **Multi-Seed Noise Floor:** Tracks ASR/BPR confidence bounds ($K=3$ seeds).
- **Audit Traceability:** SQLite audit log with migration history and Git commit tracking on `policies` branch.

## Quickstart
```bash
# Install dependencies using uv
uv sync --extra dev

# Run test suite in mock mode
uv run pytest -q
```
