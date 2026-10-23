#!/usr/bin/env bash
# scripts/setup_models.sh - Idempotently pull models required by StrikeShield

set -euo pipefail

MODELS=(
    "llama3.1:8b-instruct-q4_K_M"
    "qwen2.5:7b-instruct-q4_K_M"
    "mistral:7b-instruct-q4_K_M"
)

echo "=== StrikeShield Ollama Model Setup ==="

if ! command -v ollama &> /dev/null; then
    echo "ERROR: 'ollama' command not found. Please install Ollama from https://ollama.com"
    exit 1
fi

INSTALLED_MODELS=$(ollama list)

for MODEL in "${MODELS[@]}"; do
    if echo "$INSTALLED_MODELS" | grep -q "$MODEL"; then
        echo "[OK] Model already installed: $MODEL"
    else
        echo "[PULLING] Downloading model: $MODEL ..."
        ollama pull "$MODEL"
        echo "[OK] Successfully installed: $MODEL"
    fi
done

echo "=== All required models are verified and ready. ==="
