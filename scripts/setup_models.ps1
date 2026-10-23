# scripts/setup_models.ps1 - Idempotently pull models required by StrikeShield in PowerShell
$ErrorActionPreference = "Stop"

$models = @(
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "mistral:7b-instruct-q4_K_M"
)

Write-Host "=== StrikeShield Ollama Model Setup ==="

if (-not (Get-Command "ollama" -ErrorAction SilentlyContinue)) {
    Write-Error "Ollama command not found. Please install Ollama from https://ollama.com"
    exit 1
}

$installed = ollama list

foreach ($model in $models) {
    if ($installed -match [regex]::Escape($model)) {
        Write-Host "[OK] Model already installed: $model" -ForegroundColor Green
    } else {
        Write-Host "[PULLING] Downloading model: $model ..." -ForegroundColor Yellow
        ollama pull $model
        Write-Host "[OK] Successfully installed: $model" -ForegroundColor Green
    }
}

Write-Host "=== All required models are verified and ready. ===" -ForegroundColor Cyan
