# Starts the SARP-Agent model gateway and the ngrok tunnel on the team PC, so the online website can use
# the AI model running on this PC's GPU.
#
# Usage (PowerShell, from anywhere):
#   $env:SARP_GATEWAY_KEY = "<the same key as SARP_LLM_API_KEY on Vercel>"
#   .\agent\start_gateway.ps1 -Domain your-name.ngrok-free.app
#
# Needs: Ollama running with the model pulled (ollama pull gemma3:4b), ngrok installed and logged in
# (ngrok config add-authtoken <token>) with a free static domain claimed in the ngrok dashboard.
param(
    [string]$Domain = $env:SARP_NGROK_DOMAIN,
    [int]$Port = 8787
)
$ErrorActionPreference = "Stop"

if (-not $env:SARP_GATEWAY_KEY) {
    Write-Error "Set SARP_GATEWAY_KEY first, for example: `$env:SARP_GATEWAY_KEY = '<key>'"
    exit 1
}
if (-not $Domain) {
    Write-Error "Pass your free ngrok static domain: -Domain your-name.ngrok-free.app"
    exit 1
}
if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
    Write-Error "ngrok is not installed. Install it with: winget install ngrok.ngrok"
    exit 1
}

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
$env:SARP_GATEWAY_PORT = "$Port"

Write-Host "Starting the model gateway on http://127.0.0.1:$Port ..."
$gateway = Start-Process -FilePath $python -ArgumentList "-m", "agent.llm_gateway" -WorkingDirectory $repo -PassThru -NoNewWindow
Start-Sleep -Seconds 4

Write-Host "Opening the tunnel at https://$Domain  (set SARP_LLM_BASE_URL=https://$Domain/v1 on Vercel)"
Write-Host "Press Ctrl+C to stop both."
try {
    ngrok http $Port --url "https://$Domain"
}
finally {
    if ($gateway -and -not $gateway.HasExited) { Stop-Process -Id $gateway.Id }
}
