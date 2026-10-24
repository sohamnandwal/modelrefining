# setup_windows.ps1
# Configures Windows WSL2 memory limits to prevent Linux OOM Killer (exit code 137)
# on 16GB RAM machines during model alignment and 16-bit LoRA merging.

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " Setting up Windows WSL2 configuration for 16GB RAM Host" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

$wslConfigPath = "$env:USERPROFILE\.wslconfig"
$wslConfigContent = @"
[wsl2]
memory=12GB
swap=16GB
localhostForwarding=true
nestedVirtualization=true
"@

Set-Content -Path $wslConfigPath -Value $wslConfigContent
Write-Host "Successfully wrote memory=12GB and swap=16GB to $wslConfigPath" -ForegroundColor Green
Write-Host "To apply the new memory settings, run: wsl --shutdown" -ForegroundColor Yellow
