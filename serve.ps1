#!/usr/bin/env pwsh
# Serve the browser radio locally.
#
# The page fetches the channel sources as siblings of index.html, so this
# rebuilds a flat _site/ folder (so edits to web/ or the channels are always
# picked up) and serves it. Stop with Ctrl+C.
#
# Usage:
#   ./serve.ps1                 # serve on 127.0.0.1:41001, open the browser
#   ./serve.ps1 -Port 41005     # use a different port
#   ./serve.ps1 -NoOpen         # don't auto-open the browser

param(
    [int]$Port = 41001,
    [switch]$NoOpen
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$site = Join-Path $root '_site'

# rebuild the flat site the page expects
if (Test-Path $site) { Remove-Item $site -Recurse -Force }
New-Item -ItemType Directory -Path $site | Out-Null
Copy-Item (Join-Path $root 'web/index.html') $site
Copy-Item (Join-Path $root 'web/radio.js') $site
Copy-Item (Join-Path $root 'radio_core.py') $site
Copy-Item (Join-Path $root 'channel_*.py') $site

$py = Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue | Select-Object -First 1 }
if (-not $py) { throw 'python not found on PATH' }

$url = "http://127.0.0.1:$Port/"
Write-Host "serving radio at $url  (Ctrl+C to stop)" -ForegroundColor Green

# open the browser shortly after the server has had time to bind
if (-not $NoOpen) {
    Start-Job -ArgumentList $url {
        param($u)
        Start-Sleep -Milliseconds 800
        Start-Process $u
    } | Out-Null
}

& $py.Source -m http.server $Port -b 127.0.0.1 -d $site
