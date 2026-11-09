param(
    [string]$Workspace = (Join-Path $PSScriptRoot '..'),
    [int]$Port = 8765,
    [string]$Project = ''
)
$ErrorActionPreference = 'Stop'
$reviewWorkspace = (Resolve-Path -LiteralPath $Workspace).Path
$reviewPython = Join-Path $reviewWorkspace '.venv-bev/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $reviewPython)) {
    $reviewPython = (Get-Command python -ErrorAction Stop).Source
}
# Foreground process: VS Code sees its URL and Ctrl+C stops it. No remote browser.
$reviewArgs = @((Join-Path $PSScriptRoot 'server.py'), '--workspace', $reviewWorkspace, '--port', $Port)
if ($Project) { $reviewArgs += @('--project', $Project) }
& $reviewPython @reviewArgs
