param(
    [Parameter(Position=0)]
    [ValidateSet('review','verify','check-models','check-3s','identify','doctor','demo','test','release','inventory','paper','environment','python')]
    [string]$Action='environment',
    [ValidateSet('auto','bev','classifier')]
    [string]$Environment='auto',
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ScriptArguments
)
$ErrorActionPreference='Stop'
$codeRoot=$PSScriptRoot
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TORCH_HOME=Join-Path $codeRoot '.runtime/torch-cache'
$runtimeConfig=Get-Content -LiteralPath (Join-Path $codeRoot 'configs/environments.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if($Action -eq 'environment') {
    foreach($name in @('bev','classifier')) {
        $entry=$runtimeConfig.environments.$name
        [pscustomobject]@{Environment=$name;Python=(Join-Path $codeRoot $entry.python);Torch=$entry.torch;Use=$entry.purpose}
    }
    return
}
$selected=$Environment
if($selected -eq 'auto') {$selected=if($Action -in @('check-models','identify','doctor','demo','test','release','inventory')){'classifier'}else{'bev'}}
$pythonPath=Join-Path $codeRoot $runtimeConfig.environments.$selected.python
if(-not(Test-Path -LiteralPath $pythonPath)){throw "Missing interpreter: $pythonPath"}
Push-Location -LiteralPath $codeRoot
try {
    switch($Action) {
        'review' { & $pythonPath 'review_web/server.py' '--port' '8765' @ScriptArguments }
        'verify' { & powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'scripts/Test-Workspace.ps1' @ScriptArguments }
        'check-models' { & $pythonPath 'classification/trajectory-avlike-transformer/smoke_models.py' @ScriptArguments }
        'check-3s' { & $pythonPath 'scripts/validate_three_second.py' @ScriptArguments }
        'identify' { & $pythonPath 'scripts/frozen_classifier.py' @ScriptArguments }
        {$_ -in @('doctor','demo','test','release','inventory')} { & $pythonPath 'manage.py' '--workspace' $codeRoot $Action @ScriptArguments }
        'paper' { & $pythonPath 'scripts/build_field_paper.py' @ScriptArguments }
        'python' {
            if(-not $ScriptArguments){throw 'Provide a script path or Python arguments after python.'}
            & $pythonPath @ScriptArguments
        }
    }
    $resultCode=$LASTEXITCODE
} finally {Pop-Location}
exit $resultCode
