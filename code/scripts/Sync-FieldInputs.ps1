[CmdletBinding(DefaultParameterSetName='Selected')]
param(
    [Parameter(Mandatory=$true,ParameterSetName='Selected')][string[]]$Clip,
    [Parameter(Mandatory=$true,ParameterSetName='All')][switch]$All,
    [string]$Workspace=(Split-Path $PSScriptRoot -Parent)
)
$ErrorActionPreference='Stop'
$Workspace=[IO.Path]::GetFullPath($Workspace)
$prefix=$Workspace.TrimEnd('\')+'\'
function SafePath([string]$relative) {
    $p=[IO.Path]::GetFullPath((Join-Path $Workspace $relative))
    if(-not $p.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw "Outside workspace: $relative"}
    return $p
}
$catalog=@(Import-Csv -LiteralPath (SafePath 'catalogs/field_clips.csv'))
if(-not $All){
    foreach($id in $Clip){if($id -notin $catalog.clip_id){throw "Unknown clip: $id"}}
    $catalog=@($catalog | Where-Object clip_id -in $Clip)
}
$destination=SafePath 'trajectory/Camera data'
[IO.Directory]::CreateDirectory($destination)|Out-Null
$manifestPath=SafePath 'catalogs/runtime_inputs.csv'
$mapped=@{}
if(Test-Path -LiteralPath $manifestPath){foreach($entry in Import-Csv -LiteralPath $manifestPath){$mapped[$entry.runtime_path]=$entry}}
$copied=0;$reused=0
foreach($row in $catalog){
    foreach($kind in @('video','gps')){
        if($kind -eq 'gps' -and $row.gps_status -ne 'present'){continue}
        $sourceRel=$row.($kind+'_path');$targetRel=$row.('runtime_'+$kind+'_path');$expected=$row.($kind+'_sha256')
        if(-not $targetRel.StartsWith('trajectory/Camera data/',[StringComparison]::Ordinal)){throw "Unexpected runtime path: $targetRel"}
        $source=SafePath $sourceRel;$target=SafePath $targetRel
        $hash=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
        if($hash -ne $expected){throw "Source changed since catalog creation: $sourceRel"}
        if(Test-Path -LiteralPath $target){
            if((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected){throw "Existing runtime file differs; not overwritten: $targetRel"}
            $reused++
        }else{
            [IO.File]::Copy($source,$target,$false)
            if((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected){throw "Copy verification failed: $targetRel"}
            $copied++
        }
        $mapped[$targetRel]=[pscustomobject][ordered]@{clip_id=$row.clip_id;kind=$kind;source_path=$sourceRel;runtime_path=$targetRel;sha256=$expected;method='independent copy'}
    }
}
$mapped.Values | Sort-Object runtime_path | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8
Write-Output "$copied files copied, $reused existing files verified. Runtime input manifest: catalogs/runtime_inputs.csv"