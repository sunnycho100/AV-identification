param([string]$Workspace=(Split-Path $PSScriptRoot -Parent))
$ErrorActionPreference='Stop'
$Workspace=[IO.Path]::GetFullPath($Workspace)
$projectBoundary=[IO.Path]::GetFullPath((Join-Path $Workspace '..'))
$prefix=$projectBoundary.TrimEnd('\')+'\'
$failures=[Collections.Generic.List[string]]::new()
$cache=@{}
$changes=@{}
$changePath=Join-Path $Workspace 'catalogs/workspace_changes.json'
if(Test-Path -LiteralPath $changePath){
    foreach($change in (Get-Content -LiteralPath $changePath -Raw -Encoding UTF8 | ConvertFrom-Json)){$changes[$change.path]=$change}
}
function SafePath([string]$relative){
    switch -Regex ($relative.Replace('\','/')) {
        '^paper/(.*)$' {$relative='../paper/'+$Matches[1];break}
        '^archives/(.*)$' {$relative='../archive/source_packages/'+$Matches[1];break}
        '^revisions/(.*)$' {$relative='../archive/revisions/'+$Matches[1];break}
        '^README.txt$' {$relative='docs/history/TRAINING_DATA_HANDOFF_20261101.txt';break}
        '^README.md$' {$relative='docs/PROJECT_HISTORY.md';break}
        '^leader_review_decisions.json$' {$relative='data/annotations/leader_review_decisions.json';break}
        '^trajectory_extension_review.json$' {$relative='data/annotations/trajectory_extension_review.json';break}
    }
    $p=[IO.Path]::GetFullPath((Join-Path $Workspace $relative))
    if(-not $p.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw "Outside workspace: $relative"}
    return $p
}
function CheckFile([string]$relative,[string]$expected){
    $p=SafePath $relative
    if(-not(Test-Path -LiteralPath $p -PathType Leaf)){$failures.Add("Missing: $relative");return}
    if(-not $cache.ContainsKey($p)){$cache[$p]=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
    if($cache[$p] -ne $expected.ToLowerInvariant()){
        if($changes.ContainsKey($relative)){
            $change=$changes[$relative]
            if($change.original_sha256 -ne $expected.ToLowerInvariant()){$failures.Add("Original revision hash mismatch: $relative")}
            CheckFile $change.backup $expected
            if($cache[$p] -ne $change.current_sha256){$failures.Add("Current revision hash mismatch: $relative")}
        }else{$failures.Add("SHA256 mismatch: $relative")}
    }
}
$archives=Get-Content -LiteralPath (SafePath 'catalogs/archives.json') -Raw -Encoding UTF8 | ConvertFrom-Json
foreach($a in $archives){CheckFile $a.path $a.sha256}
$source=@(Import-Csv -LiteralPath (SafePath 'catalogs/archive_files.csv'))
foreach($f in $source){
    CheckFile $f.path $f.sha256
    $p=SafePath $f.path
    $expectedBytes=[long]$f.bytes
    if($changes.ContainsKey($f.path)){$expectedBytes=[long]$changes[$f.path].current_bytes}
    if((Test-Path -LiteralPath $p) -and (Get-Item -LiteralPath $p).Length -ne $expectedBytes){$failures.Add("Size mismatch: $($f.path)")}
}
$expectedSourceCount=($archives | Measure-Object extracted_files -Sum).Sum
if($source.Count -ne $expectedSourceCount){$failures.Add('Extracted source count differs from archive manifest')}
if($archives.Count -ne 3){$failures.Add('Expected three source archives')}
$original=@(Get-Content -LiteralPath (SafePath 'docs/migration/SHA256SUMS.txt'))
foreach($line in $original){
    $parts=$line -split '  ',2
    if($parts.Count -ne 2){throw 'Invalid original checksum row'}
    $rel=$parts[1]
    if($rel -notmatch '^(paper|trajectory)/'){$rel='docs/migration/'+$rel}
    CheckFile $rel $parts[0]
}
$classOriginal=@(Get-Content -LiteralPath (SafePath 'docs/classification_migration/SHA256SUMS.txt'))
foreach($line in $classOriginal){
    $parts=$line -split '  ',2
    if($parts.Count -ne 2){throw 'Invalid classification checksum row'}
    $rel=$parts[1]
    if($rel.StartsWith('trajectory-avlike-transformer/',[StringComparison]::Ordinal)){$rel='classification/'+$rel}
    else{$rel='docs/classification_migration/'+$rel}
    CheckFile $rel $parts[0]
}
$classRoot=SafePath 'classification/trajectory-avlike-transformer'
$neural=@(Get-ChildItem -LiteralPath (Join-Path $classRoot 'artifacts') -Recurse -File -Filter 'best_bce.pt')
$forests=@(Get-ChildItem -LiteralPath (Join-Path $classRoot 'artifacts') -Recurse -File -Filter 'model.joblib')
$deploy=@(Get-ChildItem -LiteralPath (Join-Path $classRoot 'inference_cnn/models') -File -Filter '*.pt')
if($neural.Count -ne 60 -or $forests.Count -ne 15 -or $deploy.Count -ne 15){$failures.Add('Expected 60 neural, 15 RF, and 15 deployment weights')}
$handoff=@(Import-Csv -LiteralPath (SafePath 'catalogs/handoff_files.csv'))
foreach($f in $handoff){CheckFile $f.path $f.sha256}
$clips=@(Import-Csv -LiteralPath (SafePath 'catalogs/field_clips.csv'))
if($clips.Count -ne 60 -or @($clips.clip_id | Select-Object -Unique).Count -ne 60){$failures.Add('Expected 60 unique clip IDs')}
$missing=@($clips | Where-Object gps_status -eq 'missing' | ForEach-Object clip_id | Sort-Object)
if(($missing -join ',') -ne 'AV_F_EW_7,AV_S_EW_7,AV_T_EW_7,AV_V_EW_7'){$failures.Add('Unexpected missing-GPS set')}
foreach($row in $clips){
    CheckFile $row.video_path $row.video_sha256
    if($row.gps_status -eq 'present'){
        CheckFile $row.gps_path $row.gps_sha256
        if([int]$row.gps_columns -ne 19){$failures.Add("Unexpected GPS schema: $($row.clip_id)")}
    }
}
$videos=@(Get-ChildItem -LiteralPath (SafePath 'data/field') -Recurse -File -Filter '*.mp4')
$gps=@(Get-ChildItem -LiteralPath (SafePath 'data/field') -Recurse -File -Filter '*_trajectory.csv')
if($videos.Count -ne 60 -or $gps.Count -ne 56){$failures.Add('File counts differ from 60 videos / 56 GPS CSVs')}
$runtime=@(Import-Csv -LiteralPath (SafePath 'catalogs/runtime_inputs.csv'))
if($runtime.Count -ne 116 -or @($runtime.runtime_path | Select-Object -Unique).Count -ne 116){$failures.Add('Expected 116 unique runtime input copies')}
foreach($f in $runtime){CheckFile $f.source_path $f.sha256;CheckFile $f.runtime_path $f.sha256}
$summary=Get-Content -LiteralPath (SafePath 'catalogs/field_summary.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if($summary.videos -ne $clips.Count -or $summary.gps_csvs -ne $gps.Count){$failures.Add('Catalog summary counts differ')}
$status='PASS';if($failures.Count){$status='FAIL'}
$report=[ordered]@{
    status=$status;verified_at_utc=[DateTime]::UtcNow.ToString('o');workspace_date='2026-11-01';
    checks=[ordered]@{archive_sha256=$archives.Count;extracted_source_sha256=$source.Count;original_migration_checksums=$original.Count;original_classification_checksums=$classOriginal.Count;principal_classifier_weights=($neural.Count+$forests.Count);deployment_weight_copies=$deploy.Count;standalone_handoff_sha256=$handoff.Count;field_videos=$videos.Count;gps_csvs=$gps.Count;runtime_copy_sha256=$runtime.Count;unique_clip_ids=60};
    failures=@($failures);
    data_findings=[ordered]@{missing_gps=$missing;unmatched_gps_frame_rows=$summary.missing_frame_rows;gps_time_backsteps=$summary.gps_time_backsteps;frame_time_outside_container=$summary.clips_with_frame_time_outside_container};
    revision_ledger='catalogs/workspace_changes.json';
    field_experiment='experiments/field_baseline_20261101/RESULTS.txt';
    view_adaptation_experiment='experiments/view_adaptation_20261101/RESULTS.txt';
    view_refinement_experiment='experiments/view_refinement_20261101/RESULTS.txt';
    full_video_survey='experiments/field_full_survey_20261101/RESULTS.txt';
    core_protocol='experiments/field_core_20261101/REPORT.md';
    legacy_field_experiment='experiments/field_pilot_20261101/REPORT.md';
    todd_three_second_experiment='experiments/field_todd_3s_20261102/RESULTS.txt';
    todd_three_second_validation='experiments/field_todd_3s_20261102/validation.json';
    not_run=@('Reliable target/leader trajectories in every recording: full-video inference coverage is distinct from successful trajectory acceptance','Representative cross-scene or independent-vehicle field evaluation; within-view classifier training and video-holdout development tests have been completed','Survey-grade camera calibration and independent 3D annotation evaluation');
    historical_validation='docs/migration/VALIDATION.json and docs/classification_migration/VALIDATION.json are source-machine history, not tests rerun on this machine'
}
[IO.File]::WriteAllText((SafePath 'catalogs/verification.json'),($report | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
$report | ConvertTo-Json -Depth 8
if($failures.Count){exit 1}
