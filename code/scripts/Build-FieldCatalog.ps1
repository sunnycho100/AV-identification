param([string]$Workspace = (Split-Path $PSScriptRoot -Parent))
$ErrorActionPreference = 'Stop'
$Workspace = [IO.Path]::GetFullPath($Workspace)
$utf8 = [Text.UTF8Encoding]::new($false)
$culture = [Globalization.CultureInfo]::InvariantCulture
function Relative([string]$p) { $p.Substring($Workspace.Length+1).Replace('\','/') }
function Number([string]$s) { [double]::Parse($s,$culture) }
function U32([IO.BinaryReader]$r) {
    $b=$r.ReadBytes(4); if($b.Length -ne 4){throw 'Truncated MP4'}
    [Array]::Reverse($b); [BitConverter]::ToUInt32($b,0)
}
function U64([IO.BinaryReader]$r) {
    $b=$r.ReadBytes(8); if($b.Length -ne 8){throw 'Truncated MP4'}
    [Array]::Reverse($b); [BitConverter]::ToUInt64($b,0)
}
function MovieDuration([string]$p) {
    $s=[IO.File]::OpenRead($p); $r=[IO.BinaryReader]::new($s)
    try {
        $limit=$s.Length
        while($s.Position+8 -le $limit) {
            $start=$s.Position; $size=[long](U32 $r)
            $type=[Text.Encoding]::ASCII.GetString($r.ReadBytes(4))
            if($size -eq 1){$size=[long](U64 $r)}
            if($size -eq 0){$size=$limit-$start}
            $end=$start+$size
            if($end -gt $limit -or $end -lt $s.Position){throw "Invalid MP4 atom in $p"}
            if($type -eq 'moov'){$limit=$end; continue}
            if($type -eq 'mvhd') {
                $version=$r.ReadByte(); $null=$r.ReadBytes(3)
                if($version -eq 0){$null=$r.ReadBytes(8);$scale=U32 $r;$duration=U32 $r}
                elseif($version -eq 1){$null=$r.ReadBytes(16);$scale=U32 $r;$duration=U64 $r}
                else {throw "Unsupported mvhd version $version"}
                if($scale -eq 0){throw 'Zero MP4 timescale'}
                return [Math]::Round(([double]$duration/$scale),6)
            }
            $s.Position=$end
        }
        throw "No movie duration in $p"
    } finally {$r.Dispose();$s.Dispose()}
}
$expectedColumns='epoch_time_sec,time_sec,latitude,longitude,height,north_velocity,east_velocity,up_velocity,processed_time,processed_x,processed_y,processed_z,processed_vx,processed_vy,processed_vz,video_time_sec,nearest_frame_index,nearest_frame_time_sec,nearest_frame_time_error_sec'.Split(',')
$priority=@('AV_W_EW_3','AV_W_WE_3','AV_V_WE_3','AV_T_WE_3','HV_V_EW_3','HV_S_EW_3','HV_T_EW_3')
$historic=@{HV_T_EW_2='Historical code excludes GPS evaluation: instrumented vehicle identity unconfirmed';AV_T_WE_3='Historical code excludes GPS evaluation: earlier local cut differed from GPS-aligned video'}
$zone=[TimeZoneInfo]::FindSystemTimeZoneById('Central Standard Time')
$sourceFiles=Import-Csv -LiteralPath (Join-Path $Workspace 'catalogs/archive_files.csv')
$hashes=@{};foreach($f in $sourceFiles){$hashes[$f.path]=$f.sha256}
$clips=[Collections.Generic.List[object]]::new()
foreach($video in Get-ChildItem -LiteralPath (Join-Path $Workspace 'data/field') -Recurse -File -Filter '*.mp4' | Sort-Object FullName) {
    $id=$video.BaseName; $parts=$id.Split('_')
    if($parts.Count -ne 4 -or $parts[0] -notin @('AV','HV')){throw "Unexpected sample ID: $id"}
    $site=$video.Directory.Parent.Name; $view=$video.Directory.Name
    $role='supplementary'
    if(($site -eq 'V_Verona_Rd' -and $view -eq 'View_SE') -or ($site -eq 'T_Todd_Dr' -and $view -eq 'View_SE') -or ($site -eq 'S_Seminole_Hwy' -and $view -eq 'View_NW')){$role='core'}
    elseif($site -eq 'W_Whitney_Way'){$role='backup'}
    $duration=MovieDuration $video.FullName
    $csv=Join-Path $video.DirectoryName ($id+'_trajectory.csv')
    $row=[ordered]@{
        clip_id=$id;site=$site;view=$view;mode_from_source_filename=$parts[0];state_label_verified=$false;
        travel_direction=$parts[2];run_number=$parts[3];suggested_role=$role;
        video_path=(Relative $video.FullName);video_bytes=$video.Length;video_sha256=$hashes[(Relative $video.FullName)];
        container_duration_s=$duration;gps_status='missing';gps_path='';gps_sha256='';gps_rows=0;gps_columns=0;
        recording_date_chicago_from_gps='';gps_epoch_start='';gps_epoch_end='';gps_video_time_start_s='';gps_video_time_end_s='';
        gps_unique_frame_indices=0;gps_missing_frame_rows=0;gps_time_backsteps=0;gps_frame_time_outside_container_rows=0;gps_max_abs_frame_error_s='';
        quality_note='Not individually specified in supplied review (do not assume clean)';historical_evaluation_note='';
        runtime_video_path=('trajectory/Camera data/'+$id+'.mp4');runtime_gps_path=''
    }
    if($id -in $priority){$row.quality_note='Priority review in supplied five-frame screening; not re-reviewed here'}
    if($historic.ContainsKey($id)){$row.historical_evaluation_note=$historic[$id]}
    if(Test-Path -LiteralPath $csv) {
        $data=@(Import-Csv -LiteralPath $csv)
        if($data.Count -eq 0){throw "Empty CSV: $csv"}
        $columns=@($data[0].PSObject.Properties.Name)
        if(($columns -join ',') -ne ($expectedColumns -join ',')){throw "CSV schema mismatch: $csv"}
        $row.gps_status='present';$row.gps_path=Relative $csv;$row.gps_sha256=$hashes[$row.gps_path]
        $row.gps_rows=$data.Count;$row.gps_columns=$columns.Count
        $row.runtime_gps_path='trajectory/Camera data/'+$id+'_trajectory.csv'
        $first=Number $data[0].epoch_time_sec;$last=Number $data[-1].epoch_time_sec
        $date=[DateTimeOffset]::FromUnixTimeMilliseconds([long]($first*1000))
        $row.recording_date_chicago_from_gps=[TimeZoneInfo]::ConvertTime($date,$zone).ToString('yyyy-MM-dd')
        $row.gps_epoch_start=$first;$row.gps_epoch_end=$last
        $row.gps_video_time_start_s=Number $data[0].video_time_sec;$row.gps_video_time_end_s=Number $data[-1].video_time_sec
        $frames=[Collections.Generic.HashSet[int]]::new();$prev=[double]::NegativeInfinity;$maxError=0.0
        foreach($sample in $data) {
            $epoch=Number $sample.epoch_time_sec
            if($epoch -lt $prev){$row.gps_time_backsteps++};$prev=$epoch
            if($sample.nearest_frame_index -ne '') {
                $null=$frames.Add([int]$sample.nearest_frame_index)
                $t=Number $sample.nearest_frame_time_sec
                if($t -lt -0.1 -or $t -gt $duration+0.1){$row.gps_frame_time_outside_container_rows++}
            }else{$row.gps_missing_frame_rows++}
            if($sample.nearest_frame_time_error_sec -ne ''){$maxError=[Math]::Max($maxError,[Math]::Abs((Number $sample.nearest_frame_time_error_sec)))}
        }
        $row.gps_unique_frame_indices=$frames.Count;$row.gps_max_abs_frame_error_s=[Math]::Round($maxError,9)
    }
    $clips.Add([pscustomobject]$row)
}
if(@($clips.clip_id | Select-Object -Unique).Count -ne $clips.Count){throw 'Duplicate clip IDs; flat runtime layout would collide'}
$orphanCsv=@(Get-ChildItem -LiteralPath (Join-Path $Workspace 'data/field') -Recurse -File -Filter '*_trajectory.csv' | Where-Object {($_.BaseName -replace '_trajectory$','') -notin $clips.clip_id})
if($orphanCsv.Count){throw 'Unpaired trajectory CSVs found'}
$clips | Export-Csv -LiteralPath (Join-Path $Workspace 'catalogs/field_clips.csv') -NoTypeInformation -Encoding UTF8
$summary=[ordered]@{
    generated_date='2026-11-01'; timezone='America/Chicago';
    scope='Local file, CSV schema, and MP4 movie-header inspection; no video decoding or synchronization validation';
    videos=$clips.Count;gps_csvs=@($clips | Where-Object gps_status -eq 'present').Count;
    av_filename_labels=@($clips | Where-Object mode_from_source_filename -eq 'AV').Count;
    hv_filename_labels=@($clips | Where-Object mode_from_source_filename -eq 'HV').Count;
    container_duration_seconds=[Math]::Round(($clips | Measure-Object container_duration_s -Sum).Sum,3);
    gps_rows=($clips | Measure-Object gps_rows -Sum).Sum;
    missing_gps=@($clips | Where-Object gps_status -eq 'missing' | ForEach-Object clip_id);
    groups=@($clips | Group-Object site,view | ForEach-Object {
        $g=$_.Group;[ordered]@{site=$g[0].site;view=$g[0].view;role=$g[0].suggested_role;videos=$g.Count;gps_csvs=@($g | Where-Object gps_status -eq 'present').Count;av=@($g | Where-Object mode_from_source_filename -eq 'AV').Count;hv=@($g | Where-Object mode_from_source_filename -eq 'HV').Count}
    });
    dates_from_available_gps=@($clips | Where-Object gps_status -eq 'present' | Group-Object recording_date_chicago_from_gps | ForEach-Object {[ordered]@{date=$_.Name;gps_csvs=$_.Count}});
    missing_frame_rows=($clips | Measure-Object gps_missing_frame_rows -Sum).Sum;
    gps_time_backsteps=($clips | Measure-Object gps_time_backsteps -Sum).Sum;
    clips_with_frame_time_outside_container=@($clips | Where-Object gps_frame_time_outside_container_rows -gt 0 | Select-Object clip_id,gps_frame_time_outside_container_rows);
    label_limit='AV/HV retained as source sample labels, not independently verified per-frame control states; do not label every visible vehicle';
    quality_limit='Seven priority IDs known; other fourteen caution IDs were not enumerated in supplied README';
    original_readme_path='data/field/README.md'
}
[IO.File]::WriteAllText((Join-Path $Workspace 'catalogs/field_summary.json'),($summary | ConvertTo-Json -Depth 8),$utf8)
$summary | ConvertTo-Json -Depth 8