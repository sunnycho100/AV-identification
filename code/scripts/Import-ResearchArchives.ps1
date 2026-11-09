param([string]$Workspace = (Split-Path $PSScriptRoot -Parent))
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Workspace = [IO.Path]::GetFullPath($Workspace)
$prefix = ([IO.Path]::GetFullPath((Join-Path $Workspace '..'))).TrimEnd('\') + '\'
function WorkspacePath([string]$relative) {
    $p = [IO.Path]::GetFullPath((Join-Path $Workspace $relative))
    if (-not $p.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Path outside workspace: $relative" }
    return $p
}
$utf8 = [Text.UTF8Encoding]::new($false)
foreach ($d in @('../archive/source_packages','catalogs','docs/migration','../paper','trajectory','data/field','classification','configs')) {
    [IO.Directory]::CreateDirectory((WorkspacePath $d)) | Out-Null
}
$files = [Collections.Generic.List[object]]::new()
$archives = [Collections.Generic.List[object]]::new()
$specs = @(
    @{Name='av_roadside_migration_20261101.zip'; Prefix='av_roadside_migration_20261101/'; Kind='migration'},
    @{Name='processed data.zip'; Prefix='processed data/'; Kind='field'},
    @{Name='av_identification_module_20261101.zip'; Prefix='av_identification_module_20261101/'; Kind='classification'}
)
foreach ($spec in $specs) {
    $source = WorkspacePath ('../archive/source_packages/'+$spec.Name)
    if (-not (Test-Path -LiteralPath $source)) { $source = WorkspacePath $spec.Name }
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing input archive: $($spec.Name)" }
    $archiveHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    $zip = [IO.Compression.ZipFile]::OpenRead($source)
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $count = 0; $total = [long]0
    try {
        foreach ($entry in $zip.Entries) {
            $name = $entry.FullName.Replace('\','/')
            if (-not $name.StartsWith($spec.Prefix, [StringComparison]::Ordinal)) { throw "Unexpected archive prefix: $name" }
            $rel = $name.Substring($spec.Prefix.Length)
            if (-not $rel -or $name.EndsWith('/')) { continue }
            if ($rel -match '(^|/)\.\.(/|$)|:|^/') { throw "Unsafe ZIP path: $name" }
            if ($spec.Kind -eq 'field') { $mapped = 'data/field/'+$rel }
            elseif ($spec.Kind -eq 'classification') {
                if ($rel.StartsWith('trajectory-avlike-transformer/',[StringComparison]::Ordinal)) { $mapped = 'classification/'+$rel }
                else { $mapped = 'docs/classification_migration/'+$rel }
            }
            elseif ($rel.StartsWith('paper/')) { $mapped = '../'+$rel }
            elseif ($rel.StartsWith('trajectory/')) { $mapped = $rel }
            else { $mapped = 'docs/migration/'+$rel }
            $target = WorkspacePath $mapped
            if (-not $seen.Add($target)) { throw "Duplicate ZIP target: $mapped" }
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null
            $stream = $entry.Open()
            $sha = [Security.Cryptography.SHA256]::Create()
            try { $expected = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
            finally { $sha.Dispose(); $stream.Dispose() }
            if (Test-Path -LiteralPath $target) {
                $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($actual -ne $expected) { throw "Existing file differs; not overwritten: $mapped" }
            } else {
                $stream = $entry.Open()
                $output = [IO.File]::Open($target, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
                try { $stream.CopyTo($output) } finally { $output.Dispose(); $stream.Dispose() }
                $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($actual -ne $expected) { throw "Extraction checksum mismatch: $mapped" }
            }
            $files.Add([pscustomobject][ordered]@{archive=$spec.Name; archive_entry=$name; path=$mapped; bytes=$entry.Length; sha256=$expected})
            $count++; $total += $entry.Length
        }
    } finally { $zip.Dispose() }
    $destination = WorkspacePath ('../archive/source_packages/'+$spec.Name)
    if ($source -ne $destination) {
        if (Test-Path -LiteralPath $destination) { throw "Archive destination already exists: $destination" }
        Move-Item -LiteralPath $source -Destination $destination
    }
    $archives.Add([pscustomobject][ordered]@{path='../archive/source_packages/'+$spec.Name; sha256=$archiveHash; bytes=(Get-Item -LiteralPath $destination).Length; extracted_files=$count; extracted_bytes=$total})
    Write-Output "$($spec.Name): $count files verified, $total bytes extracted"
}
$files | Export-Csv -LiteralPath (WorkspacePath 'catalogs/archive_files.csv') -NoTypeInformation -Encoding UTF8
[IO.File]::WriteAllText((WorkspacePath 'catalogs/archives.json'), ($archives | ConvertTo-Json -Depth 6), $utf8)
Write-Output 'Source archive contents preserved byte-for-byte; existing files are never overwritten.'