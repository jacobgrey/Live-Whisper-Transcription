param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Inputs)

if (-not $Inputs -or $Inputs.Count -eq 0) {
    Write-Host "Usage: combine-flacs.ps1 <folder> | <file1.flac> <file2.flac> ..." -ForegroundColor Red
    exit 1
}

$ErrorActionPreference = 'Stop'
$ffmpeg = Join-Path $PSScriptRoot '..\tools\ffmpeg\ffmpeg.exe'

if (-not (Test-Path -LiteralPath $ffmpeg)) {
    Write-Host "ffmpeg not found at $ffmpeg" -ForegroundColor Red
    exit 1
}

$flacs = @()
foreach ($p in $Inputs) {
    $p = $p.TrimEnd('\','"')
    if (-not (Test-Path -LiteralPath $p)) {
        Write-Host "Skipping (not found): $p" -ForegroundColor Yellow
        continue
    }
    $item = Get-Item -LiteralPath $p
    if ($item.PSIsContainer) {
        $flacs += Get-ChildItem -LiteralPath $item.FullName -Filter '*.flac' -File
    } elseif ($item.Extension -ieq '.flac') {
        $flacs += $item
    } else {
        Write-Host "Skipping (not .flac): $($item.Name)" -ForegroundColor Yellow
    }
}

$flacs = $flacs | Sort-Object FullName -Unique | Sort-Object Name
if ($flacs.Count -eq 0) {
    Write-Host "No .flac files found." -ForegroundColor Red
    exit 1
}

Write-Host "Found $($flacs.Count) FLAC file(s):"
$flacs | ForEach-Object { Write-Host "  $($_.Name)" }

$ffargs = @()
foreach ($f in $flacs) { $ffargs += @('-i', $f.FullName) }
for ($i = 0; $i -lt $flacs.Count; $i++) { $ffargs += @('-map', "${i}:a") }
$ffargs += @('-ac', '1', '-c:a', 'flac')
for ($i = 0; $i -lt $flacs.Count; $i++) {
    $title = $flacs[$i].BaseName -replace '^\d+-', ''
    $ffargs += @("-metadata:s:a:$i", "title=$title")
}

$outDir = $flacs[0].DirectoryName
$outName = Split-Path -Leaf $outDir
$output = Join-Path $outDir "$outName.mkv"
$ffargs += @('-y', $output)

Write-Host ""
Write-Host "Writing: $output"
Write-Host ""

& $ffmpeg @ffargs
if ($LASTEXITCODE -ne 0) {
    Write-Host "ffmpeg failed (exit $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Done: $output" -ForegroundColor Green
