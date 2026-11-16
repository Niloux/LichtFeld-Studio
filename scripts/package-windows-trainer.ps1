[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$BuildDir,
    [Parameter(Mandatory)] [string]$Config,
    [Parameter(Mandatory)] [string]$Output,
    [Parameter(Mandatory)] [ValidatePattern('^[0-9a-f]{7,40}$')] [string]$GeosunPreRevision
)

$ErrorActionPreference = 'Stop'

function Get-Sha256([string]$Path) {
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $stream = [IO.File]::OpenRead($Path)
        try {
            return [BitConverter]::ToString($sha256.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
        } finally {
            $stream.Dispose()
        }
    } finally {
        $sha256.Dispose()
    }
}

$build = (Resolve-Path -LiteralPath $BuildDir).ProviderPath
$configPath = (Resolve-Path -LiteralPath $Config).ProviderPath
$destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
$destination = $destination.TrimEnd([char[]]'\/')
$stage = "$destination.pending"
$archive = "$destination.zip"
$stagedArchive = "$archive.pending"
$checksum = "$archive.sha256"
$stagedChecksum = "$checksum.pending"
foreach ($path in @($destination, $stage, $archive, $stagedArchive, $checksum, $stagedChecksum)) {
    if (Test-Path -LiteralPath $path) { throw "Choose a new output path. Already exists: $path" }
}

$cache = Get-Content -Raw -LiteralPath (Join-Path $build 'CMakeCache.txt')
if ($cache -notmatch '(?m)^LFS_TRAINER_ONLY:BOOL=ON\r?$' -or
    $cache -notmatch '(?m)^CMAKE_BUILD_TYPE:STRING=Release\r?$' -or
    $cache -notmatch '(?m)^CMAKE_CUDA_ARCHITECTURES:STRING=(80-virtual(;[0-9]+-real)*)\r?$') {
    throw 'Build the windows-trainer preset with BUILD_CUDA_PTX_ONLY=ON and BUILD_CUDA_MIN_SM=80 first.'
}
$null = $cache -match '(?m)^CMAKE_CUDA_ARCHITECTURES:STRING=([^\r\n]+)'
$architectures = $Matches[1]
$header = Get-Content -Raw -LiteralPath (Join-Path $build 'include/git_version.h')
if ($header -notmatch 'GIT_COMMIT_HASH_SHORT "([0-9a-f]+)"') { throw 'Missing trainer build revision.' }
$revision = $Matches[1]
$dirty = (& git -C $PSScriptRoot status --porcelain --untracked-files=no) -ne $null
if ($LASTEXITCODE -ne 0) { throw 'Cannot read git status.' }
# A delivery must be reproducible from a commit: clean tree, built from HEAD.
if ($dirty) { throw 'Working tree has uncommitted changes. Commit, rebuild, then package.' }
$head = & git -C $PSScriptRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or -not $head.StartsWith($revision)) {
    throw "Trainer was built from $revision, not HEAD $head. Rebuild, then package."
}
if ($header -notmatch 'GIT_TAGGED_VERSION "([^"]+)"') { throw 'Missing trainer build version.' }
$version = $Matches[1]

New-Item -ItemType Directory -Path $stage | Out-Null
& cmake --install $build --config Release --component trainer-runtime --prefix $stage
if ($LASTEXITCODE -ne 0) { throw "Trainer install failed. Staged files are in $stage" }
if (-not (Test-Path -LiteralPath (Join-Path $stage 'bin/geosun_3dgs_train.exe') -PathType Leaf)) {
    throw 'Trainer install did not produce bin/geosun_3dgs_train.exe.'
}
New-Item -ItemType Directory -Path (Join-Path $stage 'config') | Out-Null
Copy-Item -LiteralPath $configPath -Destination (Join-Path $stage 'config/trainer.json')
@'
# Geosun 3DGS Trainer

Run this command from the extracted directory. Pass the GeosunPre delivery
directory as input and choose a separate output directory.

```powershell
.\bin\geosun_3dgs_train.exe --config .\config\trainer.json -d "path\to\GeosunPre-output" -o "path\to\training-output"
```

Wait for the process to exit. Exit code 0 indicates success. Keep the package
directory layout so the trainer can locate its DLLs and image codecs.
Use an ASCII-only path for the extracted package to enable GPU JPEG decoding.

This package targets Windows x64 with an NVIDIA GPU of compute capability 8.0
or newer, at least 12 GB VRAM (16 GB+ recommended), and a driver that supports
the CUDA Toolkit used to build the package (CUDA 12.8 / NVIDIA R570 or later).
Precompiled SASS is included for SM 8.6, 8.9, and 12.0; other SM 8.0+ GPUs
JIT-compile the included PTX on first use.
See package.json for the paired GeosunPre revision and checksums.

## License and Corresponding Source

This trainer binary is built from modified LichtFeld Studio sources and is
licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later)
together with the LichtFeld Studio NVIDIA GPU Library Exception v1.0.
Full license texts and third-party notices are provided in LICENSE,
LICENSE-EXCEPTION.txt, THIRD_PARTY_LICENSES.md, and
share/LichtFeld-Studio/licenses/.

When packaged alongside an independent preprocessing tool as an Aggregate
under GPLv3 Section 5, this trainer runs as a separate operating-system process
communicating strictly via standard dataset and output directories on disk.

In accordance with GPLv3 Section 6, any recipient of this binary package may
obtain the complete machine-readable Corresponding Source (including build
scripts and modifications matching the commit hash recorded in `revision` in
package.json) free of charge from the distributor's published source channel
or upon written request to the distributor.
'@ | Set-Content -LiteralPath (Join-Path $stage 'README.md') -Encoding Ascii

$files = [ordered]@{}
Get-ChildItem -LiteralPath $stage -Recurse -File | Sort-Object FullName | ForEach-Object {
    $relative = $_.FullName.Substring($stage.Length + 1).Replace('\', '/')
    $files[$relative] = Get-Sha256 $_.FullName
}
$manifest = [ordered]@{
    name = 'Geosun 3DGS Trainer'
    version = $version
    revision = $revision
    geosunpre_revision = $GeosunPreRevision
    cuda_architectures = $architectures
    files = $files
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $stage 'package.json') -Encoding UTF8

Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($stage, $stagedArchive)
$digest = Get-Sha256 $stagedArchive
"$digest  $(Split-Path $archive -Leaf)" | Set-Content -LiteralPath $stagedChecksum -Encoding Ascii
[IO.Directory]::Move($stage, $destination)
[IO.File]::Move($stagedArchive, $archive)
[IO.File]::Move($stagedChecksum, $checksum)
Write-Host "Package: $archive"
