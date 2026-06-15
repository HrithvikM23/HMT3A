$ErrorActionPreference = 'Stop'

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
$BundleModels = Join-Path ([System.IO.Path]::GetTempPath()) 'kinara_bundle_models'
$ArtifactsRoot = Join-Path $ProjectRoot 'artifacts'
$DistRoot = Join-Path $ArtifactsRoot 'windows'
$BuildRoot = Join-Path $ArtifactsRoot 'pyinstaller\build'
$SpecRoot = Join-Path $ArtifactsRoot 'pyinstaller\spec'
$BuiltLauncher = Join-Path $DistRoot 'Kinara\Kinara.exe'

function Resolve-KinaraPython {
    $PathCandidates = @(
        $env:KINARA_BUILD_PYTHON,
        $env:KINARA_PYTHON,
        (Join-Path $ProjectRoot '.venv\Scripts\python.exe')
    )
    $CommandCandidate = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $CommandCandidate) {
        $PathCandidates += $CommandCandidate.Source
    }

    foreach ($Candidate in $PathCandidates) {
        if ([string]::IsNullOrWhiteSpace($Candidate)) {
            continue
        }
        $Resolved = $Candidate
        if (Test-Path -LiteralPath $Resolved -PathType Container) {
            $Resolved = Join-Path $Resolved 'python.exe'
        }
        if ((Test-Path -LiteralPath $Resolved -PathType Leaf) -and ((Split-Path -Leaf $Resolved).ToLowerInvariant() -eq 'python.exe')) {
            return $Resolved
        }
    }

    throw "Python runtime not found. Set KINARA_BUILD_PYTHON or KINARA_PYTHON to a Python 3.11 python.exe."
}

$Python = Resolve-KinaraPython
$env:PYTHONNOUSERSITE = $null
Write-Host "Using Python: $Python"
$PyInstallerVersion = & $Python -m PyInstaller --version
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed for $Python. Run: `"$Python`" -m pip install pyinstaller"
}
$PySideCheck = & $Python -c "import PySide6; print(PySide6.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "PySide6 is not installed for $Python. Run: `"$Python`" -m pip install PySide6"
}
Write-Host "Using PyInstaller: $PyInstallerVersion"
Write-Host "Using PySide6: $PySideCheck"

Push-Location $ProjectRoot
try {
    if (Test-Path -LiteralPath $BundleModels) {
        Remove-Item -LiteralPath $BundleModels -Recurse -Force
    }
    New-Item -ItemType Directory -Path $DistRoot, $BuildRoot, $SpecRoot -Force | Out-Null

    $PyInstallerArgs = @(
        '--noconfirm',
        '--clean',
        '--windowed',
        '--name', 'Kinara',
        '--distpath', "$DistRoot",
        '--workpath', "$BuildRoot",
        '--specpath', "$SpecRoot",
        '--paths', "$ProjectRoot",
        '--exclude-module', 'cv2',
        '--exclude-module', 'mediapipe',
        '--exclude-module', 'ultralytics',
        '--exclude-module', 'torch',
        '--exclude-module', 'torchvision',
        '--exclude-module', 'torchaudio',
        '--exclude-module', 'onnxruntime',
        '--exclude-module', 'jax',
        '--exclude-module', 'jaxlib',
        '--exclude-module', 'polars',
        '--exclude-module', 'pandas',
        '--exclude-module', 'scipy',
        '--exclude-module', 'matplotlib',
        '--hidden-import', 'plistlib',
        '--hidden-import', 'timeit',
        '--hidden-import', 'zoneinfo'
    )

    $ModelsRoot = Join-Path $ProjectRoot 'models'
    if (Test-Path -LiteralPath $ModelsRoot) {
        $modelFiles = @(Get-ChildItem -LiteralPath $ModelsRoot -Recurse -File | Where-Object {
            $relative = $_.FullName.Substring($ModelsRoot.Length).TrimStart([char[]]@('\', '/'))
            -not (
                $relative -like 'body\pose_landmark_*.tflite' -or
                $relative -like 'hand\mediapipe\*'
            )
        })

        foreach ($file in $modelFiles) {
            $relative = $file.FullName.Substring($ModelsRoot.Length).TrimStart([char[]]@('\', '/'))
            $destination = Join-Path $BundleModels $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
            Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
        }

        if ($modelFiles.Count -gt 0) {
            $PyInstallerArgs += @('--add-data', "$BundleModels;models")
        }
    }
    $UltralyticsConfig = Join-Path $ProjectRoot '.ultralytics'
    if (Test-Path -LiteralPath $UltralyticsConfig) {
        $PyInstallerArgs += @('--add-data', "$UltralyticsConfig;.ultralytics")
    }
    $AssetsRoot = Join-Path $ProjectRoot 'assets'
    if (Test-Path -LiteralPath $AssetsRoot) {
        $PyInstallerArgs += @('--add-data', "$AssetsRoot;assets")
    }
    $ProjectMetadataFiles = @('pyproject.toml')
    foreach ($relativeFile in $ProjectMetadataFiles) {
        $sourceFile = Join-Path $ProjectRoot $relativeFile
        if (Test-Path -LiteralPath $sourceFile -PathType Leaf) {
            $PyInstallerArgs += @('--add-data', "$sourceFile;.")
        }
    }
    $SourceDataDirs = @(
        'app',
        'camera',
        'core',
        'inference',
        'kinara',
        'network',
        'pipeline',
        'runners',
        'utils'
    )
    foreach ($relativeDir in $SourceDataDirs) {
        $sourceDir = Join-Path $ProjectRoot $relativeDir
        if (Test-Path -LiteralPath $sourceDir -PathType Container) {
            $PyInstallerArgs += @('--add-data', "$sourceDir;$relativeDir")
        }
    }
    $IconPath = Join-Path $ProjectRoot 'assets\kinara.ico'
    if (Test-Path -LiteralPath $IconPath) {
        $PyInstallerArgs += @('--icon', "$IconPath")
    }

    $PyInstallerArgs += (Join-Path $ProjectRoot 'app\kinara_launcher.py')

    & $Python -m PyInstaller @PyInstallerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
    if (Test-Path -LiteralPath $BundleModels) {
        Remove-Item -LiteralPath $BundleModels -Recurse -Force
    }
}

Write-Host "Built launcher: $BuiltLauncher"
