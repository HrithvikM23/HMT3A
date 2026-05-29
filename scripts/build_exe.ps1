$ErrorActionPreference = 'Stop'

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
$Python = 'C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe'
$PyInstaller = Join-Path $env:APPDATA 'Python\Python311\Scripts\pyinstaller.exe'
$BundleModels = Join-Path ([System.IO.Path]::GetTempPath()) 'kinara_bundle_models'
$ArtifactsRoot = Join-Path $ProjectRoot 'artifacts'
$DistRoot = Join-Path $ArtifactsRoot 'windows'
$BuildRoot = Join-Path $ArtifactsRoot 'pyinstaller\build'
$SpecRoot = Join-Path $ArtifactsRoot 'pyinstaller\spec'
$BuiltLauncher = Join-Path $DistRoot 'Kinara\Kinara.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python runtime not found: $Python"
}
if (-not (Test-Path -LiteralPath $PyInstaller)) {
    throw "PyInstaller not found: $PyInstaller"
}

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
        '--exclude-module', 'mediapipe',
        '--collect-submodules', 'ultralytics',
        '--collect-submodules', 'cv2',
        '--collect-submodules', 'onnxruntime'
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
    $IconPath = Join-Path $ProjectRoot 'assets\kinara.ico'
    if (Test-Path -LiteralPath $IconPath) {
        $PyInstallerArgs += @('--icon', "$IconPath")
    }

    $PyInstallerArgs += (Join-Path $ProjectRoot 'app\kinara_launcher.py')

    & $PyInstaller @PyInstallerArgs
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
