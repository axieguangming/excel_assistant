param(
    [string]$PythonPath = "python",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Invoke-BuildCommand {
    # 外部工具（PyInstaller、ISCC）会把进度和警告写到 stderr。在
    # $ErrorActionPreference = "Stop" 下，PowerShell 会把这类输出当成终止错误，
    # 导致明明成功的构建被判为失败。这里临时放宽并只看退出码。
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @()
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $FilePath @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
}

$iscc = $null
if (-not $SkipInstaller) {
    $iscc = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if (-not $iscc) {
        $isccCandidates = @(
            (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
            (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
            (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
        )
        $isccPath = $isccCandidates |
            Where-Object { Test-Path -LiteralPath $_ } |
            Select-Object -First 1
        if ($isccPath) {
            $iscc = @{ Source = $isccPath }
        } else {
            throw "Inno Setup 6 was not found. Install it before building the Windows installer."
        }
    }
}

Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force (Join-Path $repoRoot "build\pyinstaller-specs") | Out-Null
    $pythonInfo = & $PythonPath -c "import platform, sys; print(f'{sys.version_info.major}.{sys.version_info.minor}|{platform.architecture()[0]}|{sys.prefix}')"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not determine the active Python environment."
    }
    $pythonVersion, $pythonArchitecture, $pythonPrefix = $pythonInfo -split "\|", 3
    if ($pythonVersion -ne "3.12" -or $pythonArchitecture -ne "64bit") {
        throw "The Windows release must be built with 64-bit Python 3.12."
    }
    $staticDir = Join-Path $repoRoot "static"
    $launcherPath = Join-Path $repoRoot "desktop.py"
    $modelConfigWriter = Join-Path $repoRoot "packaging\write_model_config.py"
    $pyinstallerArgs = @(
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onedir",
        "--name", "ExcelAssistant",
        "--specpath", "build\pyinstaller-specs",
        "--workpath", "build\pyinstaller",
        "--distpath", "dist",
        "--add-data", "$staticDir;static",
        "--collect-all", "webview",
        "--collect-all", "langchain_core",
        "--collect-all", "langchain_openai",
        "--collect-all", "langgraph",
        "--hidden-import", "langchain.chat_models",
        "--hidden-import", "openpyxl",
        "--hidden-import", "xlrd",
        $launcherPath
    )
    $ffiDll = Join-Path $pythonPrefix "Library\bin\ffi.dll"
    if (Test-Path $ffiDll) {
        $pyinstallerArgs += @("--add-binary", "$ffiDll;.")
    }
    $pyinstallerExit = Invoke-BuildCommand -FilePath $PythonPath -Arguments (@("-m", "PyInstaller") + $pyinstallerArgs)
    if ($pyinstallerExit -ne 0) {
        throw "PyInstaller failed with exit code $pyinstallerExit."
    }

    $modelConfigPath = Join-Path $repoRoot "dist\ExcelAssistant\llm_config.json"
    $configExit = Invoke-BuildCommand -FilePath $PythonPath -Arguments @($modelConfigWriter, $modelConfigPath)
    if ($configExit -ne 0) {
        throw "Could not create the release model configuration."
    }

    if (-not $SkipInstaller) {
        $isccExit = Invoke-BuildCommand -FilePath $iscc.Source -Arguments @((Join-Path $PSScriptRoot "excel_assistant.iss"))
        if ($isccExit -ne 0) {
            throw "Inno Setup failed with exit code $isccExit."
        }
    }
} finally {
    Pop-Location
}
