param(
    [string]$PythonPath = "python",
    # 可选：把 llama-server 及其同目录依赖一起打进安装包
    [string]$LlamaDir = "",
    # 可选：把 .gguf 模型文件打进 models\ 子目录（llama_server 后端）
    [string]$ModelDir = "",
    # 可选：把 merged 后的 HF 模型目录打进 models\（transformers 后端）
    [string]$EmbeddedModelDir = "",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# 训练栈默认排除：llama_server 后端只通过 HTTP 调子进程，不需要 torch。
# 但 transformers 后端是在进程内跑模型的，必须把 torch / transformers 打进去，
# 否则应用一启动就 ImportError。
$excludedModules = @(
    "bitsandbytes",
    "accelerate",
    "peft",
    "trl",
    "datasets",
    "tensorflow",
    "jax",
    "IPython",
    "jedi",
    "tkinter",
    "matplotlib.tests",
    "numpy.tests",
    "pandas.tests"
)
if (-not $EmbeddedModelDir) {
    # 只有非嵌入式构建才排除 ML 运行时。
    $excludedModules = @("torch", "transformers", "sentencepiece") + $excludedModules
}

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
    if ($EmbeddedModelDir) {
        # 进程内后端要真正 import torch / transformers。transformers 会按模型
        # 类型动态导入子模块，静态分析抓不到，必须显式收集。
        $pyinstallerArgs += @(
            "--collect-all", "transformers",
            "--collect-all", "tokenizers",
            "--collect-all", "safetensors",
            "--hidden-import", "local_model_embedded",
            "--hidden-import", "finetune.local_llm"
        )
    }
    foreach ($module in $excludedModules) {
        $pyinstallerArgs += @("--exclude-module", $module)
    }

    # 让 PyInstaller 优先从构建环境解析 DLL。
    #
    # conda 会把 **base** 环境的 Library\bin 放进 PATH。PyInstaller 按 PATH 搜索
    # DLL 依赖，于是 libexpat / libssl / libcrypto / liblzma / sqlite3 会被从 base
    # 环境收集。两个版本导出表不同，而 pyexpat.pyd 是按 ordinal 引用导出的，
    # 结果打包后一启动就报：
    #   ImportError: DLL load failed while importing pyexpat: 操作系统无法运行 %1
    # （%1 = Windows 182 ERROR_INVALID_ORDINAL，即按序号导入的函数不存在）
    $previousPath = $env:PATH
    $env:PATH = (@(
        $pythonPrefix,
        (Join-Path $pythonPrefix "Library\bin"),
        (Join-Path $pythonPrefix "DLLs")
    ) -join ";") + ";" + $previousPath
    $pyinstallerExit = Invoke-BuildCommand -FilePath $PythonPath -Arguments (@("-m", "PyInstaller") + $pyinstallerArgs)
    $env:PATH = $previousPath
    if ($pyinstallerExit -ne 0) {
        throw "PyInstaller failed with exit code $pyinstallerExit."
    }

    $modelConfigPath = Join-Path $repoRoot "dist\ExcelAssistant\llm_config.json"
    $configExit = Invoke-BuildCommand -FilePath $PythonPath -Arguments @($modelConfigWriter, $modelConfigPath)
    if ($configExit -ne 0) {
        throw "Could not create the release model configuration."
    }

    # 把 llama-server 及其运行时依赖复制进安装目录，供本地模型模式自动启动。
    if ($LlamaDir) {
        $resolvedLlama = (Resolve-Path -LiteralPath $LlamaDir -ErrorAction Stop).Path
        $llamaTarget = Join-Path $repoRoot "dist\ExcelAssistant\llama"
        New-Item -ItemType Directory -Force $llamaTarget | Out-Null
        Copy-Item -Path (Join-Path $resolvedLlama "*") -Destination $llamaTarget -Recurse -Force
        $serverExe = Join-Path $llamaTarget "llama-server.exe"
        if (-not (Test-Path -LiteralPath $serverExe)) {
            throw "llama-server.exe was not found in $resolvedLlama. Point -LlamaDir at the folder that contains it."
        }
        Write-Host "Bundled llama-server from $resolvedLlama"
    }

    # 把模型文件复制进 models\ ，让安装包开箱即用（无需联网下载）。
    if ($ModelDir) {
        $resolvedModel = (Resolve-Path -LiteralPath $ModelDir -ErrorAction Stop).Path
        $modelsTarget = Join-Path $repoRoot "dist\ExcelAssistant\models"
        New-Item -ItemType Directory -Force $modelsTarget | Out-Null
        if (Test-Path -LiteralPath $resolvedModel -PathType Container) {
            Copy-Item -Path (Join-Path $resolvedModel "*.gguf") -Destination $modelsTarget -Force
        } else {
            Copy-Item -Path $resolvedModel -Destination $modelsTarget -Force
        }
        $copied = @(Get-ChildItem -Path $modelsTarget -Filter "*.gguf" -File)
        if ($copied.Count -eq 0) {
            throw "No .gguf file was found in $resolvedModel."
        }
        foreach ($file in $copied) {
            Write-Host ("Bundled model {0} ({1:N0} MB)" -f $file.Name, ($file.Length / 1MB))
        }
    }

    # 复制 merged 后的 HF 模型目录（transformers 后端用）。
    if ($EmbeddedModelDir) {
        $resolvedEmbedded = (Resolve-Path -LiteralPath $EmbeddedModelDir -ErrorAction Stop).Path
        if (-not (Test-Path -LiteralPath $resolvedEmbedded -PathType Container)) {
            throw "-EmbeddedModelDir must be a directory: $resolvedEmbedded"
        }
        if (-not (Test-Path -LiteralPath (Join-Path $resolvedEmbedded "config.json"))) {
            throw "Model directory has no config.json: $resolvedEmbedded"
        }
        $embeddedTarget = Join-Path $repoRoot "dist\ExcelAssistant\models\embedded"
        New-Item -ItemType Directory -Force $embeddedTarget | Out-Null
        Copy-Item -Path (Join-Path $resolvedEmbedded "*") -Destination $embeddedTarget -Recurse -Force
        $size = (Get-ChildItem $embeddedTarget -Recurse -File | Measure-Object Length -Sum).Sum
        Write-Host ("Bundled embedded model {0} ({1:N0} MB)" -f (Split-Path $resolvedEmbedded -Leaf), ($size / 1MB))
    }

    # 校验：配置里声明的模型/服务路径必须真的在发行目录里。
    # 这一步是补上一个真实事故：config 写入了 local_model_path，但文件没被打进
    # dist，构建仍然“成功”，安装后一启动就是 error 状态（既启动不了也下载不了）。
    $distDir = Join-Path $repoRoot "dist\ExcelAssistant"
    $releaseConfig = Get-Content -Raw -LiteralPath $modelConfigPath | ConvertFrom-Json
    $requiredPaths = @()
    if ($releaseConfig.PSObject.Properties.Name -contains "local_server" -and $releaseConfig.local_server) {
        $requiredPaths += @{ Key = "local_server"; Path = $releaseConfig.local_server }
    }
    if ($releaseConfig.PSObject.Properties.Name -contains "local_model_path" -and $releaseConfig.local_model_path) {
        # 配了下载地址时允许首启下载；否则文件必须随包。
        $hasDownload = $releaseConfig.PSObject.Properties.Name -contains "model_url" -and $releaseConfig.model_url
        if (-not $hasDownload) {
            $requiredPaths += @{ Key = "local_model_path"; Path = $releaseConfig.local_model_path }
        }
    }
    foreach ($item in $requiredPaths) {
        $candidate = Join-Path $distDir ($item.Path -replace '/', '\')
        if (-not (Test-Path -LiteralPath $candidate)) {
            throw (
                "Release config points at a missing file: {0} = '{1}'. " +
                "Expected it at '{2}'. Pass -ModelDir or -LlamaDir so it gets bundled."
            ) -f $item.Key, $item.Path, $candidate
        }
    }
    if ($requiredPaths.Count -gt 0) {
        Write-Host "Release payload verified: all declared paths exist in dist."
    }

    # 修复从错误环境收集的 DLL，并对对 DLL 敏感的标准库扩展做加载冒烟测试。
    # 这一步能拦住「DLL load failed while importing pyexpat」这类只在目标机器
    # 才暴露的问题。
    $verifyScript = Join-Path $PSScriptRoot "verify_dist.py"
    $verifyExit = Invoke-BuildCommand -FilePath $PythonPath -Arguments @(
        $verifyScript,
        $distDir
    )
    if ($verifyExit -ne 0) {
        throw "Distribution verification failed with exit code $verifyExit."
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
