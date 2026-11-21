# ============================================================
# CNN2 - release.ps1
# 阶段4.5：一键发布（可复现构建 + 打包）
#
# 做四件事：
#   1) 从 version.h 读取程序版本 CNN2_VERSION；
#   2) 用 MSBuild 编译 Release x64（带超时保护，避免工具链挂起）；
#   3) 打包到 dist\CNN2-<版本>-win-x64\：
#        CNN2.exe + 4 个运行时 dll（cudart64_12.dll / msvcp140.dll /
#        vcruntime140.dll / vcruntime140_1.dll，优先从 deploy\ 取）
#        + README.md + CHANGELOG.md + BUILDINFO.txt；
#   4) 用 Compress-Archive 压缩为 dist\CNN2-<版本>-win-x64.zip，
#      并打印产物路径与 SHA256。
#
# 用法：
#   .\release.ps1                 # 完整：编译 + 打包 + 压缩
#   .\release.ps1 -SkipBuild      # 跳过编译（用现有 x64\Release\CNN2.exe 打包）
#   .\release.ps1 -NoZip          # 只生成目录，不压缩 zip
#   .\release.ps1 -BuildTimeoutSec 1800   # 自定义编译超时（秒，默认 1800=30 分钟）
#
# 说明：
#   * 本脚本不改任何核心源码，也不改 build_and_test.ps1（原 34 条回归不受影响）。
#   * 不依赖网络；MSBuild 路径缺失时回退到 PATH 中的 MSBuild.exe。
#   * 含中文，须保存为 UTF-8 with BOM。
# ============================================================
[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$NoZip,
    [int]$BuildTimeoutSec = 1800
)

$ErrorActionPreference = 'Stop'

$root          = $PSScriptRoot
$versionHeader = Join-Path $root 'version.h'
$sln           = Join-Path $root 'CNN2.sln'
$relDir        = Join-Path $root 'x64\Release'
$deployDir     = Join-Path $root 'deploy'
$distDir       = Join-Path $root 'dist'
$msbuildDefault = 'D:\vc++\MSBuild\Current\Bin\MSBuild.exe'
$cudaPath      = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
# 运行时依赖：随 exe 一起发布，目标机器无需安装 CUDA / VC++ 运行库。
$runtimeDlls   = @('cudart64_12.dll', 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')

function Write-Step {
    param([string]$Text)
    Write-Host ''
    Write-Host ('===== ' + $Text + ' =====') -ForegroundColor Cyan
}

# 从 version.h 解析程序版本与本机模型格式版本。
function Get-VersionInfo {
    if (-not (Test-Path -LiteralPath $versionHeader)) { throw ('找不到 version.h: ' + $versionHeader) }
    $text = [IO.File]::ReadAllText($versionHeader)
    $mv = [regex]::Match($text, '#define\s+CNN2_VERSION\s+"([^"]+)"')
    if (-not $mv.Success) { throw '无法从 version.h 解析 CNN2_VERSION（期望 #define CNN2_VERSION "x.y.z"）' }
    $mm = [regex]::Match($text, '#define\s+CNN2_MODEL_VER\s+(\d+)')
    $modelVer = if ($mm.Success) { $mm.Groups[1].Value } else { '?' }
    return [pscustomobject]@{ Version = $mv.Groups[1].Value; ModelVer = $modelVer }
}

# MSBuild 编译 Release x64，带超时保护（超时则杀整棵进程树并按失败退出）。
function Invoke-Build {
    param([int]$TimeoutSec)
    $mb = $msbuildDefault
    if (-not (Test-Path -LiteralPath $mb)) {
        $cmd = Get-Command MSBuild.exe -ErrorAction SilentlyContinue
        if ($cmd) { $mb = $cmd.Source }
        else { throw '找不到 MSBuild.exe（请检查 D:\vc++\MSBuild\Current\Bin\MSBuild.exe 或 PATH）' }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $cudaPath 'bin\nvcc.exe'))) {
        Write-Host ('提示：未找到 CUDA 工具链 ' + $cudaPath + '，继续尝试编译（可能失败）') -ForegroundColor Yellow
    }
    $env:CUDA_PATH_V12_9 = $cudaPath
    Write-Host ('编译 Release x64: ' + $sln) -ForegroundColor Cyan
    Write-Host ('  MSBuild : ' + $mb)
    Write-Host ('  超时    : ' + $TimeoutSec + ' 秒')

    # 用 .NET ProcessStartInfo 直接启动（Start-Process 在本环境拿不到 ExitCode）。
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName  = $mb
    $psi.Arguments = '"' + $sln + '" /p:Configuration=Release /p:Platform=x64 /m /v:minimal /nologo'
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow  = $false
    $proc = [System.Diagnostics.Process]::Start($psi)
    $timedOut = -not $proc.WaitForExit($TimeoutSec * 1000)
    if ($timedOut) {
        & taskkill.exe /F /T /PID $proc.Id 2>$null | Out-Null
        try { $proc.Kill() } catch { }
        try { $proc.WaitForExit() } catch { }
        throw ('MSBuild 超时（>' + $TimeoutSec + ' 秒），已终止')
    }
    if ($proc.ExitCode -ne 0) { throw ('MSBuild 失败 exit=' + $proc.ExitCode) }
}

# 获取文件 SHA256（大写十六进制）。
function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

# 探测工具链版本信息（用于 BUILDINFO，均为「尽力而为」，失败给占位）。
function Get-ToolchainInfo {
    $msbuildPath = $msbuildDefault
    if (-not (Test-Path -LiteralPath $msbuildPath)) {
        $cmd = Get-Command MSBuild.exe -ErrorAction SilentlyContinue
        if ($cmd) { $msbuildPath = $cmd.Source }
    }
    $msbuildVer = '?'
    if (Test-Path -LiteralPath $msbuildPath) {
        try { $msbuildVer = (Get-Item -LiteralPath $msbuildPath).VersionInfo.ProductVersion } catch { }
    }
    $cudaVer = '?'
    $nvcc = Join-Path $cudaPath 'bin\nvcc.exe'
    if (Test-Path -LiteralPath $nvcc) {
        try {
            $out = & $nvcc --version 2>&1 | Out-String
            $mc = [regex]::Match($out, 'release\s+([\d.]+)')
            if ($mc.Success) { $cudaVer = $mc.Groups[1].Value } else { $cudaVer = '12.9' }
        } catch { $cudaVer = '12.9' }
    } else {
        $cudaVer = '12.9 (未检测到 nvcc)'
    }
    return [pscustomobject]@{ MSBuild = $msbuildVer; CUDA = $cudaVer }
}

# 探测 git 提交（本项目当前非 git 仓库时给占位）。
function Get-GitInfo {
    $gitExe = Get-Command git.exe -ErrorAction SilentlyContinue
    if (-not $gitExe) { return 'n/a (未安装 git)' }
    try {
        $inside = & git.exe -C $root rev-parse --is-inside-work-tree 2>$null
        if ($LASTEXITCODE -eq 0 -and "$inside".Trim() -eq 'true') {
            $head = (& git.exe -C $root rev-parse --short HEAD 2>$null)
            $dirty = (& git.exe -C $root status --porcelain 2>$null)
            $suffix = if ($dirty) { '+dirty' } else { '' }
            return ("$head$suffix")
        }
    } catch { }
    return 'n/a (非 git 仓库)'
}

# ======================= 主流程 =======================
$stageDir = $null
try {
    Write-Step '0. 读取版本'
    $ver = Get-VersionInfo
    Write-Host ('程序版本        : ' + $ver.Version)
    Write-Host ('模型格式版本    : ver=' + $ver.ModelVer)

    if (-not $SkipBuild) {
        Write-Step '1. 编译 Release x64'
        Invoke-Build -TimeoutSec $BuildTimeoutSec
    } else {
        Write-Step '1. 编译 Release x64（已 -SkipBuild）'
    }

    $exe = Join-Path $relDir 'CNN2.exe'
    if (-not (Test-Path -LiteralPath $exe)) { throw ('找不到可执行文件: ' + $exe + '（请先编译或不加 -SkipBuild）') }

    Write-Step '2. 准备目录'
    if (-not (Test-Path -LiteralPath $distDir)) { New-Item -ItemType Directory -Force -Path $distDir | Out-Null }
    $stageName = 'CNN2-' + $ver.Version + '-win-x64'
    $stageDir  = Join-Path $distDir $stageName
    if (Test-Path -LiteralPath $stageDir) { Remove-Item -LiteralPath $stageDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $stageDir | Out-Null
    Write-Host ('发布目录: ' + $stageDir)

    Write-Step '3. 复制 exe 与运行时 dll'
    Copy-Item -LiteralPath $exe -Destination (Join-Path $stageDir 'CNN2.exe') -Force
    Write-Host ('  + CNN2.exe')
    $missingDlls = @()
    foreach ($dll in $runtimeDlls) {
        $src = Join-Path $deployDir $dll
        if (-not (Test-Path -LiteralPath $src)) {
            # 回退：deploy 缺失时尝试 x64\Release 同目录（CUDA 运行时会随构建生成）
            $fallback = Join-Path $relDir $dll
            if (Test-Path -LiteralPath $fallback) {
                $src = $fallback
                Write-Host ('  ! deploy\' + $dll + ' 缺失，回退使用 x64\Release\' + $dll) -ForegroundColor Yellow
            } else {
                $missingDlls += $dll
                continue
            }
        }
        Copy-Item -LiteralPath $src -Destination (Join-Path $stageDir $dll) -Force
        Write-Host ('  + ' + $dll)
    }
    if ($missingDlls.Count -gt 0) {
        throw ('缺少运行时 dll: ' + ($missingDlls -join ', ') + '（请放入 deploy\ 后重试；目标机器将无法运行）')
    }

    Write-Step '4. 复制文档'
    foreach ($doc in @('README.md', 'CHANGELOG.md')) {
        $src = Join-Path $root $doc
        if (Test-Path -LiteralPath $src) {
            Copy-Item -LiteralPath $src -Destination (Join-Path $stageDir $doc) -Force
            Write-Host ('  + ' + $doc)
        } else {
            Write-Host ('  ! 缺少文档 ' + $doc + '（跳过）') -ForegroundColor Yellow
        }
    }

    Write-Step '5. 生成 BUILDINFO.txt'
    $exeHash  = Get-Sha256 (Join-Path $stageDir 'CNN2.exe')
    $exeItem  = Get-Item -LiteralPath (Join-Path $stageDir 'CNN2.exe')
    $buildTime = $exeItem.LastWriteTime.ToString('yyyy-MM-ddTHH:mm:sszzz')
    $tool = Get-ToolchainInfo
    $hostName = if ($env:COMPUTERNAME) { $env:COMPUTERNAME } else { 'unknown-host' }
    $userName = if ($env:USERNAME) { $env:USERNAME } else { 'unknown-user' }
    $gitInfo  = Get-GitInfo
    # 注意：此处必须用双引号插值，不可在 @() 内用 'x' + $y —— PowerShell 中逗号
    # 优先级高于 +，会把每个 "标签 + 值" 拆成两个数组元素（见阶段4.5 修复记录）。
    $lines = @(
        'CNN2 发布构建信息 (BUILDINFO)',
        '========================================',
        "Version        : $($ver.Version)",
        "ModelFormatVer : ver=$($ver.ModelVer)",
        "BuildDate      : $buildTime",
        "BuildHost      : $hostName",
        "BuildUser      : $userName",
        "GitCommit      : $gitInfo",
        'Toolchain      : MSVC v143 (Visual Studio 2022), C++17 (stdcpp17)',
        "MSBuildVersion : $($tool.MSBuild)",
        "CUDA           : $($tool.CUDA)",
        'Configuration  : Release x64',
        'Executable     : CNN2.exe',
        "ExeSizeBytes   : $($exeItem.Length)",
        "ExeSHA256      : $exeHash",
        "PackageDate    : $((Get-Date).ToString('yyyy-MM-ddTHH:mm:sszzz'))",
        '========================================'
    )
    $buildInfoPath = Join-Path $stageDir 'BUILDINFO.txt'
    # 含中文，写为 UTF-8 with BOM（与项目其它含中文文件一致，避免 ANSI 工具乱码）
    [IO.File]::WriteAllText($buildInfoPath, (($lines -join "`r`n") + "`r`n"), (New-Object Text.UTF8Encoding($true)))
    Write-Host ('  + BUILDINFO.txt (ExeSHA256=' + $exeHash.Substring(0, 16) + '...)')

    Write-Step '6. 打包 zip'
    $zipPath = $null
    if (-not $NoZip) {
        $zipPath = Join-Path $distDir ($stageName + '.zip')
        if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
        Compress-Archive -Path (Join-Path $stageDir '*') -DestinationPath $zipPath -CompressionLevel Optimal -Force
        Write-Host ('  + ' + $zipPath)
    } else {
        Write-Host '  （-NoZip：跳过压缩）'
    }

    Write-Step '7. 汇总'
    Write-Host ('程序版本    : ' + $ver.Version)
    Write-Host ('发布目录    : ' + $stageDir)
    Write-Host ('  CNN2.exe SHA256 : ' + $exeHash)
    if ($zipPath) {
        $zipHash = Get-Sha256 $zipPath
        $zipSize = (Get-Item -LiteralPath $zipPath).Length
        Write-Host ('压缩包      : ' + $zipPath)
        Write-Host ('  ZIP 大小       : ' + $zipSize + ' 字节')
        Write-Host ('  ZIP SHA256     : ' + $zipHash)
    } else {
        Write-Host '压缩包      : （未生成）'
    }
    Write-Host ''
    Write-Host '发布完成。校验方式：Get-FileHash <文件> -Algorithm SHA256' -ForegroundColor Green
    exit 0
}
catch {
    Write-Host ''
    Write-Host ('[FAIL] ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
