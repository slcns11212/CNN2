# ============================================================
# CNN2 - bench.ps1  （阶段5：性能基准 / 可复跑）
#
# 目的：用现有 exe 在固定 seed、固定数据下测量并输出一张基准表：
#   * 数据加载耗时（--time 报告值）
#   * 若干 epoch 的训练墙钟（--epochs，取平均）
#   * 训练吞吐（样本/秒，按扩增后样本数估算）
#   * CPU / GPU 推理吞吐（test --cpu / --gpu，样本/秒）
#   * GPU 显存估算（程序打印的“预计显存占用”）
#
# 设计原则：
#   * 不改核心源码；只用现有命令行参数 --time + 解析 stdout（GBK=936）。
#   * 固定 --seed，保证训练可复现；推理用本次训练出的模型（同配置）。
#   * 每个 exe 独立进程 + 超时树杀；缺 GPU 自动 Skip。
#   * 输出 bench_results\bench_<时间戳>.md + bench_results\bench_latest.md，并在 stdout 打印。
#
# 覆盖配置：默认 depth2 / --depth 3 / --res --depth 3 / --BN。
#   * 训练：GPU 全跑；CPU 跑 depth2/depth3/res（CPU 不支持 BN 训练，自动跳过）。
#   * 推理：同配置模型的 CPU + GPU 推理。
#
# 用法：
#   .\bench.ps1                    # 默认 2 epoch、seed=12345
#   .\bench.ps1 -Epochs 1          # 更快
#   .\bench.ps1 -SkipCpuTrain      # 只跑 GPU 训练（跳过 CPU 训练）
#   .\bench.ps1 -SkipGpu           # 无 GPU 环境
#
# 本脚本不改任何源码，也不修改 build_and_test.ps1。
# ============================================================
[CmdletBinding()]
param(
    [int]$Seed = 12345,
    [int]$Epochs = 2,
    [switch]$SkipCpuTrain,
    [switch]$SkipGpu,
    [int]$TimeoutSec = 1800,
    [string]$OutDir = 'bench_results'
)

$ErrorActionPreference = 'Stop'

$root     = $PSScriptRoot
$rel      = Join-Path $root 'x64\Release'
$exe      = Join-Path $rel 'CNN2.exe'
$testDir  = [IO.Path]::GetFullPath((Join-Path $rel '..\testDigits'))
$outRoot  = Join-Path $root $OutDir
$modelDir = Join-Path $outRoot 'models'
$tmp      = Join-Path ([IO.Path]::GetTempPath()) ('cnn2_bench_' + [guid]::NewGuid().ToString('N'))

$script:gpuOk = -not $SkipGpu
$inv = [System.Globalization.CultureInfo]::InvariantCulture

New-Item -ItemType Directory -Force -Path $outRoot, $modelDir, $tmp | Out-Null

# ---------------- 工具函数 ----------------

# 以 GBK 捕获 exe 输出；工作目录固定为 x64\Release（相对路径 trainingDigits / ..\testDigits 依赖它）。
# 超时保护：独立进程 + WaitForExit(ms)，超过 TimeoutSec 秒即树杀，返回 TimedOut=$true。
function Invoke-Exe {
    param([string]$Argline, [int]$TimeoutSec = 1800)
    $out  = Join-Path $tmp ([guid]::NewGuid().ToString('N') + '.txt')
    $line = 'chcp 936>nul & cd /d "' + $rel + '" & "' + $exe + '" ' + $Argline + ' > "' + $out + '" 2>&1'
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $env:ComSpec
    $psi.Arguments = '/c ' + $line
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $proc = [System.Diagnostics.Process]::Start($psi)
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $timedOut = -not $proc.WaitForExit($TimeoutSec * 1000)
    $sw.Stop()
    if ($timedOut) {
        & taskkill.exe /F /T /PID $proc.Id 2>$null | Out-Null
        try { $proc.Kill() } catch { }
        try { $proc.WaitForExit() } catch { }
    }
    $code = $proc.ExitCode
    $text = ''
    if (Test-Path -LiteralPath $out) {
        $text = [IO.File]::ReadAllText($out, [Text.Encoding]::GetEncoding(936))
        Remove-Item -LiteralPath $out -Force -ErrorAction SilentlyContinue
    }
    return @{ Exit = $code; Text = $text; TimedOut = $timedOut; WallSec = $sw.Elapsed.TotalSeconds }
}

# 把 exe 的友好时长（"1.23 秒" / "1 分 23.45 秒"）转成秒；无法解析返回 $null。
function ConvertTo-Seconds {
    param([string]$s)
    if ([string]::IsNullOrWhiteSpace($s)) { return $null }
    $m = [regex]::Match($s, '(?:(\d+)\s*分\s*)?([0-9]+(?:\.[0-9]+)?)\s*秒')
    if (-not $m.Success) { return $null }
    $min = if ($m.Groups[1].Success) { [double]$m.Groups[1].Value } else { 0.0 }
    return $min * 60.0 + [double]$m.Groups[2].Value
}

function Test-GpuUnavailable {
    param([string]$Text)
    return ($Text -match 'no CUDA device available')
}

# 解析一次训练 stdout（--time）。返回加载/总时长、样本数、每 epoch 墙钟与计算时长、显存估算。
function Parse-Train {
    param([string]$Text)
    $o = [ordered]@{ LoadSec = $null; TotalSec = $null; Samples = $null; EpochWalls = @(); EpochComputes = @(); MemMB = $null }

    $m = [regex]::Match($Text, '扩增后\s*(\d+)\s*样本')
    if ($m.Success) { $o.Samples = [int]$m.Groups[1].Value }

    # GPU 路径有显式的“数据加载耗时:”；CPU 路径只有每 epoch 明细与“数据加载累计”。
    $lm = [regex]::Match($Text, '数据加载耗时:\s*([^\r\n]+)')
    if ($lm.Success) { $o.LoadSec = ConvertTo-Seconds $lm.Groups[1].Value }
    if ($null -eq $o.LoadSec) {
        $lc = [regex]::Match($Text, '数据加载累计\s*([^，）\r\n]+)')
        if ($lc.Success) { $o.LoadSec = ConvertTo-Seconds $lc.Groups[1].Value }
    }

    $tm = [regex]::Match($Text, '总训练时间:\s*([^\r\n（]+)')
    if ($tm.Success) { $o.TotalSec = ConvertTo-Seconds $tm.Groups[1].Value }

    foreach ($line in ($Text -split "`n")) {
        if ($line -match '^epoch\s+(\d+)\s+耗时:') {
            $idx  = $line.IndexOf('（')
            $head = if ($idx -ge 0) { $line.Substring(0, $idx) } else { $line }
            $dm = [regex]::Match($head, '耗时:\s*(.+)$')
            if ($dm.Success) {
                $w = ConvertTo-Seconds $dm.Groups[1].Value
                if ($null -ne $w) { $o.EpochWalls += $w }
            }
            # CPU 明细：“（数据加载 X，计算 Y）”；GPU 无此括号。
            $cm = [regex]::Match($line, '计算\s*([^）]+)')
            if ($cm.Success) {
                $c = ConvertTo-Seconds $cm.Groups[1].Value
                if ($null -ne $c) { $o.EpochComputes += $c }
            }
        }
    }

    $mm = [regex]::Match($Text, '≈\s*([0-9.]+)\s*MB')
    if ($mm.Success) { $o.MemMB = [double]$mm.Groups[1].Value }

    return $o
}

function Get-Avg {
    param($Arr)
    if ($null -eq $Arr -or $Arr.Count -eq 0) { return $null }
    $sum = 0.0; foreach ($v in $Arr) { $sum += [double]$v }
    return $sum / $Arr.Count
}

function Fmt {
    param($v, [int]$d = 2)
    if ($null -eq $v) { return '-' }
    return ([double]$v).ToString('N' + $d, $inv)
}

# 采集环境信息（表格头部）。
function Get-EnvInfo {
    $info = [ordered]@{}
    $info['主机'] = $env:COMPUTERNAME
    try { $info['操作系统'] = (Get-CimInstance Win32_OperatingSystem).Caption } catch { $info['操作系统'] = '未知' }
    try {
        $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
        $info['CPU'] = $cpu.Name
        $info['核心'] = ('{0} 物理核 / {1} 逻辑核' -f $cpu.NumberOfCores, $cpu.NumberOfLogicalProcessors)
        $info['内存'] = ('{0:N1} GB' -f ((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB))
    } catch { $info['CPU'] = '未知' }

    # GPU / 驱动
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($smi) {
        try {
            $g = & $smi.Source --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
            if ($LASTEXITCODE -eq 0 -and $g) { $info['GPU'] = (($g | Where-Object { $_ }) -join '; ') } else { $info['GPU'] = 'nvidia-smi 无输出' }
        } catch { $info['GPU'] = 'nvidia-smi 调用失败' }
    } else { $info['GPU'] = '未找到 nvidia-smi' }

    # CUDA toolkit（从常见安装目录读 version.json）
    $cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $cudaRoot) {
        $vdir = Get-ChildItem -LiteralPath $cudaRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($vdir) {
            $ver = $vdir.Name
            $vj = Join-Path $vdir.FullName 'version.json'
            if (Test-Path -LiteralPath $vj) {
                try {
                    $j = Get-Content -LiteralPath $vj -Raw | ConvertFrom-Json
                    if ($j.cuda_nvcc.version) { $ver = $j.cuda_nvcc.version }
                    elseif ($j.cuda.version) { $ver = $j.cuda.version }
                } catch { }
            }
            $info['CUDA'] = $ver
        }
    } else { $info['CUDA'] = '未检测到 CUDA toolkit 目录' }

    # exe 构建信息（--version，不初始化 CUDA）
    if (Test-Path -LiteralPath $exe) {
        $vr = Invoke-Exe '--version' -TimeoutSec 60
        $b = [regex]::Match($vr.Text, '构建时间[：:]\s*([^\r\n]+)')
        if ($b.Success) { $info['exe 构建'] = $b.Groups[1].Value.Trim() }
        $info['exe SHA256'] = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.Substring(0, 16)
        $info['exe 大小'] = ('{0:N0} 字节' -f (Get-Item -LiteralPath $exe).Length)
    } else { $info['exe'] = ('缺失: ' + $exe) }

    # 训练集文件数：CPU 训练在线增强、不预扩增，也没有“扩增后”打印，
    # 故缓存此值作为 CPU 每 epoch 样本数（= 每 epoch 遍历的原始样本数）。
    $script:trainFiles = @(Get-ChildItem -LiteralPath (Join-Path $rel 'trainingDigits') -Filter *.txt).Count
    $info['训练数据'] = ('trainingDigits（' + $script:trainFiles + ' 张）')
    $info['测试数据'] = ('..\testDigits（' + ((Get-ChildItem -LiteralPath $testDir -Filter *.txt).Count) + ' 张）')
    $info['seed'] = $Seed
    $info['epochs'] = $Epochs
    return $info
}

# ---------------- 基准执行 ----------------

function Invoke-TrainBench {
    param($Cfg, [string]$BackendArg, [string]$ModelPath, [string]$ExtraArgs)
    $backend = if ($BackendArg -eq '--gpu') { 'GPU' } else { 'CPU' }
    $row = [pscustomobject]@{
        Config = $Cfg.Name; Backend = $backend; LoadSec = $null; EpochSec = $null
        TotalSec = $null; Tput = $null; MemMB = $null; Status = ''; Note = ''
    }
    if ($BackendArg -eq '--gpu' -and -not $script:gpuOk) { $row.Status = 'SKIP'; $row.Note = 'GPU 不可用'; return $row }

    $argline = 'trainingDigits bkc ' + $BackendArg + ' --epochs ' + $Epochs + ' --seed ' + $Seed +
               ' --time --model "' + $ModelPath + '"'
    if ($ExtraArgs) { $argline += ' ' + $ExtraArgs }

    $r = Invoke-Exe $argline -TimeoutSec $TimeoutSec
    if ($r.TimedOut) { $row.Status = 'FAIL'; $row.Note = ('超时(已 Kill, ' + $TimeoutSec + 's)'); return $row }
    if (Test-GpuUnavailable $r.Text) { $script:gpuOk = $false; $row.Status = 'SKIP'; $row.Note = 'GPU 不可用'; return $row }
    if ($r.Exit -ne 0) { $row.Status = 'FAIL'; $row.Note = ('exit=' + $r.Exit); return $row }

    $p = Parse-Train $r.Text
    if ($null -eq $p.Samples -and $null -ne $script:trainFiles) {
        # CPU 训练无“扩增后 N 样本”行（该段在 GPU 分支内）：CPU 每 epoch 在线增强，
        # 每 epoch 样本数 = 训练集原始文件数。用缓存值回退，使 CPU 吞吐可计算。
        $p.Samples = $script:trainFiles
        $row.Note = 'CPU 样本数=训练集文件数'
    }
    $row.LoadSec = $p.LoadSec
    $row.TotalSec = $p.TotalSec
    $row.MemMB = $p.MemMB
    $row.EpochSec = Get-Avg $p.EpochWalls
    $compute = Get-Avg $p.EpochComputes
    if ($null -eq $compute) { $compute = $row.EpochSec }   # GPU：epoch 耗时即计算时长
    if ($null -ne $p.Samples -and $null -ne $compute -and $compute -gt 0) {
        $row.Tput = $p.Samples / $compute
    }
    $row.Status = 'OK'
    return $row
}

function Invoke-InferBench {
    param($Cfg, [string]$BackendArg, [string]$ModelPath, [string]$ExtraArgs)
    $backend = if ($BackendArg -eq '--gpu') { 'GPU' } else { 'CPU' }
    $row = [pscustomobject]@{
        Config = $Cfg.Name; Backend = $backend; Files = $null; WallSec = $null
        Tput = $null; Acc = $null; Status = ''; Note = ''
    }
    if ($BackendArg -eq '--gpu' -and -not $script:gpuOk) { $row.Status = 'SKIP'; $row.Note = 'GPU 不可用'; return $row }

    $argline = '..\testDigits test ' + $BackendArg + ' --model "' + $ModelPath + '"'
    if ($ExtraArgs) { $argline += ' ' + $ExtraArgs }

    $r = Invoke-Exe $argline -TimeoutSec $TimeoutSec
    $row.WallSec = $r.WallSec
    if ($r.TimedOut) { $row.Status = 'FAIL'; $row.Note = ('超时(已 Kill, ' + $TimeoutSec + 's)'); return $row }
    if (Test-GpuUnavailable $r.Text) { $script:gpuOk = $false; $row.Status = 'SKIP'; $row.Note = 'GPU 不可用'; return $row }
    if ($r.Exit -ne 0) { $row.Status = 'FAIL'; $row.Note = ('exit=' + $r.Exit); return $row }

    $fm = [regex]::Match($r.Text, '测试文件数:\s*(\d+)')
    if ($fm.Success) { $row.Files = [int]$fm.Groups[1].Value }
    $am = [regex]::Match($r.Text, '标准准确率:\s*([0-9.]+)\s*\((\d+)/(\d+)\)')
    if ($am.Success) { $row.Acc = [double]$am.Groups[1].Value }
    if ($null -ne $row.Files -and $row.WallSec -gt 0) { $row.Tput = $row.Files / $row.WallSec }
    $row.Status = 'OK'
    return $row
}

# ---------------- 主流程 ----------------

if (-not (Test-Path -LiteralPath $exe)) { throw ('找不到 exe: ' + $exe + '（请先编译 Release x64）') }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$startTime = Get-Date
$envInfo = Get-EnvInfo

function Write-Log { param([string]$Msg, [string]$Color = 'Gray') Write-Host $Msg -ForegroundColor $Color }

Write-Log ''
Write-Log '==================== CNN2 性能基准 ====================' 'Cyan'
Write-Log ('seed=' + $Seed + '  epochs=' + $Epochs + '  CPU训练=' + (-not $SkipCpuTrain) + '  GPU=' + (-not $SkipGpu))
Write-Log ('exe = ' + $exe)
Write-Log ''

# 配置表：TrainExtra 传给训练，CpuTrain 是否跑 CPU 训练，InferExtra 传给推理。
$configs = @(
    [pscustomobject]@{ Name = 'depth2'; TrainExtra = '';                CpuTrain = $true;  InferExtra = ''     },
    [pscustomobject]@{ Name = 'depth3'; TrainExtra = '--depth 3';       CpuTrain = $true;  InferExtra = ''     },
    [pscustomobject]@{ Name = 'res';    TrainExtra = '--res --depth 3'; CpuTrain = $true;  InferExtra = ''     },
    [pscustomobject]@{ Name = 'BN';     TrainExtra = '--BN';            CpuTrain = $false; InferExtra = '--BN' }
)

$trainRows  = New-Object System.Collections.ArrayList
$inferRows  = New-Object System.Collections.ArrayList

foreach ($cfg in $configs) {
    Write-Log ('---- 配置 ' + $cfg.Name + ' ----') 'Cyan'
    $modelGpu = Join-Path $modelDir ($cfg.Name + '-gpu.txt')
    $modelCpu = Join-Path $modelDir ($cfg.Name + '-cpu.txt')
    Remove-Item -LiteralPath $modelGpu, $modelCpu -Force -ErrorAction SilentlyContinue

    # 1) GPU 训练（BN 只能 GPU 训练）
    Write-Log ('  [训练] GPU ' + $cfg.Name + ' ...')
    $tr = Invoke-TrainBench -Cfg $cfg -BackendArg '--gpu' -ModelPath $modelGpu -ExtraArgs $cfg.TrainExtra
    [void]$trainRows.Add($tr)
    Write-Log ('    -> ' + $tr.Status + '  加载=' + (Fmt $tr.LoadSec) + 's  epoch=' + (Fmt $tr.EpochSec) +
               's  吞吐=' + (Fmt $tr.Tput 0) + ' 样本/s  显存=' + (Fmt $tr.MemMB 1) + 'MB ' + $tr.Note) `
              $(if ($tr.Status -eq 'OK') { 'Green' } elseif ($tr.Status -eq 'SKIP') { 'Yellow' } else { 'Red' })

    # 2) CPU 训练（BN 不支持；-SkipCpuTrain 可整体跳过）
    if ($SkipCpuTrain) {
        $tc = [pscustomobject]@{ Config = $cfg.Name; Backend = 'CPU'; LoadSec = $null; EpochSec = $null; TotalSec = $null; Tput = $null; MemMB = $null; Status = 'SKIP'; Note = '-SkipCpuTrain' }
        [void]$trainRows.Add($tc)
        Write-Log '  [训练] CPU  SKIP (-SkipCpuTrain)' 'Yellow'
    } elseif (-not $cfg.CpuTrain) {
        $tc = [pscustomobject]@{ Config = $cfg.Name; Backend = 'CPU'; LoadSec = $null; EpochSec = $null; TotalSec = $null; Tput = $null; MemMB = $null; Status = 'SKIP'; Note = 'CPU 训练不支持 BN' }
        [void]$trainRows.Add($tc)
        Write-Log '  [训练] CPU  SKIP（CPU 训练不支持 BN）' 'Yellow'
    } else {
        Write-Log ('  [训练] CPU ' + $cfg.Name + ' ...')
        # CPU 训练参数去掉 --BN（BN 配置本就不会到这）；保留 --depth/--res
        $cpuExtra = ($cfg.TrainExtra -replace '--BN', '').Trim()
        $tc = Invoke-TrainBench -Cfg $cfg -BackendArg '--cpu' -ModelPath $modelCpu -ExtraArgs $cpuExtra
        [void]$trainRows.Add($tc)
        Write-Log ('    -> ' + $tc.Status + '  加载=' + (Fmt $tc.LoadSec) + 's  epoch=' + (Fmt $tc.EpochSec) +
                   's  吞吐=' + (Fmt $tc.Tput 0) + ' 样本/s  ' + $tc.Note) `
                  $(if ($tc.Status -eq 'OK') { 'Green' } elseif ($tc.Status -eq 'SKIP') { 'Yellow' } else { 'Red' })
    }

    # 3) 推理：优先用 GPU 训练出的模型，否则用 CPU 模型
    $infModel = $null
    if (Test-Path -LiteralPath $modelGpu) { $infModel = $modelGpu }
    elseif (Test-Path -LiteralPath $modelCpu) { $infModel = $modelCpu }

    foreach ($bk in @('--cpu', '--gpu')) {
        if ($bk -eq '--gpu' -and $SkipGpu) {
            [void]$inferRows.Add([pscustomobject]@{ Config = $cfg.Name; Backend = 'GPU'; Files = $null; WallSec = $null; Tput = $null; Acc = $null; Status = 'SKIP'; Note = '-SkipGpu' })
            continue
        }
        if ($null -eq $infModel) {
            [void]$inferRows.Add([pscustomobject]@{ Config = $cfg.Name; Backend = $(if ($bk -eq '--gpu') { 'GPU' } else { 'CPU' }); Files = $null; WallSec = $null; Tput = $null; Acc = $null; Status = 'SKIP'; Note = '无可用训练模型' })
            continue
        }
        $ir = Invoke-InferBench -Cfg $cfg -BackendArg $bk -ModelPath $infModel -ExtraArgs $cfg.InferExtra
        [void]$inferRows.Add($ir)
        Write-Log ('  [推理] ' + $ir.Backend + ' ' + $cfg.Name + ' -> ' + $ir.Status +
                   '  样本=' + $(if ($null -ne $ir.Files) { $ir.Files } else { '-' }) +
                   '  墙钟=' + (Fmt $ir.WallSec) + 's  吞吐=' + (Fmt $ir.Tput 0) + ' 样本/s  准确率=' + (Fmt $ir.Acc 4) + ' ' + $ir.Note) `
                  $(if ($ir.Status -eq 'OK') { 'Green' } elseif ($ir.Status -eq 'SKIP') { 'Yellow' } else { 'Red' })
    }
    Write-Log ''
}

# ---------------- 生成报告 ----------------

$sb = New-Object System.Text.StringBuilder
function Add-Line { param([string]$s = '') [void]$sb.AppendLine($s) }

Add-Line '# CNN2 性能基准报告'
Add-Line ''
Add-Line ('- 生成时间: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Add-Line ('- 耗时: ' + (Fmt ((Get-Date) - $startTime).TotalSeconds 1) + ' 秒')
Add-Line ('- seed: ' + $Seed + '，epochs: ' + $Epochs)
Add-Line ''
Add-Line '## 环境'
Add-Line ''
Add-Line '| 项 | 值 |'
Add-Line '| --- | --- |'
foreach ($k in $envInfo.Keys) { Add-Line ('| ' + $k + ' | ' + $envInfo[$k] + ' |') }
Add-Line ''

Add-Line '## 训练基准'
Add-Line ''
Add-Line '| 配置 | 后端 | 加载(s) | epoch墙钟(s/epoch) | 训练吞吐(样本/s) | 总训练(s) | 显存(MB) | 状态 | 备注 |'
Add-Line '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |'
foreach ($r in $trainRows) {
    Add-Line ('| ' + $r.Config + ' | ' + $r.Backend + ' | ' + (Fmt $r.LoadSec) + ' | ' + (Fmt $r.EpochSec) +
              ' | ' + (Fmt $r.Tput 0) + ' | ' + (Fmt $r.TotalSec) + ' | ' + (Fmt $r.MemMB 1) + ' | ' + $r.Status + ' | ' + $r.Note + ' |')
}
Add-Line ''

Add-Line '## 推理基准'
Add-Line ''
Add-Line '| 配置 | 后端 | 样本数 | 墙钟(s) | 吞吐(样本/s) | 准确率 | 状态 | 备注 |'
Add-Line '| --- | --- | ---: | ---: | ---: | ---: | --- | --- |'
foreach ($r in $inferRows) {
    Add-Line ('| ' + $r.Config + ' | ' + $r.Backend + ' | ' + $(if ($null -ne $r.Files) { $r.Files } else { '-' }) +
              ' | ' + (Fmt $r.WallSec) + ' | ' + (Fmt $r.Tput 0) + ' | ' + (Fmt $r.Acc 4) + ' | ' + $r.Status + ' | ' + $r.Note + ' |')
}
Add-Line ''

Add-Line '## 指标口径'
Add-Line ''
Add-Line '- **数据加载(s)**：`--time` 报告的加载耗时（GPU：`数据加载耗时`；CPU：`数据加载累计`）。'
Add-Line '- **epoch墙钟(s/epoch)**：各 epoch `耗时` 的平均值（CPU 含该 epoch 加载，GPU 不含加载）。'
Add-Line '- **训练吞吐(样本/s)**：扩增后样本数 ÷ 计算时长（CPU 取 `计算` 明细，GPU 取 epoch 耗时）。'
Add-Line '- **显存(MB)**：程序 `预计显存占用 ≈ X MB`（仅 GPU 训练打印）。'
Add-Line '- **推理吞吐**：测试文件数 ÷ 进程端到端墙钟（含启动/加载模型/预处理/推理）。'
Add-Line '- **推理模型**：本次训练产物（同配置）；文件位于 `bench_results\models\`。'
Add-Line ''

$md = $sb.ToString()

$reportPath = Join-Path $outRoot ('bench_' + $stamp + '.md')
$latestPath = Join-Path $outRoot 'bench_latest.md'
[IO.File]::WriteAllText($reportPath, $md, [Text.UTF8Encoding]::new($true))
[IO.File]::WriteAllText($latestPath, $md, [Text.UTF8Encoding]::new($true))

Write-Log ''
Write-Log '==================== 报告 ====================' 'Cyan'
Write-Host $md
Write-Log ('已写入: ' + $reportPath, 'Green')
Write-Log ('已写入: ' + $latestPath, 'Green')

Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue

$failCount = @($trainRows | Where-Object { $_.Status -eq 'FAIL' }).Count + @($inferRows | Where-Object { $_.Status -eq 'FAIL' }).Count
if ($failCount -gt 0) { exit 1 } else { exit 0 }
