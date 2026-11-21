# ============================================================
# CNN2 - build_and_test.ps1
# 阶段0：回归测试 golden 基线 + 一键 CI
#
# 做三件事：
#   1) MSBuild 编译 Release x64（需本机 VS/CUDA，遵循项目既有工具链）
#   2) 逐条运行 tests\golden.txt 中的回归命令并比对
#   3) 打印 PASS/FAIL/SKIP 与汇总；有失败则非零退出码
#
# 用法：
#   .\build_and_test.ps1                # 编译 + 全部回归 + 训练冒烟
#   .\build_and_test.ps1 -SkipBuild     # 跳过编译（用现有 exe 跑回归）
#   .\build_and_test.ps1 -SkipSmoke     # 跳过 1 epoch 训练冒烟
#   .\build_and_test.ps1 -SkipGpu       # 跳过 GPU 用例（无 GPU 时也可自动跳过）
#
# 注意：
#   * 本脚本不改任何源码，只做编译 + 回归。
#   * 推理（固定模型 + 固定数据）为精确数值 golden；gen 只做结构 golden。
#   * 夹具生成一次后固定；本脚本不会重建 model64.txt / model_bn.txt（否则会改变权重）。
#   * exe 中文输出为 GBK(936)，本脚本以 GBK 读取后再匹配中文关键字。
# ============================================================
[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$SkipSmoke,
    [switch]$SkipGpu,
    [switch]$OnlyAcm      # T8：仅运行 acm 专项块（跳过其余回归，便于快速验证）
)

$ErrorActionPreference = 'Stop'

# ---- T8：既有回归隔离 ------------------------------------------------------
# 默认保存格式已改为 acm 二进制，但既有回归（Assert-Struct 用正则解析 #SIG / K1 …
# 结构行、Ensure-Fixtures 读 model32.txt 改 FC 行造坏夹具、tests\fixtures\*.txt、
# epsave 断言 ep_0.txt）全部依赖「文本」模型。这里在脚本最前（任何 CNN2.exe 调用
# 之前）注入 CNN2_TXTMODEL=1，使既有回归一律走 txt 路径，从而上述断言零改动继续通过。
# acm 专项用例（文末 ===== 8 =====）会临时清除该变量以显式走 acm 路径。
$env:CNN2_TXTMODEL = '1'

$root     = $PSScriptRoot
$rel      = Join-Path $root 'x64\Release'
$testsDir = Join-Path $root 'tests'
$fx       = Join-Path $testsDir 'fixtures'
$exe      = Join-Path $rel 'CNN2.exe'
$sln      = Join-Path $root 'CNN2.sln'
$tmp      = Join-Path $testsDir '_run'
$msbuildDefault = 'D:\vc++\MSBuild\Current\Bin\MSBuild.exe'
$cudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'

$script:pass = 0
$script:fail = 0
$script:skip = 0
$script:gpuOk = -not $SkipGpu
$script:results = New-Object System.Collections.ArrayList

function Write-Result {
    param([string]$Name, [string]$Status, [string]$Detail = '')
    $color = 'Gray'
    switch ($Status) {
        'PASS' { $color = 'Green' }
        'FAIL' { $color = 'Red' }
        'SKIP' { $color = 'Yellow' }
    }
    Write-Host ("[{0}] {1} {2}" -f $Status, $Name, $Detail) -ForegroundColor $color
    [void]$script:results.Add([pscustomobject]@{ Name = $Name; Status = $Status; Detail = $Detail })
    switch ($Status) {
        'PASS' { $script:pass++ }
        'FAIL' { $script:fail++ }
        'SKIP' { $script:skip++ }
    }
}

# 以 GBK 捕获 exe 输出；工作目录默认 x64\Release（相对路径 ..\testDigits 等依赖它）。
#   T8：新增可选 -WorkDir 参数，使 acm 专项用例可在临时目录内生成/读取（默认名 model.acm
#       落点可控），既不影响既有调用的相对路径语义，也避免污染 x64\Release。
# 超时保护：每个 exe 用独立进程运行 + WaitForExit(ms)，超过 $TimeoutSec 秒即 Kill，
#            避免任何死循环/病态慢把整个脚本无限挂起（返回 TimedOut=$true）。
function Invoke-Exe {
    param([string]$Argline, [int]$TimeoutSec = 600, [string]$WorkDir = $rel)
    $out = Join-Path $tmp ([guid]::NewGuid().ToString('N') + '.txt')
    $line = 'chcp 936>nul & cd /d "' + $WorkDir + '" & "' + $exe + '" ' + $Argline + ' > "' + $out + '" 2>&1'
    # 用 .NET ProcessStartInfo 直接调用 cmd.exe，避免 PowerShell 对含引号/重定向的命令行二次转义
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $env:ComSpec
    $psi.Arguments = '/c ' + $line
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $proc = [System.Diagnostics.Process]::Start($psi)
    $timedOut = -not $proc.WaitForExit($TimeoutSec * 1000)
    if ($timedOut) {
        # 杀整棵进程树：cmd.exe 被杀后其子进程 CNN2.exe 不会自动退，需 /T 一并终止
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
    return @{ Exit = $code; Text = $text; TimedOut = $timedOut }
}

function Test-GpuUnavailable {
    param([string]$Text)
    return ($Text -match 'no CUDA device available')
}

# 推理数值 golden：断言 exit=0 且“标准准确率 (correct/total)”精确匹配
function Assert-Acc {
    param([string]$Name, [string]$Argline, [int]$Num, [int]$Den, [switch]$Gpu)
    if ($Gpu -and (-not $script:gpuOk)) { Write-Result $Name 'SKIP' 'GPU 不可用'; return }
    $r = Invoke-Exe $Argline
    if ($r.TimedOut) { Write-Result $Name 'FAIL' '执行超时（已 Kill）'; return }
    if (Test-GpuUnavailable $r.Text) { $script:gpuOk = $false; Write-Result $Name 'SKIP' 'GPU 不可用'; return }
    if ($r.Exit -ne 0) { Write-Result $Name 'FAIL' ('exit=' + $r.Exit); return }
    $m = [regex]::Match($r.Text, '标准准确率:\s*([0-9.]+)\s*\((\d+)/(\d+)\)')
    if (-not $m.Success) { Write-Result $Name 'FAIL' '未找到“标准准确率 (correct/total)”；可用 -SkipBuild 单独排查'; return }
    $c = [int]$m.Groups[2].Value
    $t = [int]$m.Groups[3].Value
    if ($c -eq $Num -and $t -eq $Den) { Write-Result $Name 'PASS' ($c.ToString() + '/' + $t.ToString()) }
    else { Write-Result $Name 'FAIL' ('期望 ' + $Num + '/' + $Den + '，实得 ' + $c + '/' + $t) }
}

# gen 结构 golden：运行 gen 后读取生成文件的结构行，断言包含全部期望片段
function Assert-Struct {
    param([string]$Name, [string]$Argline, [string]$ModelPath, [string[]]$Expect)
    $r = Invoke-Exe $Argline
    if ($r.TimedOut) { Write-Result $Name 'FAIL' '执行超时（已 Kill）'; return }
    if ($r.Exit -ne 0) { Write-Result $Name 'FAIL' ('gen exit=' + $r.Exit); return }
    if (-not (Test-Path -LiteralPath $ModelPath)) { Write-Result $Name 'FAIL' '未生成模型文件'; return }
    $lines = [IO.File]::ReadAllLines($ModelPath, [Text.Encoding]::GetEncoding(936))
    $struct = ($lines | Where-Object { $_ -match '^(#SIG|K[0-9]+ |B[0-9]+ |BN[0-9]+ |P[0-9]+ |R[0-9]+ |RP[0-9]+ |PB[0-9]+ |FC )' }) -join ' || '
    $missing = @()
    foreach ($e in $Expect) { if ($struct.IndexOf($e) -lt 0) { $missing += $e } }
    if ($missing.Count -eq 0) { Write-Result $Name 'PASS' $struct }
    else { Write-Result $Name 'FAIL' ('缺少 [' + ($missing -join '; ') + '] 实得: ' + $struct) }
}

# 报错 golden：断言 exit=1 且输出含关键提示
function Assert-Error {
    param([string]$Name, [string]$Argline, [string]$Keyword)
    $r = Invoke-Exe $Argline
    if ($r.TimedOut) { Write-Result $Name 'FAIL' '执行超时（已 Kill）'; return }
    if ($r.Exit -ne 1) { Write-Result $Name 'FAIL' ('exit=' + $r.Exit + '（期望 1）'); return }
    if ($r.Text.IndexOf($Keyword) -ge 0) { Write-Result $Name 'PASS' ('exit=1 且含提示：' + $Keyword) }
    else { Write-Result $Name 'FAIL' ('exit=1 但缺少提示：' + $Keyword) }
}

# 版本 golden（阶段4.2）：--version 退出码=0 且输出含程序版本号与模型格式版本。
function Assert-Version {
    $r = Invoke-Exe '--version'
    if ($r.TimedOut) { Write-Result 'version' 'FAIL' '执行超时（已 Kill）'; return }
    if ($r.Exit -ne 0) { Write-Result 'version' 'FAIL' ('exit=' + $r.Exit + '（期望 0）'); return }
    $missing = @()
    foreach ($k in @('1.0.0', 'ver=2')) { if ($r.Text.IndexOf($k) -lt 0) { $missing += $k } }
    if ($missing.Count -eq 0) { Write-Result 'version' 'PASS' '退出 0 且含 1.0.0 / ver=2' }
    else { Write-Result 'version' 'FAIL' ('缺少 [' + ($missing -join '; ') + ']') }
}

# 训练冒烟：只验证能跑完、loss 有限、无崩溃（不做数值 golden）
function Assert-Smoke {
    $smokeModel = Join-Path $fx '_smoke_model.txt'
    Remove-Item -LiteralPath $smokeModel -Force -ErrorAction SilentlyContinue
    $r = Invoke-Exe 'trainingDigits bkc --epochs 1 --cpu --seed 12345 --model "..\..\tests\fixtures\_smoke_model.txt"' -TimeoutSec 900
    if ($r.TimedOut) { Write-Result 'smoke' 'FAIL' '执行超时（已 Kill）'; return }
    if ($r.Exit -ne 0) { Write-Result 'smoke' 'FAIL' ('exit=' + $r.Exit); return }
    $m = [regex]::Match($r.Text, 'loss\s+([0-9eE+\-.]+)')
    if (-not $m.Success) { Write-Result 'smoke' 'FAIL' '未找到 loss 输出'; return }
    $lv = 0.0
    if (-not [double]::TryParse($m.Groups[1].Value, [ref]$lv)) { Write-Result 'smoke' 'FAIL' ('loss 解析失败: ' + $m.Groups[1].Value); return }
    if ([double]::IsNaN($lv) -or [double]::IsInfinity($lv)) { Write-Result 'smoke' 'FAIL' ('loss 非有限: ' + $m.Groups[1].Value); return }
    Write-Result 'smoke' 'PASS' ('loss=' + $m.Groups[1].Value)
}

# 残差网络训练冒烟（阶段3）：从固定残差夹具出发跑 1 epoch CPU 训练，
# 验证残差前向/反向（梯度分流）能跑通、loss 有限、无 NaN、无崩溃。
function Assert-SmokeRes {
    Copy-Item -LiteralPath (Join-Path $fx 'model_res.txt') -Destination (Join-Path $fx '_smoke_res.txt') -Force
    $r = Invoke-Exe 'trainingDigits bkc --epochs 1 --cpu --seed 12345 --model "..\..\tests\fixtures\_smoke_res.txt"' -TimeoutSec 900
    if ($r.TimedOut) { Write-Result 'smoke_res' 'FAIL' '执行超时（已 Kill）'; return }
    if ($r.Exit -ne 0) { Write-Result 'smoke_res' 'FAIL' ('exit=' + $r.Exit); return }
    $m = [regex]::Match($r.Text, 'loss\s+([0-9eE+\-.]+)')
    if (-not $m.Success) { Write-Result 'smoke_res' 'FAIL' '未找到 loss 输出'; return }
    $lv = 0.0
    if (-not [double]::TryParse($m.Groups[1].Value, [ref]$lv)) { Write-Result 'smoke_res' 'FAIL' ('loss 解析失败: ' + $m.Groups[1].Value); return }
    if ([double]::IsNaN($lv) -or [double]::IsInfinity($lv)) { Write-Result 'smoke_res' 'FAIL' ('loss 非有限: ' + $m.Groups[1].Value); return }
    Write-Result 'smoke_res' 'PASS' ('loss=' + $m.Groups[1].Value)
}

# 梯度校验（阶段3 严查）：运行 --gradcheck（自建小网络 + 固定种子 + 中心差分对拍），
# 一次运行登记 3 条结果：总 PASS、残差网 PASS、BN 网 PASS。
function Assert-GradCheck {
    $r = Invoke-Exe '--gradcheck' -TimeoutSec 600
    if ($r.TimedOut) {
        Write-Result 'gradcheck'     'FAIL' '执行超时（已 Kill）'
        Write-Result 'gradcheck_res' 'FAIL' '执行超时（已 Kill）'
        Write-Result 'gradcheck_bn'  'FAIL' '执行超时（已 Kill）'
        Write-Result 'gradcheck_proj'   'FAIL' '执行超时（已 Kill）'
        Write-Result 'gradcheck_preact' 'FAIL' '执行超时（已 Kill）'
        return
    }
    if ($r.Exit -ne 0) {
        Write-Result 'gradcheck'     'FAIL' ('exit=' + $r.Exit)
        Write-Result 'gradcheck_res' 'FAIL' ('exit=' + $r.Exit)
        Write-Result 'gradcheck_bn'  'FAIL' ('exit=' + $r.Exit)
        Write-Result 'gradcheck_proj'   'FAIL' ('exit=' + $r.Exit)
        Write-Result 'gradcheck_preact' 'FAIL' ('exit=' + $r.Exit)
        return
    }
    if ($r.Text.IndexOf('[GRADCHECK] SUMMARY PASS') -ge 0) {
        Write-Result 'gradcheck' 'PASS' 'SUMMARY PASS'
    } else {
        Write-Result 'gradcheck' 'FAIL' '缺少 [GRADCHECK] SUMMARY PASS'
    }
    if ([regex]::IsMatch($r.Text, '\[GRADCHECK\]\s+depth3res\s+maxRelErr=[\d.eE+-]+\s+PASS')) {
        $m = [regex]::Match($r.Text, '\[GRADCHECK\]\s+depth3res\s+conv_w\s+maxRelErr=([\d.eE+-]+)')
        Write-Result 'gradcheck_res' 'PASS' ('残差 conv_w maxRelErr=' + $m.Groups[1].Value)
    } else {
        Write-Result 'gradcheck_res' 'FAIL' 'depth3res 未 PASS'
    }
    if ([regex]::IsMatch($r.Text, '\[GRADCHECK\]\s+depth2bn\s+maxRelErr=[\d.eE+-]+\s+PASS')) {
        $m = [regex]::Match($r.Text, '\[GRADCHECK\]\s+depth2bn\s+bn_gamma\s+maxRelErr=([\d.eE+-]+)')
        Write-Result 'gradcheck_bn' 'PASS' ('BN gamma maxRelErr=' + $m.Groups[1].Value)
    } else {
        Write-Result 'gradcheck_bn' 'FAIL' 'depth2bn 未 PASS'
    }
    # 阶段6：1×1 投影残差（proj_w/proj_b）梯度校验
    if ([regex]::IsMatch($r.Text, '\[GRADCHECK\]\s+projres\s+maxRelErr=[\d.eE+-]+\s+PASS')) {
        $m = [regex]::Match($r.Text, '\[GRADCHECK\]\s+projres\s+proj_w\s+maxRelErr=([\d.eE+-]+)')
        Write-Result 'gradcheck_proj' 'PASS' ('投影 proj_w maxRelErr=' + $m.Groups[1].Value)
    } else {
        Write-Result 'gradcheck_proj' 'FAIL' 'projres 未 PASS'
    }
    # 阶段6：pre-activation 残差（resPos=1）梯度校验
    if ([regex]::IsMatch($r.Text, '\[GRADCHECK\]\s+preactres\s+maxRelErr=[\d.eE+-]+\s+PASS')) {
        $m = [regex]::Match($r.Text, '\[GRADCHECK\]\s+preactres\s+conv_w\s+maxRelErr=([\d.eE+-]+)')
        Write-Result 'gradcheck_preact' 'PASS' ('preact conv_w maxRelErr=' + $m.Groups[1].Value)
    } else {
        Write-Result 'gradcheck_preact' 'FAIL' 'preactres 未 PASS'
    }
}

# ===================== T8：acm 二进制格式专项回归 =====================
# 既有回归靠脚本最前的 CNN2_TXTMODEL=1 走 txt；本组用例由 Invoke-AcmTests 临时清除
# 该环境变量，显式走「默认 acm」路径（配合 -WorkDir 在临时子目录内生成/读取，用后清理）：
#   * acm_default        ：无 --model / 无开关 → 默认名 model.acm，read 识别为 acm；
#   * acm_suffix_acm/txt ：后缀分流 .acm→acm / .txt→txt（read 校验格式）；
#   * acm_suffix_conflict：.acm + --txtmodel → 报错退出（断言冲突提示）；
#   * acm_txt_equiv      ：同 seed 的 acm 与 txt，read 结构行完全一致；
#   * acm_size           ：文件大小 == 1024 + total*4；
#   * acm_precision/_load：篡改头 precision≠fp32 → read / test 加载均报「暂不支持」；
#   * acm_read_smoke     ：read <acm> 与 read <txt> 均 exit 0；
#   * acm_test_golden    ：同 seed acm/txt，test --cpu 数值（correct/total）完全一致（R7）；
#   * acm_best_suffix    ：--val 产出 <stem>.best.acm（.best 后缀随格式，R7）；
#   * acm_gen_txtmodel   ：gen --txtmodel（无 --model）产出 model.txt 而非 model.acm（R7）。

# 从 read 输出中提取「网络 / 逐层 / FC」结构行，供 acm↔txt 结构比对（排除首行文件大小差异）。
function Get-ReadStructLines {
    param([string]$Text)
    return (($Text -split "`r?`n") | Where-Object { $_ -match '^(网络:|  L[0-9]+:|FC:)' }) -join ' || '
}

# T8-1：默认 acm —— 无 --model、无 --txtmodel、无 env 时 gen 产出默认名 model.acm。
function Assert-AcmDefault {
    $dir = Join-Path $tmp 'acm_default'
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $r = Invoke-Exe 'gen --seed 1' -WorkDir $dir
    if ($r.TimedOut) { Write-Result 'acm_default' 'FAIL' '执行超时（已 Kill）'; return }
    if ($r.Exit -ne 0) { Write-Result 'acm_default' 'FAIL' ('gen exit=' + $r.Exit); return }
    $def = Join-Path $dir 'model.acm'
    if (-not (Test-Path -LiteralPath $def)) { Write-Result 'acm_default' 'FAIL' '未按默认名产出 model.acm'; return }
    $rr = Invoke-Exe ('read "' + $def + '"') -WorkDir $dir
    if ($rr.TimedOut) { Write-Result 'acm_default' 'FAIL' 'read 超时（已 Kill）'; return }
    if ($rr.Exit -ne 0) { Write-Result 'acm_default' 'FAIL' ('read exit=' + $rr.Exit); return }
    if ($rr.Text.IndexOf('格式: acm') -ge 0) { Write-Result 'acm_default' 'PASS' '无 --model 产出 model.acm，read 识别为 acm(二进制)' }
    else { Write-Result 'acm_default' 'FAIL' 'read 未识别为 acm(二进制)' }
}

# T8-2：后缀分流 —— .acm→acm / .txt→txt / .acm+--txtmodel→冲突报错。
function Assert-AcmSuffix {
    $dir = Join-Path $tmp 'acm_suffix'
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $acm = Join-Path $dir 'sfx.acm'
    $txt = Join-Path $dir 'sfx.txt'
    $con = Join-Path $dir 'cfx.acm'

    # (a) --model x.acm → acm（二进制）
    $r1 = Invoke-Exe ('gen --seed 2 --model "' + $acm + '"') -WorkDir $dir
    if ($r1.TimedOut) { Write-Result 'acm_suffix_acm' 'FAIL' '执行超时（已 Kill）' }
    elseif ($r1.Exit -ne 0) { Write-Result 'acm_suffix_acm' 'FAIL' ('gen exit=' + $r1.Exit) }
    else {
        $rd = Invoke-Exe ('read "' + $acm + '"') -WorkDir $dir
        if ($rd.Exit -eq 0 -and $rd.Text.IndexOf('格式: acm') -ge 0) { Write-Result 'acm_suffix_acm' 'PASS' '--model .acm → acm(二进制)' }
        else { Write-Result 'acm_suffix_acm' 'FAIL' ('read exit=' + $rd.Exit + '，未识别为 acm') }
    }

    # (b) --model x.txt → txt（文本）
    $r2 = Invoke-Exe ('gen --seed 2 --model "' + $txt + '"') -WorkDir $dir
    if ($r2.TimedOut) { Write-Result 'acm_suffix_txt' 'FAIL' '执行超时（已 Kill）' }
    elseif ($r2.Exit -ne 0) { Write-Result 'acm_suffix_txt' 'FAIL' ('gen exit=' + $r2.Exit) }
    else {
        $rd = Invoke-Exe ('read "' + $txt + '"') -WorkDir $dir
        if ($rd.Exit -eq 0 -and $rd.Text.IndexOf('格式: txt') -ge 0) { Write-Result 'acm_suffix_txt' 'PASS' '--model .txt → txt(文本)' }
        else { Write-Result 'acm_suffix_txt' 'FAIL' ('read exit=' + $rd.Exit + '，未识别为 txt') }
    }

    # (c) .acm + --txtmodel → 冲突，exit=1
    $r3 = Invoke-Exe ('gen --seed 2 --model "' + $con + '" --txtmodel') -WorkDir $dir
    if ($r3.TimedOut) { Write-Result 'acm_suffix_conflict' 'FAIL' '执行超时（已 Kill）' }
    elseif ($r3.Exit -ne 1) { Write-Result 'acm_suffix_conflict' 'FAIL' ('exit=' + $r3.Exit + '（期望 1）') }
    elseif ($r3.Text.IndexOf('冲突') -ge 0) { Write-Result 'acm_suffix_conflict' 'PASS' '.acm + --txtmodel → exit=1 且提示冲突' }
    else { Write-Result 'acm_suffix_conflict' 'FAIL' 'exit=1 但缺少冲突提示' }
}

# T8-3：acm↔txt 等价 —— 同 seed 生成，read 的「网络/逐层/FC」结构行完全一致。
function Assert-AcmTxtEquiv {
    $dir = Join-Path $tmp 'acm_equiv'
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $acm = Join-Path $dir 'eq.acm'
    $txt = Join-Path $dir 'eq.txt'
    $r1 = Invoke-Exe ('gen --seed 7 --model "' + $acm + '"') -WorkDir $dir
    $r2 = Invoke-Exe ('gen --seed 7 --model "' + $txt + '"') -WorkDir $dir
    if ($r1.TimedOut -or $r2.TimedOut) { Write-Result 'acm_txt_equiv' 'FAIL' '执行超时（已 Kill）'; return }
    if ($r1.Exit -ne 0 -or $r2.Exit -ne 0) { Write-Result 'acm_txt_equiv' 'FAIL' ('gen exit=' + $r1.Exit + '/' + $r2.Exit); return }
    $d1 = Invoke-Exe ('read "' + $acm + '"') -WorkDir $dir
    $d2 = Invoke-Exe ('read "' + $txt + '"') -WorkDir $dir
    if ($d1.Exit -ne 0 -or $d2.Exit -ne 0) { Write-Result 'acm_txt_equiv' 'FAIL' ('read exit=' + $d1.Exit + '/' + $d2.Exit); return }
    $s1 = Get-ReadStructLines $d1.Text
    $s2 = Get-ReadStructLines $d2.Text
    if ($s1 -and ($s1 -eq $s2)) { Write-Result 'acm_txt_equiv' 'PASS' ('同 seed acm/txt 结构一致: ' + $s1) }
    else { Write-Result 'acm_txt_equiv' 'FAIL' ('结构不一致: acm=[' + $s1 + '] txt=[' + $s2 + ']') }
}

# T8-4：大小锚点 —— acm 文件大小 == 1024 + total*4（total 由 read 输出解析）。
function Assert-AcmSize {
    $dir = Join-Path $tmp 'acm_size'
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $acm = Join-Path $dir 'sz.acm'
    $r1 = Invoke-Exe ('gen --seed 4 --model "' + $acm + '"') -WorkDir $dir
    if ($r1.TimedOut) { Write-Result 'acm_size' 'FAIL' '执行超时（已 Kill）'; return }
    if ($r1.Exit -ne 0) { Write-Result 'acm_size' 'FAIL' ('gen exit=' + $r1.Exit); return }
    $rd = Invoke-Exe ('read "' + $acm + '"') -WorkDir $dir
    $m = [regex]::Match($rd.Text, '参数量 total=(\d+)')
    if (-not $m.Success) { Write-Result 'acm_size' 'FAIL' 'read 未输出「参数量 total=」'; return }
    $total = [int]$m.Groups[1].Value
    $expect = 1024 + $total * 4
    $actual = (Get-Item -LiteralPath $acm).Length
    if ($actual -eq $expect) { Write-Result 'acm_size' 'PASS' ('大小 ' + $actual + ' = 1024 + total(' + $total + ')*4') }
    else { Write-Result 'acm_size' 'FAIL' ('大小 ' + $actual + ' ≠ 1024 + ' + $total + '*4 = ' + $expect) }
}

# T8-5：精度占位 —— 篡改 acm 头 precision（偏移 7）为 1，read 与 test 加载均应报「暂不支持」。
function Assert-AcmPrecision {
    $dir = Join-Path $tmp 'acm_prec'
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $good = Join-Path $dir 'p.acm'
    $bad  = Join-Path $dir 'p_bad.acm'
    $r1 = Invoke-Exe ('gen --seed 3 --model "' + $good + '"') -WorkDir $dir
    if ($r1.TimedOut) { Write-Result 'acm_precision' 'FAIL' '执行超时（已 Kill）'; Write-Result 'acm_precision_load' 'FAIL' '未运行'; return }
    if ($r1.Exit -ne 0) { Write-Result 'acm_precision' 'FAIL' ('gen exit=' + $r1.Exit); Write-Result 'acm_precision_load' 'FAIL' '未运行'; return }
    # 篡改偏移 7（precision）为 1（非 fp32），其余字节不变
    $b = [IO.File]::ReadAllBytes($good)
    $b[7] = 1
    [IO.File]::WriteAllBytes($bad, $b)

    $rd = Invoke-Exe ('read "' + $bad + '"') -WorkDir $dir
    if ($rd.Exit -eq 1 -and $rd.Text.IndexOf('暂不支持') -ge 0) { Write-Result 'acm_precision' 'PASS' 'read 对 precision=1 报「暂不支持」' }
    else { Write-Result 'acm_precision' 'FAIL' ('read exit=' + $rd.Exit + '，输出：' + ($rd.Text.Trim())) }

    # load 路径（test 加载）与 read 共用 decodeAcmHeader，应同样报「暂不支持」
    $tl = Invoke-Exe ('test "' + (Join-Path $rel 'testDigits') + '" --cpu --model "' + $bad + '"') -WorkDir $dir
    if ($tl.TimedOut) { Write-Result 'acm_precision_load' 'FAIL' '执行超时（已 Kill）' }
    elseif ($tl.Exit -eq 1 -and $tl.Text.IndexOf('暂不支持') -ge 0) { Write-Result 'acm_precision_load' 'PASS' 'test 加载 precision=1 报「暂不支持」' }
    else { Write-Result 'acm_precision_load' 'FAIL' ('exit=' + $tl.Exit + '，输出：' + ($tl.Text.Trim())) }
}

# T8-6：read 冒烟 —— read <acm> 与 read <txt> 均 exit 0。
function Assert-AcmReadSmoke {
    $dir = Join-Path $tmp 'acm_read'
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $acm = Join-Path $dir 'r.acm'
    $txt = Join-Path $dir 'r.txt'
    $r1 = Invoke-Exe ('gen --seed 6 --model "' + $acm + '"') -WorkDir $dir
    $r2 = Invoke-Exe ('gen --seed 6 --model "' + $txt + '"') -WorkDir $dir
    if ($r1.TimedOut -or $r2.TimedOut) { Write-Result 'acm_read_smoke' 'FAIL' '执行超时（已 Kill）'; return }
    if ($r1.Exit -ne 0 -or $r2.Exit -ne 0) { Write-Result 'acm_read_smoke' 'FAIL' ('gen exit=' + $r1.Exit + '/' + $r2.Exit); return }
    $a = Invoke-Exe ('read "' + $acm + '"') -WorkDir $dir
    $t = Invoke-Exe ('read "' + $txt + '"') -WorkDir $dir
    if ($a.TimedOut -or $t.TimedOut) { Write-Result 'acm_read_smoke' 'FAIL' '执行超时（已 Kill）'; return }
    if ($a.Exit -eq 0 -and $t.Exit -eq 0) { Write-Result 'acm_read_smoke' 'PASS' 'read <acm> / read <txt> 均 exit=0' }
    else { Write-Result 'acm_read_smoke' 'FAIL' ('read exit=' + $a.Exit + '(acm)/' + $t.Exit + '(txt)') }
}

# T8-7：acm 正向端到端数值 golden —— 同 seed 生成 acm 与 txt，test --cpu 的
#        「标准准确率 (correct/total)」须完全一致（证明 loadModelAcm→scatterParams→
#        推理成功路径与 txt 等价；acm 载入权重未错位）。
function Assert-AcmTestGolden {
    $dir = Join-Path $tmp 'acm_test'
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $acm = Join-Path $dir 'tg.acm'
    $txt = Join-Path $dir 'tg.txt'
    $ra = Invoke-Exe ('gen --seed 11 --model "' + $acm + '"') -WorkDir $dir
    $rt = Invoke-Exe ('gen --seed 11 --model "' + $txt + '"') -WorkDir $dir
    if ($ra.TimedOut -or $rt.TimedOut) { Write-Result 'acm_test_golden' 'FAIL' 'gen 超时（已 Kill）'; return }
    if ($ra.Exit -ne 0 -or $rt.Exit -ne 0) { Write-Result 'acm_test_golden' 'FAIL' ('gen exit=' + $ra.Exit + '/' + $rt.Exit); return }
    # 由 $rel（x64\Release）运行，模型用绝对路径；..\testDigits 与既有回归一致。
    $ta = Invoke-Exe ('..\testDigits test --cpu --prep_target 20 --model "' + $acm + '"') -TimeoutSec 900
    $tt = Invoke-Exe ('..\testDigits test --cpu --prep_target 20 --model "' + $txt + '"') -TimeoutSec 900
    if ($ta.TimedOut -or $tt.TimedOut) { Write-Result 'acm_test_golden' 'FAIL' 'test 超时（已 Kill）'; return }
    if ($ta.Exit -ne 0) { Write-Result 'acm_test_golden' 'FAIL' ('test <acm> exit=' + $ta.Exit); return }
    if ($tt.Exit -ne 0) { Write-Result 'acm_test_golden' 'FAIL' ('test <txt> exit=' + $tt.Exit); return }
    $ma = [regex]::Match($ta.Text, '标准准确率:\s*([0-9.]+)\s*\((\d+)/(\d+)\)')
    $mt = [regex]::Match($tt.Text, '标准准确率:\s*([0-9.]+)\s*\((\d+)/(\d+)\)')
    if (-not $ma.Success -or -not $mt.Success) { Write-Result 'acm_test_golden' 'FAIL' '未找到“标准准确率 (correct/total)”'; return }
    $ca = [int]$ma.Groups[2].Value; $na = [int]$ma.Groups[3].Value
    $ct = [int]$mt.Groups[2].Value; $nt = [int]$mt.Groups[3].Value
    if ($ca -eq $ct -and $na -eq $nt) { Write-Result 'acm_test_golden' 'PASS' ('acm/txt 数值一致: ' + $ca + '/' + $na) }
    else { Write-Result 'acm_test_golden' 'FAIL' ('acm=' + $ca + '/' + $na + ' ≠ txt=' + $ct + '/' + $nt) }
}

# T8-8：--val 的 .best 后缀随格式 —— acm 模型（.acm）+ --val 须产出 <stem>.best.acm。
function Assert-AcmBestSuffix {
    $base = Join-Path $tmp 'acm_best'
    $tr   = Join-Path $base 'train'
    $va   = Join-Path $base 'val'
    $mdl  = Join-Path $base 'best_model.acm'
    Remove-Item -LiteralPath $base -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $tr, $va | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $rel 'trainingDigits') -File -Filter '*.txt' |
        Group-Object { $_.Name.Substring(0, 1) } | ForEach-Object {
            $_.Group | Select-Object -First 2 | ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $tr $_.Name) -Force
            }
            $_.Group | Select-Object -Skip 2 -First 1 | ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $va $_.Name) -Force
            }
        }
    $r = Invoke-Exe ('"' + $tr + '" bkc --cpu --epochs 1 --seed 1 --val "' + $va + '" --model "' + $mdl + '"') -TimeoutSec 900
    if ($r.TimedOut) { Write-Result 'acm_best_suffix' 'FAIL' '执行超时（已 Kill）'; return }
    if ($r.Exit -ne 0) { Write-Result 'acm_best_suffix' 'FAIL' ('exit=' + $r.Exit); return }
    $best = Join-Path $base 'best_model.best.acm'
    if (-not (Test-Path -LiteralPath $best)) { Write-Result 'acm_best_suffix' 'FAIL' '未产出 <stem>.best.acm'; return }
    $rd = Invoke-Exe ('read "' + $best + '"') -WorkDir $base
    if ($rd.Exit -eq 0 -and $rd.Text.IndexOf('格式: acm') -ge 0) { Write-Result 'acm_best_suffix' 'PASS' '--val 产出 best_model.best.acm 且 read 识别为 acm' }
    else { Write-Result 'acm_best_suffix' 'FAIL' ('read exit=' + $rd.Exit + '，未识别为 acm') }
}

# T8-9：gen --txtmodel（无 --model）—— 默认保存名切回 model.txt（对照默认 acm），
#        且不该再产出 model.acm。
function Assert-AcmGenTxtmodel {
    $dir = Join-Path $tmp 'acm_gen_txt'
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $r = Invoke-Exe 'gen --seed 5 --txtmodel' -WorkDir $dir
    if ($r.TimedOut) { Write-Result 'acm_gen_txtmodel' 'FAIL' '执行超时（已 Kill）'; return }
    if ($r.Exit -ne 0) { Write-Result 'acm_gen_txtmodel' 'FAIL' ('gen exit=' + $r.Exit); return }
    $txt = Join-Path $dir 'model.txt'
    $acm = Join-Path $dir 'model.acm'
    if (-not (Test-Path -LiteralPath $txt)) { Write-Result 'acm_gen_txtmodel' 'FAIL' 'gen --txtmodel 未产出 model.txt'; return }
    if (Test-Path -LiteralPath $acm) { Write-Result 'acm_gen_txtmodel' 'FAIL' 'gen --txtmodel 仍产出 model.acm'; return }
    $rd = Invoke-Exe ('read "' + $txt + '"') -WorkDir $dir
    if ($rd.Exit -eq 0 -and $rd.Text.IndexOf('格式: txt') -ge 0) { Write-Result 'acm_gen_txtmodel' 'PASS' 'gen --txtmodel 产出 model.txt（read 识别为 txt）' }
    else { Write-Result 'acm_gen_txtmodel' 'FAIL' ('read exit=' + $rd.Exit + '，未识别为 txt') }
}

# 统一入口：临时清除 CNN2_TXTMODEL（走默认 acm），跑完全部 acm 用例后恢复。
#   * 崩溃（异常）也会经 finally 恢复环境，避免污染后续。
function Invoke-AcmTests {
    $saved = $env:CNN2_TXTMODEL
    Remove-Item -Path 'Env:\CNN2_TXTMODEL' -ErrorAction SilentlyContinue
    try {
        Assert-AcmDefault
        Assert-AcmSuffix
        Assert-AcmTxtEquiv
        Assert-AcmSize
        Assert-AcmPrecision
        Assert-AcmReadSmoke
        Assert-AcmTestGolden
        Assert-AcmBestSuffix
        Assert-AcmGenTxtmodel
    } finally {
        if ($null -ne $saved) { $env:CNN2_TXTMODEL = $saved }
        else { Remove-Item -Path 'Env:\CNN2_TXTMODEL' -ErrorAction SilentlyContinue }
    }
}

# 计算文件 SHA256（阶段4.1 逐字节一致性比对用）。
function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

# 可复现性 golden（阶段4.1）：
#   * gen --seed 1 两次 → 生成文件 SHA256 相同（同 seed 可复现）；
#   * gen --seed 1 与 --seed 2 → SHA256 不同（证明确受 seed 控制）。
function Assert-ReproGen {
    $g1 = Join-Path $fx '_repro_gen1.txt'
    $g2 = Join-Path $fx '_repro_gen2.txt'
    $g3 = Join-Path $fx '_repro_gen3.txt'
    Remove-Item -LiteralPath $g1, $g2, $g3 -Force -ErrorAction SilentlyContinue
    $r1 = Invoke-Exe ('..\testDigits gen --seed 1 --model "..\..\tests\fixtures\_repro_gen1.txt"')
    $r2 = Invoke-Exe ('..\testDigits gen --seed 1 --model "..\..\tests\fixtures\_repro_gen2.txt"')
    $r3 = Invoke-Exe ('..\testDigits gen --seed 2 --model "..\..\tests\fixtures\_repro_gen3.txt"')
    foreach ($r in @($r1, $r2, $r3)) {
        if ($r.TimedOut) { Write-Result 'repro_gen_same' 'FAIL' '执行超时（已 Kill）'; Write-Result 'repro_gen_diff' 'FAIL' '执行超时（已 Kill）'; return }
    }
    if ($r1.Exit -ne 0 -or $r2.Exit -ne 0 -or $r3.Exit -ne 0) {
        $d = 'gen exit=' + $r1.Exit + '/' + $r2.Exit + '/' + $r3.Exit
        Write-Result 'repro_gen_same' 'FAIL' $d; Write-Result 'repro_gen_diff' 'FAIL' $d; return
    }
    if (-not (Test-Path -LiteralPath $g1) -or -not (Test-Path -LiteralPath $g2) -or -not (Test-Path -LiteralPath $g3)) {
        Write-Result 'repro_gen_same' 'FAIL' '未生成模型文件'; Write-Result 'repro_gen_diff' 'FAIL' '未生成模型文件'; return
    }
    $h1 = Get-Sha256 $g1; $h2 = Get-Sha256 $g2; $h3 = Get-Sha256 $g3
    if ($h1 -eq $h2) { Write-Result 'repro_gen_same' 'PASS' ('seed=1 两次 SHA256 相同 ' + $h1.Substring(0, 12)) }
    else { Write-Result 'repro_gen_same' 'FAIL' ('同 seed 不一致: ' + $h1 + ' vs ' + $h2) }
    if ($h1 -ne $h3) { Write-Result 'repro_gen_diff' 'PASS' ('seed 1/2 不同 ' + $h1.Substring(0, 12) + ' vs ' + $h3.Substring(0, 12)) }
    else { Write-Result 'repro_gen_diff' 'FAIL' '不同 seed 结果相同（seed 未生效）' }
}

# bkc 训练可复现：--seed 1 --epochs 1 --cpu 两次（均从随机初始化开始）→ 模型 SHA256 相同。
function Assert-ReproBkc {
    $b1 = Join-Path $fx '_repro_bkc1.txt'
    $b2 = Join-Path $fx '_repro_bkc2.txt'
    Remove-Item -LiteralPath $b1, $b2 -Force -ErrorAction SilentlyContinue
    $r1 = Invoke-Exe 'trainingDigits bkc --epochs 1 --cpu --seed 1 --model "..\..\tests\fixtures\_repro_bkc1.txt"' -TimeoutSec 900
    $r2 = Invoke-Exe 'trainingDigits bkc --epochs 1 --cpu --seed 1 --model "..\..\tests\fixtures\_repro_bkc2.txt"' -TimeoutSec 900
    foreach ($r in @($r1, $r2)) {
        if ($r.TimedOut) { Write-Result 'repro_bkc' 'FAIL' '执行超时（已 Kill）'; return }
    }
    if ($r1.Exit -ne 0 -or $r2.Exit -ne 0) {
        Write-Result 'repro_bkc' 'FAIL' ('bkc exit=' + $r1.Exit + '/' + $r2.Exit); return
    }
    if (-not (Test-Path -LiteralPath $b1) -or -not (Test-Path -LiteralPath $b2)) {
        Write-Result 'repro_bkc' 'FAIL' '未生成模型文件'; return
    }
    $h1 = Get-Sha256 $b1; $h2 = Get-Sha256 $b2
    if ($h1 -eq $h2) { Write-Result 'repro_bkc' 'PASS' ('同 seed 两次 SHA256 相同 ' + $h1.Substring(0, 12)) }
    else { Write-Result 'repro_bkc' 'FAIL' ('同 seed 不一致: ' + $h1 + ' vs ' + $h2) }
}

# L2 weight decay 位保护 golden（阶段7）：
#   同 seed、同配置 CPU 训练 1 epoch，一次不带 --wd、一次带 --wd 0，
#   断言两模型文件 SHA256 相同（wd==0 不进叠加路径，训练逐位不变）。
function Assert-WdBitProtect {
    $m0 = Join-Path $tmp 'wd_bp0.txt'
    $m1 = Join-Path $tmp 'wd_bp1.txt'
    Remove-Item -LiteralPath $m0, $m1 -Force -ErrorAction SilentlyContinue
    $r0 = Invoke-Exe ('trainingDigits bkc --epochs 1 --cpu --seed 1 --model "' + $m0 + '"') -TimeoutSec 900
    $r1 = Invoke-Exe ('trainingDigits bkc --epochs 1 --cpu --seed 1 --wd 0 --model "' + $m1 + '"') -TimeoutSec 900
    foreach ($r in @($r0, $r1)) {
        if ($r.TimedOut) { Write-Result 'wd_bp' 'FAIL' '执行超时（已 Kill）'; return }
    }
    if ($r0.Exit -ne 0 -or $r1.Exit -ne 0) {
        Write-Result 'wd_bp' 'FAIL' ('bkc exit=' + $r0.Exit + '/' + $r1.Exit); return
    }
    if (-not (Test-Path -LiteralPath $m0) -or -not (Test-Path -LiteralPath $m1)) {
        Write-Result 'wd_bp' 'FAIL' '未生成模型文件'; return
    }
    $h0 = Get-Sha256 $m0; $h1 = Get-Sha256 $m1
    if ($h0 -eq $h1) { Write-Result 'wd_bp' 'PASS' ('--wd 0 与缺省 SHA256 相同 ' + $h0.Substring(0, 12)) }
    else { Write-Result 'wd_bp' 'FAIL' ('--wd 0 改变了训练结果: ' + $h0 + ' vs ' + $h1) }
}

# L2 weight decay 生效 golden（阶段7）：
#   CPU 训练 1 epoch 加 --wd 0.01，断言 exit=0 且 loss 有限（非 NaN/inf）；
#   并断言其模型 SHA256 与 --wd 0 的不同（证明 weight decay 确实参与叠加）。
function Assert-WdActive {
    $m0 = Join-Path $tmp 'wd_zero.txt'
    $m1 = Join-Path $tmp 'wd_decay.txt'
    Remove-Item -LiteralPath $m0, $m1 -Force -ErrorAction SilentlyContinue
    $r0 = Invoke-Exe ('trainingDigits bkc --epochs 1 --cpu --seed 1 --wd 0 --model "' + $m0 + '"') -TimeoutSec 900
    $r1 = Invoke-Exe ('trainingDigits bkc --epochs 1 --cpu --seed 1 --wd 0.01 --model "' + $m1 + '"') -TimeoutSec 900
    foreach ($r in @($r0, $r1)) {
        if ($r.TimedOut) { Write-Result 'wd_active' 'FAIL' '执行超时（已 Kill）'; Write-Result 'wd_active_diff' 'FAIL' '执行超时（已 Kill）'; return }
    }
    if ($r1.Exit -ne 0) {
        Write-Result 'wd_active' 'FAIL' ('exit=' + $r1.Exit); Write-Result 'wd_active_diff' 'FAIL' '未运行'; return
    }
    $m = [regex]::Match($r1.Text, 'loss\s+([0-9eE+\-.]+)')
    if (-not $m.Success) { Write-Result 'wd_active' 'FAIL' '未找到 loss 输出'; Write-Result 'wd_active_diff' 'FAIL' '未运行'; return }
    $lv = 0.0
    if (-not [double]::TryParse($m.Groups[1].Value, [ref]$lv) -or [double]::IsNaN($lv) -or [double]::IsInfinity($lv)) {
        Write-Result 'wd_active' 'FAIL' ('loss 非有限: ' + $m.Groups[1].Value); Write-Result 'wd_active_diff' 'FAIL' '未运行'; return
    }
    Write-Result 'wd_active' 'PASS' ('exit=0 loss=' + $m.Groups[1].Value)
    if ($r0.Exit -ne 0) { Write-Result 'wd_active_diff' 'FAIL' ('--wd 0 基线 exit=' + $r0.Exit); return }
    if (-not (Test-Path -LiteralPath $m0) -or -not (Test-Path -LiteralPath $m1)) {
        Write-Result 'wd_active_diff' 'FAIL' '未生成模型文件'; return
    }
    $h0 = Get-Sha256 $m0; $h1 = Get-Sha256 $m1
    if ($h0 -ne $h1) { Write-Result 'wd_active_diff' 'PASS' ('--wd 0.01 与 --wd 0 SHA256 不同 ' + $h0.Substring(0, 8) + ' vs ' + $h1.Substring(0, 8)) }
    else { Write-Result 'wd_active_diff' 'FAIL' '--wd 0.01 与 --wd 0 结果相同（wd 未生效）' }
}

# --epsave 冒烟（阶段7）：bkc --epochs 2 --cpu --seed 1 --epsave <临时目录>，
#   断言生成 ep_0.txt 与 ep_1.txt，且其中一个能被 test 正常加载（exit=0）。
function Assert-Epsave {
    $dir = Join-Path $tmp 'epsave'
    $mdl = Join-Path $tmp 'epsave_model.txt'
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $mdl -Force -ErrorAction SilentlyContinue
    $r = Invoke-Exe ('trainingDigits bkc --epochs 2 --cpu --seed 1 --epsave "' + $dir + '" --model "' + $mdl + '"') -TimeoutSec 900
    if ($r.TimedOut) { Write-Result 'epsave' 'FAIL' '执行超时（已 Kill）'; return }
    if ($r.Exit -ne 0) { Write-Result 'epsave' 'FAIL' ('exit=' + $r.Exit); return }
    $e0 = Join-Path $dir 'ep_0.txt'
    $e1 = Join-Path $dir 'ep_1.txt'
    if (-not (Test-Path -LiteralPath $e0) -or -not (Test-Path -LiteralPath $e1)) {
        Write-Result 'epsave' 'FAIL' ('缺少 ep_0.txt / ep_1.txt（dir=' + $dir + '）'); return
    }
    $rt = Invoke-Exe ('..\testDigits test --cpu --model "' + $e0 + '"') -TimeoutSec 900
    if ($rt.TimedOut) { Write-Result 'epsave' 'FAIL' '存档模型 test 加载超时（已 Kill）'; return }
    if ($rt.Exit -ne 0) { Write-Result 'epsave' 'FAIL' ('存档 ep_0.txt 无法被 test 加载: exit=' + $rt.Exit); return }
    Write-Result 'epsave' 'PASS' 'ep_0.txt + ep_1.txt 生成，ep_0 可被 test 加载'
}

# --epsn 冒烟（阶段7）：bkc --epochs 4 --cpu --seed 1 --epsave <临时目录> --epsn 2，
#   断言只生成 ep_0.txt、ep_2.txt（epoch 0/2 存档），无 ep_1/ep_3。
function Assert-Epsn {
    $dir = Join-Path $tmp 'epsn'
    $mdl = Join-Path $tmp 'epsn_model.txt'
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $mdl -Force -ErrorAction SilentlyContinue
    $r = Invoke-Exe ('trainingDigits bkc --epochs 4 --cpu --seed 1 --epsave "' + $dir + '" --epsn 2 --model "' + $mdl + '"') -TimeoutSec 900
    if ($r.TimedOut) { Write-Result 'epsn' 'FAIL' '执行超时（已 Kill）'; return }
    if ($r.Exit -ne 0) { Write-Result 'epsn' 'FAIL' ('exit=' + $r.Exit); return }
    $have = @(Get-ChildItem -LiteralPath $dir -File -Filter 'ep_*.txt' -ErrorAction SilentlyContinue |
              ForEach-Object { $_.Name })
    $missing = @()
    foreach ($want in @('ep_0.txt', 'ep_2.txt')) { if ($have -notcontains $want) { $missing += $want } }
    $extra = @()
    foreach ($bad in @('ep_1.txt', 'ep_3.txt')) { if ($have -contains $bad) { $extra += $bad } }
    if ($missing.Count -eq 0 -and $extra.Count -eq 0) {
        Write-Result 'epsn' 'PASS' ('仅生成 ' + (($have | Sort-Object) -join ', '))
    } else {
        Write-Result 'epsn' 'FAIL' ('缺少 [' + ($missing -join ';') + '] 多出 [' + ($extra -join ';') + '] 实得: ' + (($have | Sort-Object) -join ', '))
    }
}

# 训练日志 golden（阶段4.3 可观测性）：
#   小数据（每类取 2 张，共 20 张）跑 2 epoch CPU 训练 + --log，断言日志文件：
#     1) 存在；2) 首行 CSV 表头正确；3) 指标行数=2；4) 每指标行 7 个字段；
#     5) 至少 1 行 CHECKPOINT 元数据（模型↔指标↔时间可追溯）。
function Assert-TrainLog {
    $dataDir  = Join-Path $tmp 'logdata'
    $logFile  = Join-Path $tmp 'trainlog.csv'
    $logModel = Join-Path $tmp 'trainlog_model.txt'
    Remove-Item -LiteralPath $dataDir -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $logFile, $logModel -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
    # 从训练目录每类取前 2 张，构成 20 张的小数据集（快，适合 2 epoch 冒烟）
    Get-ChildItem -LiteralPath (Join-Path $rel 'trainingDigits') -File -Filter '*.txt' |
        Group-Object { $_.Name.Substring(0, 1) } | ForEach-Object {
            $_.Group | Select-Object -First 2 | ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $dataDir $_.Name) -Force
            }
        }
    # 用绝对路径 + 引号（工作目录固定为 x64\Release）
    $r = Invoke-Exe ('"' + $dataDir + '" bkc --cpu --epochs 2 --seed 1 --log "' + $logFile + '" --model "' + $logModel + '"') -TimeoutSec 900
    if ($r.TimedOut) { Write-Result 'train_log' 'FAIL' '执行超时（已 Kill）'; return }
    if ($r.Exit -ne 0) { Write-Result 'train_log' 'FAIL' ('exit=' + $r.Exit); return }
    if (-not (Test-Path -LiteralPath $logFile)) { Write-Result 'train_log' 'FAIL' '未生成 --log 文件'; return }
    $lines  = [IO.File]::ReadAllLines($logFile)
    $header = 'epoch,loss,acc,lr_conv1,lr_conv2,lr_fc,epoch_sec'
    if ($lines.Count -lt 1 -or $lines[0] -ne $header) {
        Write-Result 'train_log' 'FAIL' ('表头不符: ' + ($(if ($lines.Count) { $lines[0] } else { '(空文件)' }))); return
    }
    $metric = @($lines | Where-Object { $_ -match '^\d+,' })
    if ($metric.Count -ne 2) { Write-Result 'train_log' 'FAIL' ('指标行数=' + $metric.Count + '（期望 2）'); return }
    foreach ($ln in $metric) {
        if (($ln -split ',').Count -ne 7) { Write-Result 'train_log' 'FAIL' ('指标行字段数≠7: ' + $ln); return }
    }
    $ckpt = @($lines | Where-Object { $_ -match '^CHECKPOINT ' })
    if ($ckpt.Count -lt 1) { Write-Result 'train_log' 'FAIL' '缺少 CHECKPOINT 元数据行'; return }
    Write-Result 'train_log' 'PASS' ('表头 + 2 指标行(7 列) + CHECKPOINT ' + $ckpt.Count + ' 条')
}

# 独立验证集 golden（阶段4.4）：小数据（每类训练 2 张、验证 1 张）跑 1 epoch CPU 训练 + --val + --log。
# 断言：① stdout 含 `val acc`；② CSV 表头含 `val_acc` 列；③ 指标行 8 列且 val_acc ∈ [0,1]。
function Assert-Val {
    $base = Join-Path $tmp 'valdata'
    $tr   = Join-Path $base 'train'
    $va   = Join-Path $base 'val'
    $log  = Join-Path $base 'val_log.csv'
    $mdl  = Join-Path $base 'val_model.txt'
    Remove-Item -LiteralPath $base -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $tr, $va | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $rel 'trainingDigits') -File -Filter '*.txt' |
        Group-Object { $_.Name.Substring(0, 1) } | ForEach-Object {
            $_.Group | Select-Object -First 2 | ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $tr $_.Name) -Force
            }
            $_.Group | Select-Object -Skip 2 -First 1 | ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $va $_.Name) -Force
            }
        }
    $r = Invoke-Exe ('"' + $tr + '" bkc --cpu --epochs 1 --seed 1 --val "' + $va + '" --log "' + $log + '" --model "' + $mdl + '"') -TimeoutSec 900
    if ($r.TimedOut) { Write-Result 'val' 'FAIL' '执行超时（已 Kill）'; return }
    if ($r.Exit -ne 0) { Write-Result 'val' 'FAIL' ('exit=' + $r.Exit); return }
    if ($r.Text.IndexOf('val acc') -lt 0) { Write-Result 'val' 'FAIL' 'stdout 未包含 val acc'; return }
    if (-not (Test-Path -LiteralPath $log)) { Write-Result 'val' 'FAIL' '未生成 --log 文件'; return }
    $lines = [IO.File]::ReadAllLines($log)
    if ($lines.Count -lt 1 -or $lines[0].IndexOf('val_acc') -lt 0) {
        Write-Result 'val' 'FAIL' ('CSV 表头缺少 val_acc: ' + $(if ($lines.Count) { $lines[0] } else { '(空文件)' })); return
    }
    $metric = @($lines | Where-Object { $_ -match '^\d+,' })
    if ($metric.Count -lt 1) { Write-Result 'val' 'FAIL' '无指标行'; return }
    $cols = $metric[0] -split ','
    if ($cols.Count -ne 8) { Write-Result 'val' 'FAIL' ('指标行列数=' + $cols.Count + '（期望 8）'); return }
    $v = 0.0
    if (-not [double]::TryParse($cols[7], [ref]$v)) { Write-Result 'val' 'FAIL' ('val_acc 解析失败: ' + $cols[7]); return }
    if ($v -lt 0.0 -or $v -gt 1.0) { Write-Result 'val' 'FAIL' ('val_acc 越界: ' + $v); return }
    Write-Result 'val' 'PASS' ('val_acc=' + $v + '，表头含 val_acc')
}

# 数据校验 golden（阶段4.4）：构造 1 个合法 + 1 个非法（第 5 行含 'x'）文件。
#   * --checkdata：exit 0 且报告列出非法文件名；
#   * --checkdata --strict-data：遇非法即 exit=1。
function Assert-CheckData {
    $base = Join-Path $tmp 'checkdata'
    Remove-Item -LiteralPath $base -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $base | Out-Null
    $good = Get-ChildItem -LiteralPath (Join-Path $rel 'trainingDigits') -File -Filter '0_*.txt' | Select-Object -First 1
    Copy-Item -LiteralPath $good.FullName -Destination (Join-Path $base '0_ok.txt') -Force
    $bad  = Join-Path $base '9_bad.txt'
    $rows = @()
    for ($i = 0; $i -lt 32; ++$i) {
        $rows += $(if ($i -eq 4) { '01x1' + ('0' * 28) } else { '0' * 32 })
    }
    [IO.File]::WriteAllText($bad, ($rows -join "`r`n") + "`r`n")

    $r = Invoke-Exe ('"' + $base + '" --checkdata')
    if ($r.TimedOut) {
        Write-Result 'data_check'  'FAIL' '执行超时（已 Kill）'
        Write-Result 'data_strict' 'FAIL' '执行超时（已 Kill）'
        return
    }
    if ($r.Exit -ne 0) {
        Write-Result 'data_check'  'FAIL' ('exit=' + $r.Exit + '（期望 0）')
        Write-Result 'data_strict' 'FAIL' '未运行'
        return
    }
    if ($r.Text.IndexOf('9_bad.txt') -ge 0) { Write-Result 'data_check' 'PASS' '报告列出非法文件 9_bad.txt' }
    else { Write-Result 'data_check' 'FAIL' '报告未列出非法文件 9_bad.txt' }

    $rs = Invoke-Exe ('"' + $base + '" --checkdata --strict-data')
    if ($rs.TimedOut) { Write-Result 'data_strict' 'FAIL' '执行超时（已 Kill）'; return }
    if ($rs.Exit -eq 1) { Write-Result 'data_strict' 'PASS' 'strict 遇非法 exit=1' }
    else { Write-Result 'data_strict' 'FAIL' ('exit=' + $rs.Exit + '（期望 1）') }
}

# 夹具保障：model32 可由 Release\model.txt 复制；非法维度夹具由 model32 确定性派生；
#            model64 / model_bn 属于“生成一次后固定”的数值夹具，缺失则报错而非重建。
function Ensure-Fixtures {
    New-Item -ItemType Directory -Force -Path $fx | Out-Null
    $m32 = Join-Path $fx 'model32.txt'
    if (-not (Test-Path -LiteralPath $m32)) {
        Copy-Item -LiteralPath (Join-Path $rel 'model.txt') -Destination $m32 -Force
    }
    foreach ($n in @('model64.txt', 'model_bn.txt', 'model_d3.txt', 'model_res.txt',
                     'model_proj.txt', 'model_preact.txt')) {
        if (-not (Test-Path -LiteralPath (Join-Path $fx $n))) {
            throw ('缺少夹具 ' + $n + '：请用 gen 生成一次后固定（见 tests\golden.txt 顶部说明），回归中不得重建')
        }
    }
    $base = [IO.File]::ReadAllLines($m32, [Text.Encoding]::GetEncoding(936))
    $b20 = $base | ForEach-Object { if ($_ -match '^FC\s') { 'FC 20 1152' } else { $_ } }
    [IO.File]::WriteAllLines((Join-Path $fx 'model_bad_fc20.txt'), $b20, [Text.Encoding]::GetEncoding(936))
    $b999 = $base | ForEach-Object { if ($_ -match '^FC\s') { 'FC 10 999' } else { $_ } }
    [IO.File]::WriteAllLines((Join-Path $fx 'model_bad_fc999.txt'), $b999, [Text.Encoding]::GetEncoding(936))
}

function Invoke-Build {
    $mb = $msbuildDefault
    if (-not (Test-Path -LiteralPath $mb)) {
        $cmd = Get-Command MSBuild.exe -ErrorAction SilentlyContinue
        if ($cmd) { $mb = $cmd.Source } else { throw '找不到 MSBuild.exe（请检查 D:\vc++\MSBuild\Current\Bin\MSBuild.exe 或 PATH）' }
    }
    $env:CUDA_PATH_V12_9 = $cudaPath
    Write-Host ('编译 Release x64: ' + $sln) -ForegroundColor Cyan
    # 编译同样加超时保护（30 分钟），避免工具链挂起拖死脚本。
    # 用 .NET ProcessStartInfo 直接启动（Start-Process 在本环境拿不到 ExitCode）。
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $mb
    $psi.Arguments = '"' + $sln + '" /p:Configuration=Release /p:Platform=x64 /m /v:minimal /nologo'
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $false
    $proc = [System.Diagnostics.Process]::Start($psi)
    $timedOut = -not $proc.WaitForExit(1800000)
    if ($timedOut) {
        & taskkill.exe /F /T /PID $proc.Id 2>$null | Out-Null
        try { $proc.Kill() } catch { }
        try { $proc.WaitForExit() } catch { }
        throw 'MSBuild 超时（>30 分钟），已终止'
    }
    $code = $proc.ExitCode
    if ($code -ne 0) { throw ('MSBuild 失败 exit=' + $code) }
}

# ======================= 主流程 =======================
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

try {
    if (-not $SkipBuild) { Invoke-Build }
    if (-not (Test-Path -LiteralPath $exe)) { throw ('找不到 exe: ' + $exe) }
    if (-not $OnlyAcm) { Ensure-Fixtures }

    # T8：-OnlyAcm 时只跑 acm 专项块（跳过其余回归），便于快速验证新用例。
    if ($OnlyAcm) {
        Write-Host ''
        Write-Host '===== 8. acm 二进制格式专项（T8：默认 acm / 后缀分流 / 等价 / 精度占位）=====' -ForegroundColor Cyan
        Invoke-AcmTests
    } else {

    $p32 = '..\..\tests\fixtures\model32.txt'
    $p64 = '..\..\tests\fixtures\model64.txt'
    $pbn = '..\..\tests\fixtures\model_bn.txt'

    Write-Host ''
    Write-Host '===== 0. 版本 golden（--version：退出 0 + 关键子串）=====' -ForegroundColor Cyan
    Assert-Version

    Write-Host ''
    Write-Host '===== 1. 推理数值 golden（CPU/GPU）=====' -ForegroundColor Cyan
    Assert-Acc 'cpu32' ('..\testDigits test --cpu --prep_target 20 --model "' + $p32 + '"')                          1278 10000
    Assert-Acc 'gpu32' ('..\testDigits test --gpu --prep_target 20 --model "' + $p32 + '"')                          1278 10000 -Gpu
    Assert-Acc 'cpu64' ('..\testDigits test --cpu --prep_target 20 --model "' + $p64 + '"')                           309 10000
    Assert-Acc 'gpu64' ('..\testDigits test --gpu --prep_target 20 --model "' + $p64 + '"')                           309 10000 -Gpu
    Assert-Acc 'cpubn' ('..\testDigits test --cpu --BN --prep_target 20 --model "' + $pbn + '"')                       945 10000
    Assert-Acc 'gpubn' ('..\testDigits test --gpu --BN --prep_target 20 --model "' + $pbn + '"')                       945 10000 -Gpu

    Write-Host ''
    Write-Host '===== 2. 集成 / TTA golden（同模型两次，确定性）=====' -ForegroundColor Cyan
    Assert-Acc 'tta64' ('..\testDigits test --gpu --prep_target 20 --fuse logit --tta --tta_k 5 --model "' + $p64 + ',' + $p64 + '"') 303 10000 -Gpu

    Write-Host ''
    Write-Host '===== 2b. depth=3 数值 golden（固定夹具，CPU/GPU 交叉验证）=====' -ForegroundColor Cyan
    $pd3 = '..\..\tests\fixtures\model_d3.txt'
    Assert-Acc 'cpud3' ('..\testDigits test --cpu --prep_target 20 --model "' + $pd3 + '"') 892 10000
    Assert-Acc 'gpud3' ('..\testDigits test --gpu --prep_target 20 --model "' + $pd3 + '"') 892 10000 -Gpu

    Write-Host ''
    Write-Host '===== 2c. 残差网络数值 golden（固定夹具，CPU/GPU 交叉验证）=====' -ForegroundColor Cyan
    $pres = '..\..\tests\fixtures\model_res.txt'
    Assert-Acc 'cpu_res' ('..\testDigits test --cpu --prep_target 20 --model "' + $pres + '"') 984 10000
    Assert-Acc 'gpu_res' ('..\testDigits test --gpu --prep_target 20 --model "' + $pres + '"') 984 10000 -Gpu

    Write-Host ''
    Write-Host '===== 2d. 1×1 投影残差 / pre-activation 数值 golden（阶段6，CPU/GPU 交叉验证）=====' -ForegroundColor Cyan
    $pproj = '..\..\tests\fixtures\model_proj.txt'
    $ppre  = '..\..\tests\fixtures\model_preact.txt'
    Assert-Acc 'cpu_proj'   ('..\testDigits test --cpu --prep_target 20 --model "' + $pproj + '"') 736 10000
    Assert-Acc 'gpu_proj'   ('..\testDigits test --gpu --prep_target 20 --model "' + $pproj + '"') 736 10000 -Gpu
    Assert-Acc 'cpu_preact' ('..\testDigits test --cpu --prep_target 20 --model "' + $ppre + '"') 1055 10000
    Assert-Acc 'gpu_preact' ('..\testDigits test --gpu --prep_target 20 --model "' + $ppre + '"') 1055 10000 -Gpu

    Write-Host ''
    Write-Host '===== 3. 旧模型兼容 golden（无 #SIG）=====' -ForegroundColor Cyan
    Assert-Acc 'oldcpu' '..\testDigits test --cpu --prep_target 20 --model model.txt' 1278 10000

    Write-Host ''
    Write-Host '===== 4. gen 结构 golden =====' -ForegroundColor Cyan
    Assert-Struct 'gen_struct' ('..\testDigits gen --model "' + '..\..\tests\fixtures\_gen_struct.txt' + '"') `
        (Join-Path $fx '_gen_struct.txt') `
        @('#SIG BN:OFF | depth=2', 'K1 32 1 5 5', 'B1 32', 'K2 64 32 3 3', 'B2 64', 'FC 10 2304')
    Assert-Struct 'gen_bn_struct' ('..\testDigits gen --BN --c1 8 --c2 8 --model "' + '..\..\tests\fixtures\_gen_bn_struct.txt' + '"') `
        (Join-Path $fx '_gen_bn_struct.txt') `
        @('#SIG BN:ON | depth=2', 'K1 8 1 5 5', 'BN1 8', 'K2 8 8 3 3', 'BN2 8', 'FC 10 288')
    Assert-Struct 'gen_d3_struct' ('..\testDigits gen --depth 3 --model "..\..\tests\fixtures\_gen_d3.txt"') `
        (Join-Path $fx '_gen_d3.txt') `
        @('#SIG BN:OFF | depth=3', 'ver=2', 'K1 32 1 3 3', 'P1 1 1 2', 'K2 64 32 3 3', 'P2 1 1 2', 'K3 64 64 3 3', 'P3 1 1 2', 'FC 10 1024')
    Assert-Struct 'gen_res_struct' ('..\testDigits gen --res --depth 3 --model "..\..\tests\fixtures\_gen_res.txt"') `
        (Join-Path $fx '_gen_res.txt') `
        @('#SIG BN:OFF | depth=3', 'ver=2', 'K1 32 1 3 3', 'P1 1 1 2', 'K2 32 32 3 3', 'P2 1 1 0', 'K3 32 32 3 3', 'P3 1 1 0', 'R3 1 0', 'FC 10 8192')

    Write-Host ''
    Write-Host '===== 4b. 阶段6 新能力 gen 结构 golden（深度上限 / 投影 / pre-activation）=====' -ForegroundColor Cyan
    # (a) 深度上限：gen --res --depth 16 → 16 层、stem 池化、末层 identity skip 指向 stem
    Assert-Struct 'gen_deep_struct' ('..\testDigits gen --res --depth 16 --seed 555 --model "..\..\tests\fixtures\_gen_deep.txt"') `
        (Join-Path $fx '_gen_deep.txt') `
        @('#SIG BN:OFF | depth=16', 'ver=2', 'K1 32 1 3 3', 'P1 1 1 2', 'P2 1 1 0', 'K16 32 32 3 3', 'P16 1 1 0', 'R16 1 0', 'FC 10 8192')
    # (b) 1×1 投影残差：R3 2 0 + RP3 32 32 + PB3 32
    Assert-Struct 'gen_proj_struct' ('..\testDigits gen --res2 --depth 3 --seed 777 --model "..\..\tests\fixtures\_gen_proj.txt"') `
        (Join-Path $fx '_gen_proj.txt') `
        @('#SIG BN:OFF | depth=3', 'ver=2', 'P1 1 1 2', 'P2 1 1 0', 'R3 2 0', 'RP3 32 32', 'PB3 32', 'FC 10 8192')
    # (c) pre-activation 残差：R3 1 0 1（第 4 个字段 resPos=1）
    Assert-Struct 'gen_preact_struct' ('..\testDigits gen --preact --depth 3 --seed 888 --model "..\..\tests\fixtures\_gen_preact.txt"') `
        (Join-Path $fx '_gen_preact.txt') `
        @('#SIG BN:OFF | depth=3', 'ver=2', 'P1 1 1 2', 'P2 1 1 0', 'R3 1 0 1', 'FC 10 8192')

    Write-Host ''
    Write-Host '===== 5. 报错 golden（退出码=1 + 关键提示）=====' -ForegroundColor Cyan
    Assert-Error 'e_bn_missing' ('..\testDigits test --cpu --prep_target 20 --model "' + $pbn + '"')                              '模型为 BN:ON，但未传 --BN'
    Assert-Error 'e_bn_extra'   ('..\testDigits test --cpu --BN --prep_target 20 --model "' + $p64 + '"')                        '传了 --BN，但模型为 BN:OFF'
    Assert-Error 'e_bkc_cpu_bn' ('trainingDigits bkc --cpu --BN --epochs 1 --model "' + $pbn + '"')                              'CPU 训练不支持 BN'
    Assert-Error 'e_bad_fc20'   ('..\testDigits test --cpu --prep_target 20 --model "..\..\tests\fixtures\model_bad_fc20.txt"')   'classes 超出 MAX_CLASSES(16)'
    Assert-Error 'e_bad_fc999'  ('..\testDigits test --cpu --prep_target 20 --model "..\..\tests\fixtures\model_bad_fc999.txt"')  'FC 输入维度与卷积展平不一致'

    Write-Host ''
    Write-Host '===== 6. 训练冒烟（1 epoch，1934 张，CPU）=====' -ForegroundColor Cyan
    if ($SkipSmoke) { Write-Result 'smoke' 'SKIP' '-SkipSmoke'; Write-Result 'smoke_res' 'SKIP' '-SkipSmoke' } else { Assert-Smoke; Assert-SmokeRes }

    Write-Host ''
    Write-Host '===== 6b. 可复现性 golden（阶段4.1：同 seed 逐位一致）=====' -ForegroundColor Cyan
    Assert-ReproGen
    if ($SkipSmoke) { Write-Result 'repro_bkc' 'SKIP' '-SkipSmoke' } else { Assert-ReproBkc }

    Write-Host ''
    Write-Host '===== 6c. 训练日志 golden（阶段4.3：--log 指标 + checkpoint 元数据）=====' -ForegroundColor Cyan
    Assert-TrainLog

    Write-Host ''
    Write-Host '===== 6d. 独立验证集 golden（阶段4.4：--val 评估 + val_acc 列）=====' -ForegroundColor Cyan
    Assert-Val

    Write-Host ''
    Write-Host '===== 6e. 数据校验 golden（阶段4.4：--checkdata / --strict-data）=====' -ForegroundColor Cyan
    Assert-CheckData

    Write-Host ''
    Write-Host '===== 6f. L2 weight decay golden（阶段7：--wd 位保护 / 生效）=====' -ForegroundColor Cyan
    if ($SkipSmoke) {
        Write-Result 'wd_bp' 'SKIP' '-SkipSmoke'
        Write-Result 'wd_active' 'SKIP' '-SkipSmoke'
        Write-Result 'wd_active_diff' 'SKIP' '-SkipSmoke'
    } else { Assert-WdBitProtect; Assert-WdActive }

    Write-Host ''
    Write-Host '===== 6g. 每轮存档 golden（阶段7：--epsave / --epsn）=====' -ForegroundColor Cyan
    if ($SkipSmoke) {
        Write-Result 'epsave' 'SKIP' '-SkipSmoke'
        Write-Result 'epsn' 'SKIP' '-SkipSmoke'
    } else { Assert-Epsave; Assert-Epsn }

    Write-Host ''
    Write-Host '===== 7. 梯度校验（--gradcheck：有限差分对拍残差/BN/conv/fc）=====' -ForegroundColor Cyan
    Assert-GradCheck

    Write-Host ''
    Write-Host '===== 8. acm 二进制格式专项（T8：默认 acm / 后缀分流 / 等价 / 精度占位）=====' -ForegroundColor Cyan
    Invoke-AcmTests
    }
}
finally {
    Remove-Item -LiteralPath (Join-Path $fx '_gen_struct.txt')    -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $fx '_gen_bn_struct.txt') -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $fx '_gen_d3.txt')        -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $fx '_gen_res.txt')       -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $fx '_gen_deep.txt')      -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $fx '_gen_proj.txt')      -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $fx '_gen_preact.txt')    -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $fx '_smoke_model.txt')   -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $fx '_smoke_res.txt')     -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $fx '_repro_gen1.txt')    -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $fx '_repro_gen2.txt')    -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $fx '_repro_gen3.txt')    -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $fx '_repro_bkc1.txt')    -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $fx '_repro_bkc2.txt')    -Force -ErrorAction SilentlyContinue
    # --epsave 默认目录（EPTSET\<模型名>\，相对 x64\Release）：兜底清理，避免残留
    Remove-Item -LiteralPath (Join-Path $rel 'EPTSET') -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ''
Write-Host '===== 汇总 =====' -ForegroundColor Cyan
Write-Host ('通过 ' + $script:pass + ' / 失败 ' + $script:fail + ' / 跳过 ' + $script:skip)
if ($script:fail -gt 0) { exit 1 } else { exit 0 }
