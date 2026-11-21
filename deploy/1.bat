@echo off
chcp 936 >nul
setlocal enabledelayedexpansion

rem ==================================================
rem  CNN2 六模型训练（target=28 + CD 合体配方）
rem  每模型：阶段A 30轮(One-Cycle 峰值0.01,warmup6) + 阶段B 10轮(lr0.001)
rem  全部 --prep_target 28, batch 256, 强增强；不同 seed/增强 制造多样性
rem  输出 M1..M6 到 models\  （M5 为固定两阶段，无 one-cycle）
rem ==================================================

set "ROOT=%~dp0"
set "EXE=%ROOT%CNN2.exe"
if not defined TRAIN set "TRAIN=%~dp0..\data\trainingDigits"
if not exist "%TRAIN%\" (
    echo [ERROR] 训练数据目录不存在: %TRAIN%
    echo         请设置环境变量 TRAIN 或把数据放到 deploy\..\data\trainingDigits
    exit /b 1
)
set "MODELS=%ROOT%models"
set "RUNS=%ROOT%runs_ens"
set "THREADS=8"

echo ==================================================
echo  CNN2 六模型训练（target=28 + CD）
echo  训练集 : %TRAIN%
echo  模型目录: %MODELS%
echo ==================================================
echo.

if not exist "%EXE%" (
    echo [错误] 找不到 CNN2.exe："%EXE%"
    pause
    exit /b 1
)
if not exist "%TRAIN%" (
    echo [错误] 找不到训练集目录："%TRAIN%"
    pause
    exit /b 1
)
if not exist "%MODELS%" mkdir "%MODELS%"
if not exist "%RUNS%" mkdir "%RUNS%"

set "CA=--gpu --threads %THREADS% --epochs 30 --batch 256 --momentum 0.9 --prep_target 28 --P --time"
set "CB=--gpu --threads %THREADS% --epochs 10 --batch 256 --momentum 0.9 --lr_fc 0.001 --lr_conv1 0.001 --lr_conv2 0.001 --prep_target 28 --P --time"

rem ==================== M1 ====================
echo [1/6] M1 (seed 11, one-cycle) ...
if not exist "%MODELS%\M1.txt" (
  if not exist "%RUNS%\M1" mkdir "%RUNS%\M1"
  cd /d "%RUNS%\M1"
  "%EXE%" "%TRAIN%" bkc %CA% --lr_fc 0.01 --lr_conv1 0.01 --lr_conv2 0.01 --onecycle --lr_warmup 6 --lr_min 0.01 --seed 11 --aug --aug_k 4 --aug_shift 3 --aug_rotate 12 --aug_scale 0.1 --aug_elastic 4 --aug_elastic_sigma 4.5 --aug_thick 50 --model "%MODELS%\M1.txt"
  "%EXE%" "%TRAIN%" bkc %CB% --seed 11 --aug --aug_k 4 --aug_shift 3 --aug_rotate 12 --aug_scale 0.1 --aug_elastic 4 --aug_elastic_sigma 4.5 --aug_thick 50 --model "%MODELS%\M1.txt"
)
echo.

rem ==================== M2 ====================
echo [2/6] M2 (seed 22, one-cycle, 轻增强) ...
if not exist "%MODELS%\M2.txt" (
  if not exist "%RUNS%\M2" mkdir "%RUNS%\M2"
  cd /d "%RUNS%\M2"
  "%EXE%" "%TRAIN%" bkc %CA% --lr_fc 0.01 --lr_conv1 0.01 --lr_conv2 0.01 --onecycle --lr_warmup 6 --lr_min 0.01 --seed 22 --aug --aug_k 4 --aug_shift 3 --aug_rotate 10 --aug_scale 0.08 --aug_elastic 3 --aug_elastic_sigma 4.5 --aug_thick 40 --model "%MODELS%\M2.txt"
  "%EXE%" "%TRAIN%" bkc %CB% --seed 22 --aug --aug_k 4 --aug_shift 3 --aug_rotate 10 --aug_scale 0.08 --aug_elastic 3 --aug_elastic_sigma 4.5 --aug_thick 40 --model "%MODELS%\M2.txt"
)
echo.

rem ==================== M3 ====================
echo [3/6] M3 (seed 33, one-cycle, 强旋转/弹性) ...
if not exist "%MODELS%\M3.txt" (
  if not exist "%RUNS%\M3" mkdir "%RUNS%\M3"
  cd /d "%RUNS%\M3"
  "%EXE%" "%TRAIN%" bkc %CA% --lr_fc 0.01 --lr_conv1 0.01 --lr_conv2 0.01 --onecycle --lr_warmup 6 --lr_min 0.01 --seed 33 --aug --aug_k 3 --aug_shift 3 --aug_rotate 14 --aug_scale 0.1 --aug_elastic 5 --aug_elastic_sigma 4.5 --aug_thick 50 --model "%MODELS%\M3.txt"
  "%EXE%" "%TRAIN%" bkc %CB% --seed 33 --aug --aug_k 3 --aug_shift 3 --aug_rotate 14 --aug_scale 0.1 --aug_elastic 5 --aug_elastic_sigma 4.5 --aug_thick 50 --model "%MODELS%\M3.txt"
)
echo.

rem ==================== M4 ====================
echo [4/6] M4 (seed 44, one-cycle) ...
if not exist "%MODELS%\M4.txt" (
  if not exist "%RUNS%\M4" mkdir "%RUNS%\M4"
  cd /d "%RUNS%\M4"
  "%EXE%" "%TRAIN%" bkc %CA% --lr_fc 0.01 --lr_conv1 0.01 --lr_conv2 0.01 --onecycle --lr_warmup 6 --lr_min 0.01 --seed 44 --aug --aug_k 4 --aug_shift 3 --aug_rotate 12 --aug_scale 0.1 --aug_elastic 4 --aug_elastic_sigma 4.5 --aug_thick 50 --model "%MODELS%\M4.txt"
  "%EXE%" "%TRAIN%" bkc %CB% --seed 44 --aug --aug_k 4 --aug_shift 3 --aug_rotate 12 --aug_scale 0.1 --aug_elastic 4 --aug_elastic_sigma 4.5 --aug_thick 50 --model "%MODELS%\M4.txt"
)
echo.

rem ==================== M5 (固定两阶段, 无 one-cycle) ====================
echo [5/6] M5 (seed 55, 固定两阶段) ...
if not exist "%MODELS%\M5.txt" (
  if not exist "%RUNS%\M5" mkdir "%RUNS%\M5"
  cd /d "%RUNS%\M5"
  "%EXE%" "%TRAIN%" bkc %CA% --lr_fc 0.01 --lr_conv1 0.01 --lr_conv2 0.01 --seed 55 --aug --aug_k 4 --aug_shift 3 --aug_rotate 12 --aug_scale 0.1 --aug_elastic 4 --aug_elastic_sigma 4.5 --aug_thick 50 --model "%MODELS%\M5.txt"
  "%EXE%" "%TRAIN%" bkc %CB% --seed 55 --aug --aug_k 4 --aug_shift 3 --aug_rotate 12 --aug_scale 0.1 --aug_elastic 4 --aug_elastic_sigma 4.5 --aug_thick 50 --model "%MODELS%\M5.txt"
)
echo.

rem ==================== M6 ====================
echo [6/6] M6 (seed 66, one-cycle, 小旋转) ...
if not exist "%MODELS%\M6.txt" (
  if not exist "%RUNS%\M6" mkdir "%RUNS%\M6"
  cd /d "%RUNS%\M6"
  "%EXE%" "%TRAIN%" bkc %CA% --lr_fc 0.01 --lr_conv1 0.01 --lr_conv2 0.01 --onecycle --lr_warmup 6 --lr_min 0.01 --seed 66 --aug --aug_k 4 --aug_shift 3 --aug_rotate 8 --aug_scale 0.12 --aug_elastic 4 --aug_elastic_sigma 4.5 --aug_thick 45 --model "%MODELS%\M6.txt"
  "%EXE%" "%TRAIN%" bkc %CB% --seed 66 --aug --aug_k 4 --aug_shift 3 --aug_rotate 8 --aug_scale 0.12 --aug_elastic 4 --aug_elastic_sigma 4.5 --aug_thick 45 --model "%MODELS%\M6.txt"
)
echo.

echo ==================================================
echo  六模型训练完成：M1~M6.txt（在 %MODELS%）
echo  请把 models\ 导回本地做集成。
echo ==================================================
pause
