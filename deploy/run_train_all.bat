@echo off
chcp 936 >nul
setlocal enabledelayedexpansion

rem ==================================================
rem  CNN2 一键训练 P1~P6（单集训练）
rem  目标机器：Xeon E5-2667 + Tesla P100 + Windows
rem  用法：把本脚本与 CNN2.exe、各 dll 放在同一目录，双击运行即可。
rem  训练集：..\data\trainingDigits（见下方 TRAIN）
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
set "RUNS=%ROOT%runs"
set "THREADS=8"

echo ==================================================
echo  CNN2 一键训练  P1~P6 （单集，线程数 %THREADS%）
echo  根目录     : %ROOT%
echo  训练集     : %TRAIN%
echo  模型输出   : %MODELS%
echo  工作目录   : %RUNS%
echo ==================================================
echo.

rem ---------- 依赖检查 ----------
if not exist "%EXE%" (
    echo [错误] 找不到 CNN2.exe："%EXE%"
    echo        请确认本脚本与 CNN2.exe、cudart64_12.dll 等放在同一目录。
    pause
    exit /b 1
)
if not exist "%TRAIN%" (
    echo [错误] 找不到训练集目录："%TRAIN%"
    echo        请修改脚本顶部的 TRAIN 变量后再运行。
    pause
    exit /b 1
)

rem ---------- 生成输出目录 ----------
if not exist "%MODELS%" mkdir "%MODELS%"
if not exist "%RUNS%"   mkdir "%RUNS%"


rem ==================== P1 轻几何增强 ====================
echo [1/6] P1 轻几何增强（seed 1001）...
if not exist "%RUNS%\P1" mkdir "%RUNS%\P1"
cd /d "%RUNS%\P1"
"%EXE%" "%TRAIN%" bkc --gpu --threads %THREADS% --epochs 30 --batch 512 --lr_fc 0.0002 --lr_conv1 0.0002 --lr_conv2 0.001 --momentum 0.9 --seed 1001 --aug --aug_k 2 --aug_shift 3 --aug_rotate 12 --aug_scale 0.1 --aug_elastic 0 --aug_thick 0 --P --time --model "%MODELS%\P1_geom_s1001.txt"
echo.

rem ==================== P2 弹性形变 + 笔画粗细 ====================
echo [2/6] P2 弹性形变 + 笔画粗细（seed 2002）...
if not exist "%RUNS%\P2" mkdir "%RUNS%\P2"
cd /d "%RUNS%\P2"
"%EXE%" "%TRAIN%" bkc --gpu --threads %THREADS% --epochs 30 --batch 512 --lr_fc 0.0002 --lr_conv1 0.0002 --lr_conv2 0.001 --momentum 0.9 --seed 2002 --aug --aug_k 2 --aug_shift 2 --aug_rotate 10 --aug_scale 0.08 --aug_elastic 4 --aug_elastic_sigma 4.5 --aug_thick 50 --P --time --model "%MODELS%\P2_elastic_s2002.txt"
echo.

rem ==================== P3 强旋转 + 强粗细 + 高 conv2 LR ====================
echo [3/6] P3 强旋转 + 强粗细 + 高 conv2 LR（seed 3003）...
if not exist "%RUNS%\P3" mkdir "%RUNS%\P3"
cd /d "%RUNS%\P3"
"%EXE%" "%TRAIN%" bkc --gpu --threads %THREADS% --epochs 40 --batch 512 --lr_fc 0.0002 --lr_conv1 0.0002 --lr_conv2 0.002 --momentum 0.9 --seed 3003 --aug --aug_k 3 --aug_shift 4 --aug_rotate 18 --aug_scale 0.12 --aug_elastic 2 --aug_elastic_sigma 4.5 --aug_thick 70 --P --time --model "%MODELS%\P3_rot_s3003.txt"
echo.

rem ==================== P4 无增强 + Nesterov + 大batch ====================
echo [4/6] P4 无增强 + Nesterov + 大batch（seed 4004）...
if not exist "%RUNS%\P4" mkdir "%RUNS%\P4"
cd /d "%RUNS%\P4"
"%EXE%" "%TRAIN%" bkc --gpu --threads %THREADS% --epochs 30 --batch 256 --lr_fc 0.0002 --lr_conv1 0.0002 --lr_conv2 0.001 --momentum 0.95 --nesterov --seed 4004 --P --time --model "%MODELS%\P4_nesterov_s4004.txt"
echo.

rem ==================== P5 难类过采样 + 强增强 ====================
echo [5/6] P5 强增强（难类针对性，seed 5005）...
if not exist "%RUNS%\P5" mkdir "%RUNS%\P5"
cd /d "%RUNS%\P5"
"%EXE%" "%TRAIN%" bkc --gpu --threads %THREADS% --epochs 50 --batch 256 --lr_fc 0.0002 --lr_conv1 0.0002 --lr_conv2 0.001 --momentum 0.9 --seed 5005 --aug --aug_k 2 --aug_shift 3 --aug_rotate 16 --aug_scale 0.1 --aug_elastic 3 --aug_elastic_sigma 4.5 --aug_thick 60 --P --time --model "%MODELS%\P5_strongaug_s5005.txt"
echo.

rem ==================== P6 混合强增强 ====================
echo [6/6] P6 混合强增强（seed 6006）...
if not exist "%RUNS%\P6" mkdir "%RUNS%\P6"
cd /d "%RUNS%\P6"
"%EXE%" "%TRAIN%" bkc --gpu --threads %THREADS% --epochs 40 --batch 512 --lr_fc 0.0002 --lr_conv1 0.0002 --lr_conv2 0.001 --momentum 0.9 --seed 6006 --aug --aug_k 3 --aug_shift 3 --aug_rotate 14 --aug_scale 0.1 --aug_elastic 3 --aug_elastic_sigma 4.5 --aug_thick 40 --P --time --model "%MODELS%\P6_mixaug_s6006.txt"
echo.

echo ==================================================
echo  全部 6 个配置训练完成。
echo  模型输出目录：%MODELS%
echo ==================================================
pause
