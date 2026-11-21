@echo off
chcp 936 >nul
setlocal enabledelayedexpansion

rem ==================================================
rem  CNN2 补训练脚本（2.bat）
rem  1.bat 训出的模型欠训练，本脚本用 batch 256 + 60 轮继续补训
rem  针对弱类 8/9/7 加强弹性与笔画粗细增强
rem  产出 P1_ft.txt ... P6_ft.txt（原模型保留）
rem  用法：1.bat 训练完成后运行；与 CNN2.exe、1.bat 同目录
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
set "RUNS=%ROOT%runs_ft"
set "THREADS=8"

echo ==================================================
echo  CNN2 补训练  P1~P6 -^> P1~P6_ft （batch 256, +60 epoch）
echo  根目录   : %ROOT%
echo  训练集   : %TRAIN%
echo  模型目录 : %MODELS%
echo  工作目录 : %RUNS%
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
if not exist "%MODELS%" (
    echo [错误] 找不到模型目录 "%MODELS%"，请先运行 1.bat。
    pause
    exit /b 1
)
if not exist "%RUNS%" mkdir "%RUNS%"

rem ==================== P1 补训（加弹性+粗细）====================
echo [1/6] 补训 P1（几何 + 新增弹性/粗细）...
copy /y "%MODELS%\P1_geom_s1001.txt" "%MODELS%\P1_ft.txt" >nul
if not exist "%RUNS%\P1" mkdir "%RUNS%\P1"
cd /d "%RUNS%\P1"
"%EXE%" "%TRAIN%" bkc --gpu --threads %THREADS% --epochs 60 --batch 256 --lr_fc 0.0001 --lr_conv1 0.0001 --lr_conv2 0.0005 --momentum 0.9 --seed 1101 --aug --aug_k 2 --aug_shift 3 --aug_rotate 12 --aug_scale 0.1 --aug_elastic 3 --aug_elastic_sigma 4.5 --aug_thick 50 --P --time --model "%MODELS%\P1_ft.txt"
echo.

rem ==================== P2 补训 ====================
echo [2/6] 补训 P2（弹性+粗细）...
copy /y "%MODELS%\P2_elastic_s2002.txt" "%MODELS%\P2_ft.txt" >nul
if not exist "%RUNS%\P2" mkdir "%RUNS%\P2"
cd /d "%RUNS%\P2"
"%EXE%" "%TRAIN%" bkc --gpu --threads %THREADS% --epochs 60 --batch 256 --lr_fc 0.0001 --lr_conv1 0.0001 --lr_conv2 0.0005 --momentum 0.9 --seed 2202 --aug --aug_k 2 --aug_shift 2 --aug_rotate 10 --aug_scale 0.08 --aug_elastic 4 --aug_elastic_sigma 4.5 --aug_thick 50 --P --time --model "%MODELS%\P2_ft.txt"
echo.

rem ==================== P3 补训（旋转降到 12°）====================
echo [3/6] 补训 P3（旋转降到 12 度）...
copy /y "%MODELS%\P3_rot_s3003.txt" "%MODELS%\P3_ft.txt" >nul
if not exist "%RUNS%\P3" mkdir "%RUNS%\P3"
cd /d "%RUNS%\P3"
"%EXE%" "%TRAIN%" bkc --gpu --threads %THREADS% --epochs 60 --batch 256 --lr_fc 0.0001 --lr_conv1 0.0001 --lr_conv2 0.0005 --momentum 0.9 --seed 3303 --aug --aug_k 3 --aug_shift 3 --aug_rotate 12 --aug_scale 0.1 --aug_elastic 3 --aug_elastic_sigma 4.5 --aug_thick 60 --P --time --model "%MODELS%\P3_ft.txt"
echo.

rem ==================== P4 补训（保持无增强）====================
echo [4/6] 补训 P4（无增强 + Nesterov）...
copy /y "%MODELS%\P4_nesterov_s4004.txt" "%MODELS%\P4_ft.txt" >nul
if not exist "%RUNS%\P4" mkdir "%RUNS%\P4"
cd /d "%RUNS%\P4"
"%EXE%" "%TRAIN%" bkc --gpu --threads %THREADS% --epochs 60 --batch 256 --lr_fc 0.0001 --lr_conv1 0.0001 --lr_conv2 0.0005 --momentum 0.95 --nesterov --seed 4404 --P --time --model "%MODELS%\P4_ft.txt"
echo.

rem ==================== P5 补训 ====================
echo [5/6] 补训 P5（强增强）...
copy /y "%MODELS%\P5_strongaug_s5005.txt" "%MODELS%\P5_ft.txt" >nul
if not exist "%RUNS%\P5" mkdir "%RUNS%\P5"
cd /d "%RUNS%\P5"
"%EXE%" "%TRAIN%" bkc --gpu --threads %THREADS% --epochs 60 --batch 256 --lr_fc 0.0001 --lr_conv1 0.0001 --lr_conv2 0.0005 --momentum 0.9 --seed 5505 --aug --aug_k 2 --aug_shift 3 --aug_rotate 16 --aug_scale 0.1 --aug_elastic 3 --aug_elastic_sigma 4.5 --aug_thick 60 --P --time --model "%MODELS%\P5_ft.txt"
echo.

rem ==================== P6 补训 ====================
echo [6/6] 补训 P6（混合强增强）...
copy /y "%MODELS%\P6_mixaug_s6006.txt" "%MODELS%\P6_ft.txt" >nul
if not exist "%RUNS%\P6" mkdir "%RUNS%\P6"
cd /d "%RUNS%\P6"
"%EXE%" "%TRAIN%" bkc --gpu --threads %THREADS% --epochs 60 --batch 256 --lr_fc 0.0001 --lr_conv1 0.0001 --lr_conv2 0.0005 --momentum 0.9 --seed 6606 --aug --aug_k 3 --aug_shift 3 --aug_rotate 14 --aug_scale 0.1 --aug_elastic 3 --aug_elastic_sigma 4.5 --aug_thick 40 --P --time --model "%MODELS%\P6_ft.txt"
echo.

echo ==================================================
echo  补训完成。产出：P1_ft.txt ... P6_ft.txt（在 %MODELS%）
echo ==================================================
pause
