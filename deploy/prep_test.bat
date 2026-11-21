@echo off
chcp 936 >nul
setlocal enabledelayedexpansion

rem ==================================================
rem  CNN2 预处理诊断脚本
rem  目的：固定训练配方，只改预处理，比较哪种预处理能让单模型测试精度更高
rem  配方：M_best 风格，25 轮，三层 lr=0.01，batch 256，seed 77
rem  每个变体独立工作目录，模型输出到 %ROOT%models\PREP_*.txt
rem
rem  预处理语法（--prep 显式指定时会先把所有步骤关掉，再按清单逐个开启）：
rem    --prep crop,deskew,resize,center,thick   逗号分隔步骤名；none 或 1 = 全部关闭
rem    --prep_target N        resize 目标框最长边（16-30，默认 28）
rem    --prep_stretch         拉伸到 target×target（不加=保持长宽比）
rem    --prep_ink X           thick 目标墨迹比（默认 0.20，0=关闭）
rem    --prep_deskew_max D    deskew 角度限幅（度，默认 15）
rem    --diag                 打印预处理前后统计
rem  注意：训练与测试必须使用完全相同的预处理参数。
rem ==================================================

set "ROOT=%~dp0"
set "EXE=%ROOT%CNN2.exe"
if not defined TRAIN set "TRAIN=%~dp0..\data\trainingDigits"
if not exist "%TRAIN%\" (
    echo [ERROR] 训练数据目录不存在: %TRAIN%
    echo         请设置环境变量 TRAIN 或把数据放到 deploy\..\data\trainingDigits
    exit /b 1
)
if not defined TEST set "TEST=%~dp0..\data\testDigits"
if not exist "%TEST%\" (
    echo [ERROR] 测试数据目录不存在: %TEST%
    echo         请设置环境变量 TEST 或把数据放到 deploy\..\data\testDigits
    exit /b 1
)
set "MODELS=%ROOT%models"
set "RUNS=%ROOT%runs"
set "THREADS=8"

rem 固定训练配方（各变体共用，只有预处理参数不同）
set "COMMON=--gpu --threads %THREADS% --epochs 25 --batch 256 --lr_fc 0.01 --lr_conv1 0.01 --lr_conv2 0.01 --momentum 0.9 --seed 77 --aug --aug_k 2 --aug_shift 3 --aug_rotate 12 --aug_scale 0.1 --aug_elastic 3 --aug_elastic_sigma 4.5 --aug_thick 40 --P --time"

echo ==================================================
echo  CNN2 预处理诊断（固定配方 25 轮 lr=0.01 batch256 seed77）
echo  训练集 : %TRAIN%
echo  测试集 : %TEST%
echo  模型   : %MODELS%\PREP_*.txt
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
if not exist "%TEST%" (
    echo [错误] 找不到测试集目录："%TEST%"
    pause
    exit /b 1
)
if not exist "%MODELS%" mkdir "%MODELS%"
if not exist "%RUNS%" mkdir "%RUNS%"

rem ================================================================
rem 变体 1：默认预处理（crop=1 deskew=0 resize=1 center=1 thick=0 target=28 stretch=0）
rem         不加任何 --prep 参数
rem ================================================================
echo [1/5] 默认预处理 ...
if not exist "%RUNS%\prep_def" mkdir "%RUNS%\prep_def"
cd /d "%RUNS%\prep_def"
if exist "%MODELS%\PREP_DEF.txt" del /q "%MODELS%\PREP_DEF.txt"
"%EXE%" "%TRAIN%" bkc %COMMON% --model "%MODELS%\PREP_DEF.txt"
echo [测试] 默认预处理 ...
"%EXE%" "%TEST%" test --gpu --model "%MODELS%\PREP_DEF.txt"
echo.

rem ================================================================
rem 变体 2：开启 deskew（在默认基础上加 deskew）
rem         --prep 显式清单：crop,deskew,resize,center
rem ================================================================
echo [2/5] 开启 deskew ...
if not exist "%RUNS%\prep_deskew" mkdir "%RUNS%\prep_deskew"
cd /d "%RUNS%\prep_deskew"
if exist "%MODELS%\PREP_DESKEW.txt" del /q "%MODELS%\PREP_DESKEW.txt"
"%EXE%" "%TRAIN%" bkc %COMMON% --prep crop,deskew,resize,center --model "%MODELS%\PREP_DESKEW.txt"
echo [测试] 开启 deskew ...
"%EXE%" "%TEST%" test --gpu --model "%MODELS%\PREP_DESKEW.txt" --prep crop,deskew,resize,center
echo.

rem ================================================================
rem 变体 3：开启 thick（在默认基础上加 thick，墨迹比用默认 0.20）
rem         --prep 显式清单：crop,resize,center,thick
rem ================================================================
echo [3/5] 开启 thick（墨迹比 0.20）...
if not exist "%RUNS%\prep_thick" mkdir "%RUNS%\prep_thick"
cd /d "%RUNS%\prep_thick"
if exist "%MODELS%\PREP_THICK.txt" del /q "%MODELS%\PREP_THICK.txt"
"%EXE%" "%TRAIN%" bkc %COMMON% --prep crop,resize,center,thick --model "%MODELS%\PREP_THICK.txt"
echo [测试] 开启 thick（墨迹比 0.20）...
"%EXE%" "%TEST%" test --gpu --model "%MODELS%\PREP_THICK.txt" --prep crop,resize,center,thick
echo.

rem ================================================================
rem 变体 4：resize 目标框改为 28（默认预处理 + --prep_target 28）
rem ================================================================
echo [4/5] resize target=28 ...
if not exist "%RUNS%\prep_t28" mkdir "%RUNS%\prep_t28"
cd /d "%RUNS%\prep_t28"
if exist "%MODELS%\PREP_T28.txt" del /q "%MODELS%\PREP_T28.txt"
"%EXE%" "%TRAIN%" bkc %COMMON% --prep_target 28 --model "%MODELS%\PREP_T28.txt"
echo [测试] resize target=28 ...
"%EXE%" "%TEST%" test --gpu --model "%MODELS%\PREP_T28.txt" --prep_target 28
echo.

rem ================================================================
rem 变体 5：开启 stretch（默认预处理 + 拉伸到 target×target=28×28）
rem ================================================================
echo [5/5] 开启 stretch ...
if not exist "%RUNS%\prep_stretch" mkdir "%RUNS%\prep_stretch"
cd /d "%RUNS%\prep_stretch"
if exist "%MODELS%\PREP_STRETCH.txt" del /q "%MODELS%\PREP_STRETCH.txt"
"%EXE%" "%TRAIN%" bkc %COMMON% --prep_stretch --model "%MODELS%\PREP_STRETCH.txt"
echo [测试] 开启 stretch ...
"%EXE%" "%TEST%" test --gpu --model "%MODELS%\PREP_STRETCH.txt" --prep_stretch
echo.

echo ==================================================
echo  全部 5 个变体训练 + 测试完成。
echo  上方每条 "[测试]" 输出里的准确率即为该预处理的结果。
echo  模型文件：
echo    %MODELS%\PREP_DEF.txt      默认
echo    %MODELS%\PREP_DESKEW.txt   开启 deskew
echo    %MODELS%\PREP_THICK.txt    开启 thick
echo    %MODELS%\PREP_T28.txt      target=28
echo    %MODELS%\PREP_STRETCH.txt  开启 stretch
echo ==================================================
pause
