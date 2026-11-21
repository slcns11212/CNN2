@echo off
chcp 936 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ============================================
rem  B方案：3模型集成 + TTA（seed 11/22/33）
rem  每模型两阶段：A(30轮 one-cycle + wd) + B(10轮微调)
rem  预计 ~5h45min + 集成测试
rem ============================================

set "EXE=%~dp0cnn2.exe"
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
set "MODELS=%~dp0models"
set "RESULT=%~dp0ensemble_result.txt"

if not exist "%EXE%" ( echo [错误] 找不到 %EXE% & pause & exit /b 1 )
if not exist "%TRAIN%" ( echo [错误] 找不到训练集 %TRAIN% & pause & exit /b 1 )
if not exist "%MODELS%" mkdir "%MODELS%"

set "COMMON=--gpu --BN --depth 4 --threads 8 --wd 3e-4 --aug --aug_k 6 --aug_shift 3 --aug_rotate 12 --aug_scale 0.1 --aug_elastic 4 --aug_elastic_sigma 4.5 --aug_thick 50 --prep_target 28"

echo === B方案集成训练开始 %date% %time% === > "%RESULT%"
echo [提示] 将重新训练 3 个模型（seed 11/22/33），会覆盖 b4_1/2/3.acm >> "%RESULT%"
echo [提示] 覆盖前会将旧模型轻量备份为 b4_*.acm.bak（仅首次，保留原始旧模型）>> "%RESULT%"

call :train 1 11
call :train 2 22
call :train 3 33

echo. >> "%RESULT%"
echo === 单模型测试 (+TTA k=11) === >> "%RESULT%"
for %%N in (1 2 3) do (
  echo --- b4_%%N.acm --- >> "%RESULT%"
  "%EXE%" "%TEST%" test --gpu --BN --prep_target 28 --tta --tta_k 11 --model "%MODELS%\b4_%%N.acm" >> "%RESULT%" 2>&1
)

echo. >> "%RESULT%"
echo === 集成 3模型 soft融合 + TTA k=11 === >> "%RESULT%"
"%EXE%" "%TEST%" test --gpu --BN --prep_target 28 --tta --tta_k 11 --fuse soft --model "%MODELS%\b4_1.acm,%MODELS%\b4_2.acm,%MODELS%\b4_3.acm" >> "%RESULT%" 2>&1

echo. >> "%RESULT%"
echo === 完成 %date% %time% === >> "%RESULT%"
type "%RESULT%"
pause
exit /b 0

:train
set "IDX=%~1"
set "SEED=%~2"
if exist "%MODELS%\b4_%IDX%.acm" (
  if not exist "%MODELS%\b4_%IDX%.acm.bak" (
    echo [提示] 备份旧模型到 b4_%IDX%.acm.bak >> "%RESULT%"
    copy /y "%MODELS%\b4_%IDX%.acm" "%MODELS%\b4_%IDX%.acm.bak" >nul
  )
  echo [提示] 删除已存在的 b4_%IDX%.acm（确保从头训练）>> "%RESULT%"
  del /q "%MODELS%\b4_%IDX%.acm"
)
echo. >> "%RESULT%"
echo #### 训练 b4_%IDX%.acm (seed %SEED%) 阶段A #### >> "%RESULT%"
"%EXE%" "%TRAIN%" bkc %COMMON% --epochs 30 --batch 256 --lr_fc 0.05 --lr_conv1 0.05 --lr_conv2 0.05 --onecycle --lr_warmup 5 --lr_min 0.1 --momentum 0.9 --nesterov --seed %SEED% --P --time --model "%MODELS%\b4_%IDX%.acm"
echo #### 训练 b4_%IDX%.acm (seed %SEED%) 阶段B微调 #### >> "%RESULT%"
"%EXE%" "%TRAIN%" bkc %COMMON% --epochs 10 --batch 256 --lr_fc 0.001 --lr_conv1 0.001 --lr_conv2 0.001 --momentum 0.9 --nesterov --seed %SEED% --time --model "%MODELS%\b4_%IDX%.acm"
goto :eof
