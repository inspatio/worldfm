@echo off
setlocal enabledelayedexpansion

set PY_SCRIPT=%~dp0vbench_runner.py

cd /d "%~dp0.."

:: Default paths relative to this repo — override by passing them as arguments:
::   vbench.bat [VBENCH_JSON] [VBENCH_CROP] [FPS]
if not "%~1"=="" (set VBENCH_JSON=%~1) else (set VBENCH_JSON=%CD%\..\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\i2v-bench-info.json)
if not "%~2"=="" (set VBENCH_CROP=%~2) else (set VBENCH_CROP=%CD%\..\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\crop\1-1)
if not "%~3"=="" (set FPS=%~3) else (set FPS=30)

echo Output:    %CD%\outputs\vbench
echo VBench:    %VBENCH_JSON%
echo Crop:      %VBENCH_CROP%
echo FPS:       %FPS%

python "%PY_SCRIPT%" ^
    --vbench-json "%VBENCH_JSON%" ^
    --vbench-crop "%VBENCH_CROP%" ^
    --work-dir "%CD%" ^
    --fps %FPS%

exit /b %ERRORLEVEL%
