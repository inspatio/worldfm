@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
call .venv\Scripts\activate.bat

set VBENCH_JSON=..\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\i2v-bench-info.json
set VBENCH_CROP=..\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\crop\1-1

echo Output:  %CD%\results_vbench\videos
echo VBench:  %VBENCH_JSON%
echo Crop:    %VBENCH_CROP%

python run_vbench.py ^
    --vbench-json "%VBENCH_JSON%" ^
    --vbench-crop "%VBENCH_CROP%" ^
    --config vbench.yaml ^
    --num-samples 12 ^
    --skip-existing

exit /b %ERRORLEVEL%
