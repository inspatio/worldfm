@echo off
cd /d "%~dp0"

if not exist .venvv\Scripts\activate.bat (
    echo Creating venv...
    py -3.12 -m venv .venvv
    if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
)
call .venvv\Scripts\activate.bat

python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python -m pip install -r requirements2.txt --no-build-isolation -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python download_ckpts.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python run_vbench.py vbench ^
    --output_dir results_vbench/videos ^
    --image_types "indoor,scenery"
