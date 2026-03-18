@echo off
cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo Creating venv...
    python -m venv venv
    if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
)
call venv\Scripts\activate.bat

python -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python -m pip install -r requirements.txt -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python download_ckpts.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python run_vbench.py vbench ^
    --output_dir results_vbench/videos ^
    --image_types "indoor,scenery"
