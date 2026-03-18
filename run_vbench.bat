@echo off
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat

cd /d "%~dp0"

python download_ckpts.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python run_vbench.py vbench ^
    --output_dir results_vbench/videos ^
    --image_types "indoor,scenery"
