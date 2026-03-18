@echo off
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat

cd /d "%~dp0"

python run_vbench.py vbench ^
    --output_dir results_vbench/videos ^
    --image_types "indoor,scenery"
