@echo off
cd /d "%~dp0"

if not exist .venv\Scripts\activate.bat (
    echo Creating venv...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
)
call .venv\Scripts\activate.bat

python -m pip install wheel setuptools packaging -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem Install torch+torchvision+xformers together so pip resolves a compatible set
python -m pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128 -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem Clean corrupted ~orch dist-info left by failed torch installs
for /d %%D in (".venv\Lib\site-packages\~*") do rmdir /s /q "%%D" 2>nul

rem Fix basicsr: uninstall broken original, install patched fork
python -m pip uninstall basicsr -y -q 2>nul
python -m pip install basicsr-fixed -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python -m pip install -r requirements.txt --no-build-isolation -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem Reinstall torch+xformers after requirements.txt to ensure consistent versions
python -m pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128 -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python download_ckpts.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python run_vbench.py vbench ^
    --output_dir results_vbench/videos ^
    --image_types "indoor,scenery"
