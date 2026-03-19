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

rem Install torch nightly first so xformers picks the right version
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem Clean corrupted ~orch dist-info left by failed torch installs
for /d %%D in (".venv\Lib\site-packages\~*") do rmdir /s /q "%%D" 2>nul

rem Fix basicsr: uninstall broken original, install patched fork
python -m pip uninstall basicsr -y -q 2>nul
python -m pip install basicsr-fixed -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python -m pip install -r requirements.txt --no-build-isolation -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem Reinstall torch nightly after requirements.txt to ensure correct version
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem Install xformers from nightly index to match torch nightly
python -m pip install --pre xformers --index-url https://download.pytorch.org/whl/nightly/cu128 -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python download_ckpts.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python run_vbench.py vbench ^
    --output_dir results_vbench/videos ^
    --image_types "indoor,scenery"
