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

rem Fix basicsr: uninstall broken original, install patched fork (--no-deps to avoid clobbering torch)
python -m pip uninstall basicsr -y -q 2>nul
python -m pip install basicsr-fixed --no-deps -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python -m pip install -r requirements.txt --no-build-isolation -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem Reinstall torch+xformers after requirements.txt to ensure consistent versions
python -m pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128 -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem Install Real-ESRGAN submodule (generates version.py via setup.py develop)
if not exist submodules\Real-ESRGAN\realesrgan\version.py (
    pushd submodules\Real-ESRGAN
    python setup.py develop -q
    if %ERRORLEVEL% neq 0 ( popd & exit /b %ERRORLEVEL% )
    popd
)

rem Install ZIM submodule
python -m pip install -e submodules\ZIM -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem Patch basicsr degradations.py: functional_tensor was removed in torchvision 0.17+
python -c "import pathlib,shutil; p=pathlib.Path('.venv/Lib/site-packages/basicsr/data/degradations.py'); t=p.read_text(encoding='utf-8'); fixed=t.replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale','from torchvision.transforms.functional import rgb_to_grayscale'); p.write_text(fixed,encoding='utf-8'); c=p.parent/'__pycache__'; [f.unlink() for f in c.glob('degradations*.pyc')] if c.exists() else None; print('basicsr patch applied' if fixed!=t else 'basicsr already patched')"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python download_ckpts.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

python run_vbench.py vbench ^
    --output_dir results_vbench/videos ^
    --image_types "indoor,scenery"
