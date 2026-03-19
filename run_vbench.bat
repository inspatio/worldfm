@echo off
cd /d "%~dp0"

rem ── 1. Create venv ─────────────────────────────────────────────────────────
if not exist .venv\Scripts\activate.bat (
    echo Creating venv...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
)
call .venv\Scripts\activate.bat

rem ── 2. Base build tools ────────────────────────────────────────────────────
python -m pip install wheel setuptools packaging -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem ── 3. Torch + xformers (CUDA 12.8) ───────────────────────────────────────
python -m pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128 -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem Clean corrupted ~orch dist-info left by failed torch installs
for /d %%D in (".venv\Lib\site-packages\~*") do rmdir /s /q "%%D" 2>nul

rem ── 4. basicsr-fixed (before requirements.txt pulls in broken basicsr) ─────
python -m pip uninstall basicsr -y -q 2>nul
python -m pip install basicsr-fixed --no-deps -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem ── 5. Project requirements ────────────────────────────────────────────────
python -m pip install -r requirements.txt --no-build-isolation -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem Pin utils3d to the commit required by MoGe (PyPI version dropped the .np submodule)
python -m pip install "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38" -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem Reinstall torch+xformers after requirements.txt to ensure consistent versions
python -m pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128 -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem ── 6. Git submodules ──────────────────────────────────────────────────────
git submodule update --init --recursive
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem Pin MoGe to the tested commit
pushd submodules\MoGe
git checkout 7807b5de2bc0c1e80519f5f3d1f38a606f8f9925
if %ERRORLEVEL% neq 0 ( popd & exit /b %ERRORLEVEL% )
popd

rem ── 7. Real-ESRGAN + ZIM submodules ───────────────────────────────────────
python -m pip install facexlib gfpgan -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

if not exist submodules\Real-ESRGAN\realesrgan\version.py (
    pushd submodules\Real-ESRGAN
    python setup.py develop -q
    if %ERRORLEVEL% neq 0 ( popd & exit /b %ERRORLEVEL% )
    popd
)

python -m pip install -e submodules\ZIM -q
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem ── 8. Patch basicsr: functional_tensor removed in torchvision 0.17+ ───────
python -c "import pathlib; p=pathlib.Path('.venv/Lib/site-packages/basicsr/data/degradations.py'); t=p.read_text(encoding='utf-8'); fixed=t.replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale','from torchvision.transforms.functional import rgb_to_grayscale'); p.write_text(fixed,encoding='utf-8'); c=p.parent/'__pycache__'; [f.unlink() for f in c.glob('degradations*.pyc')] if c.exists() else None; print('basicsr patch applied' if fixed!=t else 'basicsr already patched')"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem ── 9. Download model weights ──────────────────────────────────────────────
python download_ckpts.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

rem ── 10. Run vbench ─────────────────────────────────────────────────────────
python run_vbench.py vbench ^
    --output_dir results_vbench/videos ^
    --image_types "indoor,scenery"
