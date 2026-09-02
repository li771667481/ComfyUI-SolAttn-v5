@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "COMFY_CUDA_ARCHS=75-real;80-real;86-real;89;120"
set "PATH=F:\ComfyUi\ComfyUI\.venv\Scripts;%PATH%"
cd /d F:\ComfyUi\ComfyUI\comfy-kitchen-solattn
F:\ComfyUi\ComfyUI\.venv\Scripts\python.exe -m pip install --no-cache-dir --no-build-isolation --force-reinstall --no-deps .
exit /b %ERRORLEVEL%