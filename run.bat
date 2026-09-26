@echo off
REM Usage: run.bat yourimage.jpg   (optional extra flags, e.g. --model base --invert)
if "%~1"=="" (echo Usage: run.bat image.jpg [--model base] [--invert] & exit /b 1)
python spike.py --image %*
