@echo off
setlocal
cd /d "%~dp0"

REM Prefer local venv python if present
set PYEXE=python
if exist ".venv\Scripts\python.exe" set PYEXE=.venv\Scripts\python.exe

REM Example usage; edit the text as you wish:
%PYEXE% run_xtts_safe.py ^
  --text "Bugün test için biraz daha uzun bir cümleyi net ve doğal bir tonda okuyorum." ^
  --speaker_wav "data\wavs\trk_0001.wav" ^
  --language tr ^
  --out_path "out_safe.wav" ^
  --temperature 0.7 ^
  --gpt_cond_len 30 ^
  --gpt_cond_chunk 6 ^
  --gpu

echo.
echo [INFO] Output should be at out_safe.wav if no error.
pause
