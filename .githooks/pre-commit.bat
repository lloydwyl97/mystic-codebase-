@echo off
REM Pre-commit lint hook (Windows). Blocks commit on errors.
setlocal enabledelayedexpansion

REM Resolve project root (this script lives in .githooks)
set ROOT=%~dp0..
pushd "%ROOT%"

REM Prefer venv linters, else global
set RUFF=ruff
set FLAKE8=flake8

if exist "venv\Scripts\ruff.exe" set RUFF=venv\Scripts\ruff.exe
if exist "venv\Scripts\flake8.exe" set FLAKE8=venv\Scripts\flake8.exe

echo [pre-commit] Running Ruff...
%RUFF% check .
if errorlevel 1 (
  echo [pre-commit] Ruff failed. Fix issues before committing.
  popd
  exit /b 1
)

echo [pre-commit] Running Flake8 (critical errors only)...
%FLAKE8% . --select=E9,F63,F7,F82 --show-source --statistics
if errorlevel 1 (
  echo [pre-commit] Flake8 failed. Fix issues before committing.
  popd
  exit /b 1
)

echo [pre-commit] OK
popd
exit /b 0


