@echo off
setlocal

set WSL_PROJECT=/home/h621l/projects/science-article
set PYTHON=%WSL_PROJECT%/.venv/bin/python

echo ============================================================
echo  ML Pipeline: ResNet Transfer Learning + SoilGrids Baseline
echo ============================================================
echo.

:: ── Step 1: Transfer Learning ResNet ────────────────────────────────────────
echo [1/2] transfer_learning_resnet.py ...
echo.
wsl bash -c "cd %WSL_PROJECT% && %PYTHON% ML/transfer_learning_resnet.py"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] transfer_learning_resnet.py завершился с ошибкой (exit code %errorlevel%)
    pause
    exit /b %errorlevel%
)

echo.
echo [OK] transfer_learning_resnet.py завершён.
echo.

:: ── Step 2: SoilGrids Baseline ──────────────────────────────────────────────
echo [2/2] soilgrids_baseline.py ...
echo.
wsl bash -c "cd %WSL_PROJECT% && %PYTHON% ML/soilgrids_baseline.py"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] soilgrids_baseline.py завершился с ошибкой (exit code %errorlevel%)
    pause
    exit /b %errorlevel%
)

echo.
echo [OK] soilgrids_baseline.py завершён.
echo.
echo ============================================================
echo  [DONE] Оба скрипта выполнены успешно.
echo ============================================================
pause
endlocal
