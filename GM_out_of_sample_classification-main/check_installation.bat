@echo off
echo ============================================================
echo Checking Python and R Installation
echo ============================================================
echo.

echo Checking Python...
echo ----------------------------------------
python --version 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Python is installed
    python --version
    echo.
    echo Python location:
    where python
) else (
    echo [ERROR] Python is NOT installed or not in PATH
    echo Download from: https://www.python.org/downloads/
)

echo.
echo ----------------------------------------
echo Checking pip...
echo ----------------------------------------
pip --version 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] pip is installed
    pip --version
) else (
    echo [ERROR] pip is NOT installed
)

echo.
echo ----------------------------------------
echo Checking R...
echo ----------------------------------------
R --version 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] R is installed
    R --version | findstr "R version"
    echo.
    echo R location:
    where R
) else (
    echo [ERROR] R is NOT installed or not in PATH
    echo Download from: https://cran.r-project.org/
)

echo.
echo ----------------------------------------
echo Checking Rscript...
echo ----------------------------------------
Rscript --version 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Rscript is available
) else (
    echo [ERROR] Rscript is NOT available
)

echo.
echo ============================================================
echo Check Complete
echo ============================================================
echo.
echo Required versions:
echo   - Python 3.7 or higher
echo   - R 3.6 or higher
echo.
pause
