@echo off
setlocal
REM ======================================================================
REM  Athanor's test suite.
REM
REM    run_tests.bat                       uses the Python launcher (py -3),
REM                                        else python on PATH
REM    run_tests.bat "C:\path\python.exe"  uses that Python - e.g. the one in
REM                                        your ATK's envs\atk_core\Scripts,
REM                                        which already has llama.cpp
REM
REM  Tests that need llama.cpp run when that Python has llama-cpp-python;
REM  otherwise they are skipped and say so. Optional, before running:
REM    set ATHANOR_TEST_MODEL=D:\path\to\a\model.gguf
REM        - checks a real model too (header, round trip, vocab-only load)
REM    set ATHANOR_TEST_VOCAB_DIR=D:\path\to\llama.cpp\models
REM        - llama.cpp's whole ggml-vocab-*.gguf set
REM ======================================================================
cd /d "%~dp0"
if not "%~1"=="" (
    "%~1" -m unittest discover -s tests -v
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m unittest discover -s tests -v
    ) else (
        python -m unittest discover -s tests -v
    )
)
set "RC=%ERRORLEVEL%"
echo.
echo Athanor tests finished with exit code %RC%.
REM Pause only when double-clicked (the console would close); never when
REM run from an open console or by another program.
echo %cmdcmdline% | findstr /i /c:"%~nx0" >nul && pause
exit /b %RC%
