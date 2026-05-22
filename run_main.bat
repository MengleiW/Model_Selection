@echo off
set PYTHON_EXE=C:\Users\whisk\anaconda3\python.exe
set SCRIPT=%~dp0main.py
set MPLBACKEND=Agg

if not exist "%PYTHON_EXE%" (
  echo Python not found at %PYTHON_EXE%
  exit /b 1
)

if not exist "%SCRIPT%" (
  echo main.py not found at %SCRIPT%
  exit /b 1
)

"%PYTHON_EXE%" "%SCRIPT%"
