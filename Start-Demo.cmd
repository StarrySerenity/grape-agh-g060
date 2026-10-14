@echo off
cd /d "%~dp0"
if defined GRAPE_PYTHON (set "PYTHON_RUN=%GRAPE_PYTHON%") else (set "PYTHON_RUN=python")
start "" http://127.0.0.1:8870/web/showcase.html
"%PYTHON_RUN%" cli.py serve --port 8870
pause
