@echo off
setlocal
set "EDITOR_DIR=%~dp0"
set "REPO_DIR=%EDITOR_DIR%.."
set "VENV_PYTHON=%REPO_DIR%\.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" "%EDITOR_DIR%profile_editor_app.py"
) else (
    python "%EDITOR_DIR%profile_editor_app.py"
)
