@echo off
if exist ".venv" rmdir /s /q ".venv"
python.exe -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
python --version