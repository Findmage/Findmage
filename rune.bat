@echo off
REM Aktifkan lingkungan virtual
CALL "%~dp0Findmage\Scripts\activate.bat"

REM Jalankan skrip desk.py yang ada di dalam folder cont
python "%~dp0Findmage\cont\desk.py"

REM Nonaktifkan lingkungan virtual
CALL "%~dp0Findmage\Scripts\deactivate.bat"
