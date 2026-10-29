@echo off
echo =======================================================
echo   FedEdge-IoT: Running All 39 Unit Tests
echo =======================================================
cd /d "%~dp0"
call venv\Scripts\activate.bat
pytest tests/ -v
pause

