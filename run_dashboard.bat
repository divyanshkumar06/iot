@echo off
echo =======================================================
echo   FedEdge-IoT: Launching Streamlit Dashboard
echo =======================================================
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Starting Streamlit...
streamlit run app.py
pause

