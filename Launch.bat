@echo off
echo Launching Deposition Summarizer... Please wait.
cd /d "D:\Qsync\Christopher Snead\Work\Python AI\deposition-summarizer"
call .venv\Scripts\activate.bat
echo Starting Streamlit... The app will open in your browser.
echo Keep this window open to keep the app running.
streamlit run app.py
pause