@echo off
setlocal
cd /d "%~dp0"
if not exist .venv (
    py -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if not exist models\best_model.joblib python run_pipeline.py
python -m streamlit run app.py
endlocal

