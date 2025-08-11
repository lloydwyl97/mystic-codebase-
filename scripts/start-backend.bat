@echo off
echo Starting Backend Server...
cd backend
echo Using global Python environment...
echo Python path: C:\Users\lloyd\AppData\Local\Programs\Python\Python310\python.exe
python -m uvicorn main:app --host localhost --port 8000
pause
