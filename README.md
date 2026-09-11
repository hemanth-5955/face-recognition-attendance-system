# Face Recognition Attendance System

A desktop attendance system that uses a pretrained deep learning face recognition model (InsightFace / ArcFace) to enrol students and mark attendance automatically from a live webcam feed.

Built as part of a university dissertation project. The recognition model is pretrained this project focuses on integrating it into a complete attendance management application: enrolment, live recognition, a SQLite database and a desktop GUI.

## Features

- Register students with multiple live face samples per person
- Live webcam attendance with onscreen status (Present / Already Marked / Not Registered)
- Duplicate attendance prevention per session
- Editable student records, active/inactive status
- Searchable, sortable and editable attendance records
- Face embeddings encrypted at rest
- SQLite database (students, face embeddings, classes, sessions, attendance)

## Requirements

- Python 3.11 (a modern Tk is required for the GUI to render correctly on macOS)
- A webcam
- Internet access on first run (downloads the InsightFace model)

## Setup (macOS)

brew install python@3.11 python-tk@3.11
git clone (https://github.com/hemanth-5955/face-recognition-attendance-system.git)
cd FRS
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


## Setup (Windows)

git clone (https://github.com/hemanth-5955/face-recognition-attendance-system.git)
cd FRS
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

If PowerShell blocks venv\Scripts\Activate.ps1, either call venv\Scripts\python.exe directly or run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

## Encryption key setup (required)

Generate a local key:
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Create a file named .env in the project root containing:
ENCRYPTION_KEY= (paste the generated key here)

This key is required to read registered face data. It is excluded from version control and must never be shared. If lost, existing encrypted embeddings become permanently unreadable.

## Running

python3 gui.py

## Running the tests

python3 -m pytest tests/ -v

## Known limitations

- No liveness detection - a printed photo can currently be recognised. Documented as a scoped limitation.

## Privacy note

This prototype stores facial embeddings locally in frs.db, encrypted at rest using a key held in a local .env file. Both files are excluded from version control. Only enrol people who have given informed consent.

This encryption protects against casual inspection of the database file or a leaked repository. It does not protect against someone with full access to the running machine since the key must be readable for the app to function.
