# Developer Setup

## Prerequisites
- Python 3.9+
- Git
- Virtual environment (recommended)

## Steps
1. Clone the repo.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt -r requirements-dev.txt`
4. Copy `.env.example` to `.env` and fill in credentials.
5. Run tests: `pytest -v`
6. Run locally (polling): `python main.py`
7. (Optional) Run webhook locally using `ngrok` and `python app.py`.