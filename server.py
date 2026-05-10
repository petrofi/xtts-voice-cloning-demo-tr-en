
"""
Entry point for the FastAPI app.

Run:
  pip install -r requirements.txt
  uvicorn server:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# Allow running without installing the package (adds ./src to PYTHONPATH).
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tts_demo.api import app  # noqa: F401

