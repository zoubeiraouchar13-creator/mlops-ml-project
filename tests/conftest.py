#tests/conftest.py — rend le package src/ importable par pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))