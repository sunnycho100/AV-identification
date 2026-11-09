"""Compatibility entry point for the SSH-ready review_web/server.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'review_web'))
from server import main
if __name__ == '__main__':
    main()
