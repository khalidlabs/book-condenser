#!/usr/bin/env python3
"""Compatibility launcher for the Book Condenser CLI.

Prefer the installed `book-condenser` command after `pip install -e .`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from book_condenser.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
