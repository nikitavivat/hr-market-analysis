#!/usr/bin/env python3
"""Script to run HH.ru scraper."""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import from src.api
from src.api.main import main

if __name__ == "__main__":
    print("Starting HH.ru scraper...")
    print(f"Working directory: {project_root}")
    print(f"CSV output: data/raw/hh_vacancies.csv")
    print("-" * 50)
    asyncio.run(main())

