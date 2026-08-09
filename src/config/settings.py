"""
===========================================================
TradeVision AI
Configuration Settings

Author: Phenyo
Project: TradeVision AI

Purpose:
This file stores all project-wide settings and file paths.
Instead of hardcoding paths throughout the project,
every module imports settings from here.

This makes the project easier to maintain and scale.
===========================================================
"""

from pathlib import Path

# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

# Absolute path to the TradeVision-AI project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------
# DATA DIRECTORIES
# ---------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

EXTERNAL_DATA_DIR = DATA_DIR / "external"

# ---------------------------------------------------------
# OTHER DIRECTORIES
# ---------------------------------------------------------

MODEL_DIR = PROJECT_ROOT / "saved_models"

REPORT_DIR = PROJECT_ROOT / "reports"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

ASSET_DIR = PROJECT_ROOT / "assets"

# ---------------------------------------------------------
# DEFAULT DOWNLOAD SETTINGS
# ---------------------------------------------------------

DEFAULT_START_DATE = "2010-01-01"

DEFAULT_END_DATE = None

DEFAULT_INTERVAL = "1d"

# ---------------------------------------------------------
# PROJECT INFORMATION
# ---------------------------------------------------------

PROJECT_NAME = "TradeVision AI"

VERSION = "0.1.0"

AUTHOR = "Phenyo"

# ---------------------------------------------------------
# CREATE DIRECTORIES IF THEY DON'T EXIST
# ---------------------------------------------------------

DIRECTORIES = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    NOTEBOOK_DIR,
    ASSET_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)