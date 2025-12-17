"""
config.py

Static configuration for broker position ingestion and MSRB enrichment.

Defines:
- Broker source directories
- MSRB reference file location
- Output file locations
- Canonical output column schema

Dependency contract:
- Allowed imports: Python standard library only
- Forbidden imports: any project modules (utils/loaders/build/enrich)
"""

from pathlib import Path

BROKER_DIRS = [
    r'D:\BONDS\2-DATA SOURCE\BROKER DOWNLOAD\POSITIONS\CHASE',
    r'D:\BONDS\2-DATA SOURCE\BROKER DOWNLOAD\POSITIONS\FID',
    r'D:\BONDS\2-DATA SOURCE\BROKER DOWNLOAD\POSITIONS\MERRILL',
    r'D:\BONDS\2-DATA SOURCE\BROKER DOWNLOAD\POSITIONS\RJ',
    r'D:\BONDS\2-DATA SOURCE\BROKER DOWNLOAD\POSITIONS\SCHWAB',
    r'D:\BONDS\2-DATA SOURCE\BROKER DOWNLOAD\POSITIONS\WF'
]

MSRB_FILE = r"D:\BONDS\2-DATA SOURCE\MSRB\msrb_reference.xlsx"

OUTPUT_DIR = Path(r"D:\BONDS\3-OUTPUT")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POSITIONS_OUT = OUTPUT_DIR / "combined_positions.xlsx"
ENRICHED_OUT  = OUTPUT_DIR / "combined_positions_enriched.xlsx"

STANDARD_COLUMNS = [
    'CUSIP', 'DESCRIPTION', 'QTY', 'COUPON', 'MATURITY',
    'BASIS PRICE', 'BASIS VALUE', 'MRKT VALUE', 'MRKT PRICE',
    'MOODYS', 'S&P',
    'CALL_DATE',
    'ACQ DATE', 'BROKER', 'PRICING DATE'
]

PUNCH_OUT = OUTPUT_DIR / "msrb_punch_sheet.xlsx"

MSRB_PUNCH_OUT = OUTPUT_DIR / "msrb_punch_sheet.xlsx"
MSRB_APPEND_TEMPLATE_OUT = OUTPUT_DIR / "msrb_reference_append_template.xlsx"
MSRB_REFERENCE_UPDATED_OUT = OUTPUT_DIR / "msrb_reference_UPDATED.xlsx"
