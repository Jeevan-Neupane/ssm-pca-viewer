import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DATASET1_DIR = os.path.join(DATA_DIR, 'dataset1')
DATASET2_DIR = os.path.join(DATA_DIR, 'dataset2')

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
