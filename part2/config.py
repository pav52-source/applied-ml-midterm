from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'part2-data'
OUTPUT = Path(__file__).resolve().parent / 'outputs'
SEED = 5785
PHASES = ('I', 'II', 'III')
TEXT_SETS = {
    'clinical': ('diseases_cleaned', 'drugs', 'icdcodes'),
    'criteria': ('criteria',),
    'all_text': ('criteria', 'diseases_cleaned', 'drugs', 'icdcodes'),
}
CS = (0.1, 1.0, 10.0)
THRESHOLD = 0.5
