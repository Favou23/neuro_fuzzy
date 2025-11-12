"""Quick smoke test to validate package imports and IEC duval classification.

Run this from the repository root (the directory that contains `fuzzy_logic`).
"""
import sys
import os

# Ensure fuzzy_logic package is importable when running from repo root
sys.path.insert(0, os.path.abspath('.'))

from . import iec_rule_based

def main():
    print('Module imported successfully: iec_rule_based')
    # sample ppm values
    sample = {'CH4': 100.0, 'C2H4': 50.0, 'C2H2': 25.0}
    zone = iec_rule_based.duval_polygon_classify(sample['CH4'], sample['C2H4'], sample['C2H2'])
    print(f'Duval polygon classification for sample {sample} -> {zone}')

if __name__ == '__main__':
    main()
