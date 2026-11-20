"""Validate that every JSON file in outputs/ is strict UTF-8 JSON."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
failures = []
count = 0
for path in sorted((ROOT / 'outputs').rglob('*.json')):
    count += 1
    try:
        json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        failures.append(f'{path}: {exc}')
if failures:
    print(f'Found {len(failures)} invalid JSON file(s):')
    for failure in failures:
        print(f'  {failure}')
    sys.exit(1)
print(f'JSON OK ({count} files)')
