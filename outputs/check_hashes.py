"""Check that every published SHA-256 value is a full 64-character hex string.

Scans JSON values whose key contains "sha256", plus any 60-70 character hex
sequence in text files. Commit hashes (40 characters) and abbreviated hashes with
an ellipsis are intentionally ignored.
"""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
HEX_RUN = re.compile(r'(?<![0-9a-fA-F])[0-9a-fA-F]{60,70}(?![0-9a-fA-F])')
TEXT_SUFFIXES = {'.md', '.py', '.ps1', '.json', '.yml', '.yaml', '.txt'}


def is_sha256(value):
    return len(value) == 64 and re.fullmatch(r'[0-9a-fA-F]{64}', value) is not None


def walk_json(obj, path, problems):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if 'sha256' in str(key).lower() and isinstance(value, str):
                if not is_sha256(value):
                    problems.append(f'{path}: key {key!r} has {len(value)} chars: {value!r}')
            walk_json(value, path, problems)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            walk_json(value, f'{path}[{index}]', problems)


def main():
    problems = []
    checked = 0
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '.git' in path.parts or 'work' in path.parts:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding='utf-8-sig')
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(ROOT)

        if path.suffix.lower() == '.json':
            try:
                walk_json(json.loads(text), str(rel), problems)
            except json.JSONDecodeError as exc:
                problems.append(f'{rel}: invalid JSON: {exc}')
            checked += 1

        for match in HEX_RUN.finditer(text):
            checked += 1
            if not is_sha256(match.group(0)):
                line = text.count('\n', 0, match.start()) + 1
                problems.append(f'{rel}:{line}: {len(match.group(0))}-char hex run: {match.group(0)!r}')

    if problems:
        print(f'Found {len(problems)} hash problem(s):')
        for problem in problems:
            print(f'  {problem}')
        return 1
    print(f'SHA-256 values OK ({checked} values/runs checked)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
