"""Check local Markdown links in the top-level docs and outputs/INDEX.md.

External URLs, anchors, placeholder paths and anything under work/ are skipped.
Links inside individual reports are excluded by default; use --all to audit them.
Targets listed in outputs/private-links.txt are treated as known-private and are
skipped even with --all.
"""
from fnmatch import fnmatch
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r'\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
SKIP_MARKERS = ('<repo>', '<models>', '%USERPROFILE%', '...')
ALLOWLIST = ROOT / 'outputs' / 'private-links.txt'


def private_targets():
    if not ALLOWLIST.is_file():
        return []
    targets = []
    for line in ALLOWLIST.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            targets.append(line)
    return targets


def markdown_files(include_reports):
    yield ROOT / 'README.md'
    yield ROOT / 'BUILD.md'
    yield ROOT / 'outputs' / 'INDEX.md'
    if include_reports:
        yield from sorted((ROOT / 'outputs').rglob('*.md'))


def main():
    include_reports = '--all' in sys.argv
    allow = private_targets()
    problems = []
    checked = 0
    for source in markdown_files(include_reports):
        if not source.is_file():
            continue
        text = source.read_text(encoding='utf-8-sig', errors='replace')
        for match in LINK.finditer(text):
            target = match.group(1).strip().strip('<>')
            if (not target or target.startswith('#') or
                    target.startswith(('http://', 'https://', 'mailto:'))):
                continue
            if any(marker in target for marker in SKIP_MARKERS):
                continue
            target = target.split('#', 1)[0]
            target = unquote(target).replace('\\', '/')
            if not target or 'work/' in target:
                continue
            if any(fnmatch(target, pattern) for pattern in allow):
                continue

            checked += 1
            if target.startswith('/'):
                resolved = (ROOT / target.lstrip('/')).resolve()
            else:
                resolved = (source.parent / target).resolve()
            if not resolved.exists():
                line = text.count('\n', 0, match.start()) + 1
                problems.append(f'{source.relative_to(ROOT)}:{line}: missing {target}')

    if problems:
        print(f'Found {len(problems)} broken local link(s):')
        for problem in problems:
            print(f'  {problem}')
        return 1
    scope = 'all Markdown files' if include_reports else 'top-level docs'
    print(f'Local Markdown links OK ({checked} links checked in {scope})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
