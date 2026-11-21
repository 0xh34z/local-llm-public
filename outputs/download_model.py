#!/usr/bin/env python3
"""Download the recommended model into models/ (resumable).

Usage:
    python outputs/download_model.py
    python outputs/download_model.py --repo unsloth/Qwen3.6-35B-A3B-MTP-GGUF --file Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf
"""
import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = 'unsloth/Qwen3.6-35B-A3B-MTP-GGUF'
FILE = 'Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default=REPO)
    ap.add_argument('--file', default=FILE)
    args = ap.parse_args()

    dest = ROOT / 'models' / args.file
    part = dest.with_name(dest.name + '.part')
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        print(f'Already present: {dest}')
        return
    url = f'https://huggingface.co/{args.repo}/resolve/main/{args.file}'
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={'User-Agent': 'local-llm-model-download'})
    if have:
        req.add_header('Range', f'bytes={have}-')
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:  # already complete
            part.replace(dest)
            print(f'Done: {dest}')
            return
        sys.exit(f'Download failed: HTTP {e.code} for {url}')
    if have and resp.status != 206:  # server ignored Range; start over
        have = 0
    total = int(resp.headers.get('Content-Length', 0)) + have
    print(f'Downloading {url}\n -> {dest} ({total / 1e9:.1f} GB)')
    done = have
    with resp, open(part, 'ab' if have else 'wb') as f:
        while chunk := resp.read(8 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f'\r{done / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * done / total:5.1f}%)', end='', flush=True)
    print()
    part.replace(dest)
    print(f'Done: {dest}')


if __name__ == '__main__':
    main()
