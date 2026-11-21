#!/usr/bin/env python3
"""Install an official llama.cpp Windows ROCm release into work/backends and make it active.

Usage:
    python outputs/update_backend.py                 # newest release
    python outputs/update_backend.py --tag b12000    # specific release
    python outputs/update_backend.py --model models/x.gguf   # smoke-test before switching

The active backend is recorded in work/backends/current.txt (path relative to work/backends),
which Start-ROCmServer.ps1 reads. Old versions stay on disk for rollback.
"""
import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

API = 'https://api.github.com/repos/ggml-org/llama.cpp/releases'
PATTERN = 'llama-*-bin-win-rocm-*-x64.zip'
ROOT = Path(__file__).resolve().parent.parent
BACKENDS = ROOT / 'work' / 'backends'
CURRENT = BACKENDS / 'current.txt'
UA = {'User-Agent': 'local-llm-updater', 'Accept': 'application/vnd.github+json'}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def rocm_assets(release):
    return [a for a in release.get('assets', []) if fnmatch.fnmatch(a['name'], PATTERN)]


def find_release(tag):
    if tag:
        releases = [get_json(f'{API}/tags/{tag}')]
    else:
        # /releases/latest can point at a non-build placeholder, and upstream flags every
        # build as a pre-release, so take the newest non-draft release that ships the zip.
        releases = [r for r in get_json(f'{API}?per_page=50') if not r.get('draft')]
    for release in releases:
        if rocm_assets(release):
            return release
    sys.exit(f'No release with a Windows ROCm asset ({PATTERN}) found' + (f' for tag {tag}.' if tag else '.'))


def download(asset, dest_zip):
    print(f"Downloading {asset['browser_download_url']}")
    req = urllib.request.Request(asset['browser_download_url'], headers={'User-Agent': UA['User-Agent']})
    sha = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=60) as r, open(dest_zip, 'wb') as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            sha.update(chunk)
    digest = asset.get('digest') or ''
    if digest.startswith('sha256:') and sha.hexdigest().lower() != digest[7:].lower():
        dest_zip.unlink()
        sys.exit(f"SHA256 mismatch for {asset['name']}")


def ensure_runtime(gfx):
    """Create the isolated ROCm runtime venv (BUILD.md step 1) if it is missing."""
    venv = ROOT / 'work' / 'rocm10-runtime'
    if (venv / 'Lib' / 'site-packages' / '_rocm_sdk_core' / 'bin').is_dir():
        return
    print('ROCm runtime missing; installing into work/rocm10-runtime (large download) ...')
    subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    py = str(venv / 'Scripts' / 'python.exe')
    subprocess.run([py, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
    subprocess.run([py, '-m', 'pip', 'install', '--index-url', 'https://stable.repo.amd.com/rocm/whl-next/',
                    'rocm==10.0.0', 'rocm-sdk-core==10.0.0', 'rocm-sdk-libraries==10.0.0',
                    f'rocm-sdk-device-{gfx}==10.0.0'], check=True)


def find_server(directory):
    return next(directory.rglob('llama-server.exe'), None)


def smoke_test(server_dir, model, port, timeout):
    print(f'Smoke test on port {port} ...')
    log = ROOT / 'work' / 'update-smoke-test.log'
    cmd = ['powershell', '-NoProfile', '-File', str(ROOT / 'outputs' / 'Start-ROCmServer.ps1'),
           '-BackendDirectory', str(server_dir), '-ModelPath', str(model), '-Port', str(port),
           '-Context', '4096', '-SlotSavePath', str(ROOT / 'work' / 'slot-cache-test'), '-LogPath', str(log)]
    proc = subprocess.Popen(cmd, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    ok = False
    try:
        deadline = time.time() + timeout
        while time.time() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as r:
                    if r.status == 200:
                        ok = True
                        break
            except OSError:
                pass
            time.sleep(3)
    finally:
        if proc.poll() is None:
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True)
    return ok, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', help='release tag to install (default: newest)')
    ap.add_argument('--model', help='model used for the smoke test')
    ap.add_argument('--skip-test', action='store_true')
    ap.add_argument('--gfx', default='gfx1152', help='GPU target for the ROCm device package (default: gfx1152, Radeon 860M)')
    ap.add_argument('--test-port', type=int, default=9939)
    ap.add_argument('--test-timeout', type=int, default=300, help='seconds to wait for /health')
    args = ap.parse_args()

    BACKENDS.mkdir(parents=True, exist_ok=True)
    ensure_runtime(args.gfx)
    release = find_release(args.tag)
    assets = rocm_assets(release)
    asset = next((a for a in assets if 'rocm-10.0' in a['name']), assets[0])  # match repo's ROCm 10.0 runtime
    print(f"Release {release['tag_name']} -> {asset['name']}")

    dest = BACKENDS / asset['name'].removesuffix('.zip')
    if not (dest.exists() and find_server(dest)):
        zip_path = ROOT / 'work' / asset['name']
        download(asset, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        zip_path.unlink()
    server = find_server(dest)
    if not server:
        sys.exit(f"llama-server.exe not found after extracting {asset['name']}")
    relative = str(server.parent.relative_to(BACKENDS))

    previous = CURRENT.read_text().strip() if CURRENT.exists() else None
    if not args.skip_test:
        if not args.model or not Path(args.model).is_file():
            sys.exit('Smoke test needs a model: pass --model, or --skip-test.')
        ok, log = smoke_test(server.parent, args.model, args.test_port, args.test_timeout)
        if not ok:
            print(f'Smoke test FAILED for {release["tag_name"]}; keeping previous backend ({previous}). See {log}')
            sys.exit(1)
        print('Smoke test passed.')

    CURRENT.write_text(relative + '\n')
    print(f'Active backend: {relative}')
    if previous and previous != relative:
        print(f'Previous: {previous} (roll back with --tag, or edit work/backends/current.txt)')


if __name__ == '__main__':
    main()
