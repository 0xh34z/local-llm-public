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
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

IS_WIN = sys.platform == 'win32'
API = 'https://api.github.com/repos/ggml-org/llama.cpp/releases'
PATTERNS = (
    ['llama-*-bin-win-rocm-*-x64.zip']
    if IS_WIN
    else ['llama-*-bin-ubuntu-rocm-*-x64.tar.gz', 'llama-*-bin-ubuntu-rocm-*-x64.zip']
)
SERVER_BIN = 'llama-server.exe' if IS_WIN else 'llama-server'
ROOT = Path(__file__).resolve().parent.parent
BACKENDS = ROOT / 'work' / 'backends'
CURRENT = BACKENDS / 'current.txt'
UA = {'User-Agent': 'local-llm-updater', 'Accept': 'application/vnd.github+json'}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def rocm_assets(release):
    return [
        a for a in release.get('assets', [])
        if any(fnmatch.fnmatch(a['name'], pat) for pat in PATTERNS)
    ]


def find_release(tag):
    if tag:
        releases = [get_json(f'{API}/tags/{tag}')]
    else:
        # /releases/latest can point at a non-build placeholder, and upstream flags every
        # build as a pre-release, so take the newest non-draft release that ships the asset.
        releases = [r for r in get_json(f'{API}?per_page=50') if not r.get('draft')]
    for release in releases:
        if rocm_assets(release):
            return release
    sys.exit(
        f'No release with a ROCm asset ({", ".join(PATTERNS)}) found'
        + (f' for tag {tag}.' if tag else '.')
    )


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


def rocm_lib_dirs():
    venv = ROOT / 'work' / 'rocm10-runtime'
    if IS_WIN:
        site = venv / 'Lib' / 'site-packages'
        return [site / '_rocm_sdk_core' / 'bin', site / '_rocm_sdk_libraries' / 'bin']
    lib_candidates = list((venv / 'lib').glob('python*/site-packages'))
    if not lib_candidates:
        return []
    site = lib_candidates[0]
    return [p for p in (site / '_rocm_sdk_core' / 'lib', site / '_rocm_sdk_libraries' / 'lib') if p.is_dir()]


def ensure_runtime(gfx):
    """Create the isolated ROCm runtime venv (BUILD.md step 1) if it is missing."""
    venv = ROOT / 'work' / 'rocm10-runtime'
    if rocm_lib_dirs() and all(d.is_dir() for d in rocm_lib_dirs()):
        return
    print('ROCm runtime missing; installing into work/rocm10-runtime (large download) ...')
    subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    py = str(venv / ('Scripts/python.exe' if IS_WIN else 'bin/python'))
    subprocess.run([py, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
    subprocess.run([py, '-m', 'pip', 'install', '--index-url', 'https://stable.repo.amd.com/rocm/whl-next/',
                    'rocm==10.0.0', 'rocm-sdk-core==10.0.0', 'rocm-sdk-libraries==10.0.0',
                    f'rocm-sdk-device-{gfx}==10.0.0'], check=True)


def find_server(directory):
    return next(directory.rglob(SERVER_BIN), None)


def extract_atomic(archive_path, dest):
    tmp = dest.with_name(dest.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        if archive_path.name.endswith('.zip'):
            with zipfile.ZipFile(archive_path) as z:
                z.extractall(tmp)
        else:
            with tarfile.open(archive_path, 'r:*') as t:
                t.extractall(tmp)
        if not find_server(tmp):
            sys.exit(f'{SERVER_BIN} not found after extracting {archive_path.name}')
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def smoke_test(server_dir, model, port, timeout):
    print(f'Smoke test on port {port} ...')
    log = ROOT / 'work' / 'update-smoke-test.log'
    slot_dir = ROOT / 'work' / 'slot-cache-test'
    slot_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    libs = [str(d) for d in rocm_lib_dirs() if d.is_dir()]
    if libs:
        var = 'PATH' if IS_WIN else 'LD_LIBRARY_PATH'
        env[var] = os.pathsep.join(libs + ([env[var]] if env.get(var) else []))
    cmd = [
        str(server_dir / SERVER_BIN),
        '-m', str(model), '--device', 'ROCm0',
        '-c', '4096', '-np', '1', '-ngl', '99', '-fa', 'on', '--jinja',
        '-ctk', 'f16', '-ctv', 'f16', '-b', '2048', '-ub', '512',
        '--port', str(port), '-lm', 'auto', '-lv', '3',
        '--slot-save-path', str(slot_dir),
    ]
    if 'MTP' in Path(model).name:
        cmd += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '2']
    with open(log, 'w', encoding='utf-8') as log_f:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if IS_WIN else 0,
        )
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
                if IS_WIN:
                    subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
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

    archive_name = asset['name']
    folder_name = archive_name[:-len('.tar.gz')] if archive_name.endswith('.tar.gz') else archive_name.removesuffix('.zip')
    dest = BACKENDS / folder_name
    if not (dest.exists() and find_server(dest)):
        archive_path = ROOT / 'work' / archive_name
        download(asset, archive_path)
        extract_atomic(archive_path, dest)
        archive_path.unlink()
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
