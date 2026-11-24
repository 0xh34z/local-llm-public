#!/usr/bin/env python3
"""Cross-platform entry point for the local llama.cpp server (Windows & Linux).

Usage:
    python llm.py                         # start the server
    python llm.py --download-model        # download the recommended model into models/
    python llm.py --update                # install newest ROCm release, smoke-test, activate
    python llm.py --update --tag b12000   # install a specific release
    python llm.py --stop                  # stop the running server when idle
    python llm.py --slot save             # save slot 0 to work/slot-cache/pi-session.bin
    python llm.py --slot restore          # restore slot 0 from work/slot-cache/pi-session.bin
    python llm.py --list-devices          # list detected GPU devices
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

IS_WIN = sys.platform == 'win32'
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / 'models'
WORK_DIR = ROOT / 'work'
BACKENDS_DIR = WORK_DIR / 'backends'
CURRENT_FILE = BACKENDS_DIR / 'current.txt'
SERVER_BIN = 'llama-server.exe' if IS_WIN else 'llama-server'

# File name of a specific model in models/. Leave empty to auto-select the only .gguf there.
MODEL_NAME = ''


def resolve_model(explicit: str | None) -> Path | None:
    """Resolve the GGUF model path from --model, LLM_MODEL_PATH, or models/."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    if env_path := os.environ.get('LLM_MODEL_PATH'):
        return Path(env_path).expanduser().resolve()
    if MODEL_NAME:
        return (MODELS_DIR / MODEL_NAME).resolve()
    ggufs = sorted(MODELS_DIR.glob('*.gguf')) if MODELS_DIR.is_dir() else []
    if len(ggufs) == 1:
        return ggufs[0].resolve()
    if len(ggufs) > 1:
        sys.exit(f'Several .gguf files in {MODELS_DIR}; pass --model or set MODEL_NAME in llm.py.')
    return None


def resolve_server(backend_dir: str | None) -> Path:
    """Find llama-server executable from --backend-dir, current.txt, or PATH."""
    if backend_dir:
        exe = Path(backend_dir).expanduser().resolve() / SERVER_BIN
        if not exe.is_file():
            sys.exit(f'Missing server binary: {exe}')
        return exe
    if CURRENT_FILE.is_file():
        rel = CURRENT_FILE.read_text(encoding='utf-8').strip()
        if rel:
            exe = BACKENDS_DIR / rel / SERVER_BIN
            if exe.is_file():
                return exe
    fallback = BACKENDS_DIR / 'llama-b11888-bin-win-rocm-10.0-x64' / SERVER_BIN
    if fallback.is_file():
        return fallback
    if on_path := shutil.which(SERVER_BIN):
        return Path(on_path)
    sys.exit(
        f'No {SERVER_BIN} found in {BACKENDS_DIR} or PATH.\n'
        'Run `python llm.py --update`, pass --backend-dir, or install llama-server on PATH.'
    )


def build_runtime_env() -> dict[str, str]:
    """Prepend isolated ROCm runtime libraries if present in work/rocm10-runtime."""
    env = os.environ.copy()
    venv = WORK_DIR / 'rocm10-runtime'
    if IS_WIN:
        site = venv / 'Lib' / 'site-packages'
        lib_dirs = [site / '_rocm_sdk_core' / 'bin', site / '_rocm_sdk_libraries' / 'bin']
        var = 'PATH'
    else:
        sites = list((venv / 'lib').glob('python*/site-packages')) if (venv / 'lib').is_dir() else []
        lib_dirs = [sites[0] / '_rocm_sdk_core' / 'lib', sites[0] / '_rocm_sdk_libraries' / 'lib'] if sites else []
        var = 'LD_LIBRARY_PATH'
    existing = [str(d) for d in lib_dirs if d.is_dir()]
    if existing:
        env[var] = os.pathsep.join(existing + ([env[var]] if env.get(var) else []))
    return env


def pid_file(port: int) -> Path:
    return WORK_DIR / f'llama-server.{port}.pid'


def http_json(url: str, method: str = 'GET', body: dict | None = None, timeout: int = 20):
    data = json.dumps(body).encode('utf-8') if body is not None else None
    headers = {'Content-Type': 'application/json'} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def stop_server(port: int) -> None:
    """Stop a running server on `port` if all slots are idle."""
    base_url = f'http://127.0.0.1:{port}'
    try:
        slots = http_json(f'{base_url}/slots', timeout=10)
        if any(s.get('is_processing') for s in slots):
            sys.exit('Server is currently processing a request. Wait for it to finish before stopping.')
    except urllib.error.URLError:
        pass
    pf = pid_file(port)
    if not pf.is_file():
        print(f'No PID file found at {pf}.')
        return
    pid = int(pf.read_text(encoding='utf-8').strip())
    try:
        if IS_WIN:
            subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], check=True, capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
        print(f'Stopped server PID {pid} on port {port}.')
    except (ProcessLookupError, subprocess.CalledProcessError):
        print(f'Server PID {pid} is no longer running.')
    finally:
        pf.unlink(missing_ok=True)


def manage_slot(action: str, name: str, port: int) -> None:
    """Save or restore slot 0 on a running server."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        sys.exit('Slot name must match [A-Za-z0-9_-]+.')
    slot_dir = WORK_DIR / 'slot-cache'
    slot_dir.mkdir(parents=True, exist_ok=True)
    slot_file = f'{name}.bin'
    meta_file = slot_dir / f'{name}.json'
    base_url = f'http://127.0.0.1:{port}'

    slots = http_json(f'{base_url}/slots', timeout=20)
    if len(slots) != 1 or any(s.get('is_processing') for s in slots):
        sys.exit('Slot save/restore requires one idle slot.')

    props = http_json(f'{base_url}/props', timeout=20)
    model_path = Path(props['model_path'])
    stat = model_path.stat()
    fingerprint = {
        'model_path': str(model_path),
        'model_bytes': stat.st_size,
        'model_mtime_ns': stat.st_mtime_ns,
        'build_info': props.get('build_info'),
        'n_ctx': props.get('default_generation_settings', {}).get('n_ctx'),
    }
    if action == 'restore':
        if not meta_file.is_file():
            sys.exit(f'Snapshot metadata missing: {meta_file}')
        saved = json.loads(meta_file.read_text(encoding='utf-8'))
        if saved.get('fingerprint') != fingerprint:
            sys.exit('Model, build, or context differs from the saved slot.')

    t0 = time.monotonic()
    resp = http_json(
        f'{base_url}/slots/0?action={action}',
        method='POST',
        body={'filename': slot_file},
        timeout=600,
    )
    elapsed = round(time.monotonic() - t0, 3)
    if action == 'save':
        meta_file.write_text(json.dumps({
            'fingerprint': fingerprint,
            'filename': slot_file,
            'server_response': resp,
        }, indent=2), encoding='utf-8')
    print(json.dumps({'action': action, 'name': name, 'seconds': elapsed, 'file': str(slot_dir / slot_file)}, indent=2))


def run_server(args: argparse.Namespace, model_path: Path | None) -> int:
    """Launch llama-server and tee output to log."""
    exe = resolve_server(args.backend_dir)
    env = build_runtime_env()
    if args.kv_split:
        env['GGML_HIP_FA_KV_SPLIT'] = args.kv_split
    elif 'kv-split' in exe.parent.name and 'GGML_HIP_FA_KV_SPLIT' not in env:
        env['GGML_HIP_FA_KV_SPLIT'] = 'double'
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if args.list_devices:
        log_path = Path(args.log_path) if args.log_path else ROOT / 'outputs' / 'rocm-devices.log'
        cmd = [str(exe), '-lv', '4', '--list-devices']
    else:
        if not model_path or not model_path.is_file():
            sys.exit(
                f'No model found in {MODELS_DIR}.\n'
                'Run `python llm.py --download-model`, or pass --model <path>.'
            )
        draft_threads = args.draft_threads or args.threads
        slot_dir = WORK_DIR / 'slot-cache'
        slot_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(args.log_path) if args.log_path else ROOT / 'outputs' / 'rocm-server.log'
        cmd = [
            str(exe),
            '-m', str(model_path), '--device', args.device,
            '-c', str(args.context), '-np', '1', '-ngl', '99', '-fa', 'on', '--jinja',
            '-ctk', args.kv_cache, '-ctv', args.kv_cache,
            '-b', str(args.batch_size), '-ub', str(args.micro_batch_size),
            '-t', str(args.threads), '-tb', str(args.batch_threads),
            '-td', str(draft_threads), '-tbd', str(args.batch_threads),
            '--port', str(args.port), '-lm', 'auto', '-lv', str(args.log_verbosity),
            '--slot-save-path', str(slot_dir),
        ]
        use_mtp = args.speculative == 'mtp' or (args.speculative == 'auto' and 'MTP' in model_path.name)
        if use_mtp:
            cmd += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', str(args.draft_tokens)]
        print(f'Starting {exe.name} on {args.device} (port {args.port}). Log: {log_path}')

    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not args.list_devices:
        pid_file(args.port).write_text(str(proc.pid), encoding='utf-8')
    try:
        with open(log_path, 'w', encoding='utf-8') as log_f:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log_f.write(line)
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return proc.wait()
    finally:
        if not args.list_devices:
            pid_file(args.port).unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--download-model', action='store_true', help='download the recommended model into models/')
    ap.add_argument('--update', action='store_true', help='install/update the llama.cpp backend')
    ap.add_argument('--tag', help='release tag for --update')
    ap.add_argument('--gfx', default='gfx1152', help='ROCm GPU target for --update (default: gfx1152)')
    ap.add_argument('--skip-test', action='store_true', help='skip smoke test during --update')
    ap.add_argument('--stop', action='store_true', help='stop the server on --port when idle')
    ap.add_argument('--slot', choices=['save', 'restore'], help='save or restore slot 0')
    ap.add_argument('--name', default='pi-session', help='slot snapshot name (default: pi-session)')
    ap.add_argument('--list-devices', action='store_true', help='list detected GPU devices')
    ap.add_argument('--model', help='path to .gguf model file')
    ap.add_argument('--backend-dir', help='directory containing llama-server')
    ap.add_argument('--device', default='ROCm0', help='llama.cpp device (default: ROCm0)')
    ap.add_argument('--port', type=int, default=9931)
    ap.add_argument('--context', type=int, default=163840)
    ap.add_argument('--kv-cache', choices=['f16', 'q8_0'], default='f16')
    ap.add_argument('--kv-split', choices=['auto', 'double', 'half', 'quad'], help='override GGML_HIP_FA_KV_SPLIT (defaults to double for *-kv-split backends)')
    ap.add_argument('--speculative', choices=['auto', 'mtp', 'off'], default='auto')
    ap.add_argument('--draft-tokens', type=int, choices=[1, 2, 3], default=2)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--micro-batch-size', type=int, default=512)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--batch-threads', type=int, default=8)
    ap.add_argument('--draft-threads', type=int, default=0)
    ap.add_argument('--log-verbosity', type=int, default=3)
    ap.add_argument('--log-path', help='server log file path')
    args = ap.parse_args()

    if args.download_model:
        raise SystemExit(subprocess.call([sys.executable, str(ROOT / 'outputs' / 'download_model.py')]))

    model_path = resolve_model(args.model)
    if args.update:
        cmd = [sys.executable, str(ROOT / 'outputs' / 'update_backend.py'), '--gfx', args.gfx]
        if args.tag:
            cmd += ['--tag', args.tag]
        if model_path and model_path.is_file() and not args.skip_test:
            cmd += ['--model', str(model_path)]
        else:
            cmd += ['--skip-test']
        raise SystemExit(subprocess.call(cmd))

    if args.stop:
        stop_server(args.port)
        return

    if args.slot:
        manage_slot(args.slot, args.name, args.port)
        return

    raise SystemExit(run_server(args, model_path))


if __name__ == '__main__':
    main()
