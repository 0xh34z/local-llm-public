"""Compare Windows AC power modes on the existing eight-block ROCm runtime."""
import argparse
import ctypes
import hashlib
import json
import time
import uuid
from pathlib import Path

from benchmark_server import ROOT, TOOLS, server
from run_fine_split import CASES, EXPECTED
from run_thread_sweep import get, processes
import run_mtp_benchmark as probe


BEST_PERFORMANCE = uuid.UUID('ded574b5-45a0-4f42-8737-46345c09c238')
GUID = ctypes.c_ubyte * 16
power = ctypes.WinDLL('powrprof')


def current_mode():
    data = GUID()
    status = power.PowerGetUserConfiguredACPowerMode(ctypes.byref(data))
    if status:
        raise ctypes.WinError(status)
    return uuid.UUID(bytes_le=bytes(data))


def apply_mode(mode):
    if current_mode() != mode:
        data = GUID.from_buffer_copy(mode.bytes_le)
        status = power.PowerSetUserConfiguredACPowerMode(ctypes.byref(data))
        if status:
            raise ctypes.WinError(status)
    if current_mode() != mode:
        raise RuntimeError(f'AC mode selection failed for {mode}')


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if processes():
        raise RuntimeError('A server is already running on the benchmark port')
    original = current_mode()
    if original == BEST_PERFORMANCE:
        raise RuntimeError('Best performance is already selected')
    out.mkdir(exist_ok=False)
    order = (original, BEST_PERFORMANCE, BEST_PERFORMANCE, original)
    save(out/'manifest.json', dict(original_ac_mode=str(original), order=list(map(str, order)),
                                  backend='work/hip-kv-split-bin', microbatch=1024))
    samples = []
    try:
        for index, mode in enumerate(order, 1):
            apply_mode(mode)
            case = out/f'case-{index}'
            with server(case, backend=ROOT/'work/hip-kv-split-bin', launcher=TOOLS/'Start-KVSplitServer.ps1',
                        arguments=['-MicroBatchSize', 1024], expected={**EXPECTED, '-ub': 1024}) as props:
                if props['default_generation_settings']['n_ctx'] != 163840:
                    raise RuntimeError('Context mismatch')
                probe.OUT = case
                for label, length, budget in [('warmup', '103k', 48), ('first', '103k', 256),
                                              ('second', '160k', 256)]:
                    slot, cache_n, request = CASES[length]
                    prompt = json.loads(request.read_text())['prompt']
                    if any(s['is_processing'] for s in get('/slots')):
                        raise RuntimeError('An inference request is already active')
                    restored = probe.post('/slots/0?action=restore', {'filename': slot})
                    save(case/(label+'-restore.json'), restored)
                    if restored['n_restored'] != cache_n:
                        raise RuntimeError('Unexpected saved context')
                    started = time.time()
                    result = probe.run(label, prompt, True, budget)
                    ended = time.time()
                    timing = result['final']['timings']
                    if (timing['cache_n'], timing['prompt_n']) != (cache_n, len(prompt) - cache_n):
                        raise RuntimeError('Unexpected evaluated prefix or suffix')
                    if label != 'warmup':
                        token_ids = json.dumps(result['generated_token_ids']).encode()
                        samples.append(dict(case=index, ac_mode=str(mode), length=len(prompt),
                                            timings=timing, started_unix=started, ended_unix=ended,
                                            token_ids_sha256=hashlib.sha256(token_ids).hexdigest()))
                        save(out/'samples.json', samples)
            log = (case/'server.log').read_bytes()
            if 'HIP_FA_KV_SPLIT mode=double blocks=8' not in log.decode(
                    'utf-16' if log.startswith(b'\xff\xfe') else 'utf-8'):
                raise RuntimeError('Eight-block dispatch missing')
    finally:
        apply_mode(original)
        save(out/'restored.json', {'current_ac_mode': str(current_mode())})


if __name__ == '__main__':
    main()
