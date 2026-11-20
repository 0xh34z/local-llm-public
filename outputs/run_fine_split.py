"""Screen private gfx1152 attention splits, then compare restored long-context generation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from benchmark_server import ROOT, TOOLS, server
from run_thread_sweep import get, processes
from screen_kv_split import NUMPY_PYTHON
import run_mtp_benchmark as probe


BACKEND = ROOT / 'work/hip-kv-fine/bin'
SPLITS = (8, 6, 10, 12)
VARIANTS = ('contiguous100k', 'strided100k', 'contiguous160k', 'strided160k')
ORDERS = ((8, 6, 10, 12), (12, 10, 6, 8), (10, 8, 12, 6))
CASES = {
    '103k': ('synthetic-helper-check.bin', 103194,
             TOOLS / 'kv-split-100k-results-verified/case-02-double/repeat-1-request.json'),
    '160k': ('synthetic-160k-kv-split.bin', 160511,
             TOOLS / 'populated-160k-kv-split-results/case-02-double/measured-request.json'),
}
EXPECTED = {'-ctk': 'f16', '-ctv': 'f16', '-c': 163840, '-np': 1, '-b': 2048, '-ub': 512,
            '-t': 8, '-td': 8, '-tb': 8, '-tbd': 8, '--device': 'ROCm0', '-ngl': 99, '-fa': 'on',
            '--spec-type': 'draft-mtp', '--spec-draft-n-max': 2}
REPETITION = ('\nFor a timing test, continue the repeating sequence below indefinitely.\n'
              + ' red blue green yellow' * 48)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def mode(blocks):
    return 'double' if blocks == 8 else str(blocks)


def settings(blocks):
    return dict(GGML_HIP_FA_KV_SPLIT=mode(blocks), GGML_HIP_FA_EXPERIMENT='auto',
                GGML_HIP_FA_MMA_TILE='auto', GGML_HIP_OP_PROFILE='0')


def digest():
    return hashlib.sha256((BACKEND/'ggml-hip.dll').read_bytes()).hexdigest()


def audit(log, blocks):
    marker = f'HIP_FA_KV_SPLIT mode={mode(blocks)} blocks={blocks} '
    if marker not in log or 'HIP_FA_MMA_TILE columns=32 ' not in log:
        raise RuntimeError(f'Missing split or tile marker for {blocks} blocks')


def execute_kernel(out, action, variant, backend, configuration, tag, dump=None):
    env = os.environ.copy()
    env.update(configuration)
    env.pop('GGML_BACKEND_PATH', None)
    env['LLAMA_HIP_ATTENTION_PROBE'] = '1'
    sdk = ROOT / 'work/rocm10-runtime/Lib/site-packages'
    env['PATH'] = os.pathsep.join(map(str, (backend, sdk/'_rocm_sdk_core/bin',
                                             sdk/'_rocm_sdk_libraries/bin', env['PATH'])))
    if variant == 'contiguous100k':
        env.pop('LLAMA_HIP_ATTENTION_VARIANT', None)
    else:
        env['LLAMA_HIP_ATTENTION_VARIANT'] = variant
    env.pop('LLAMA_HIP_ATTENTION_DUMP', None)
    if dump:
        dump.mkdir()
        env['LLAMA_HIP_ATTENTION_DUMP'] = str(dump)
    log_path = out/(tag+'.log')
    with log_path.open('w') as stream:
        result = subprocess.run([str(backend/'test-attention-probe.exe'), action, '-b', 'ROCm0',
                                 '-o', 'FLASH_ATTN_EXT'], cwd=backend, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=180)
    log = log_path.read_text()
    if str(backend/'ggml-hip.dll').lower() not in log.lower():
        raise RuntimeError(f'Unexpected probe backend, see {log_path}')
    return result.returncode, log


def kernel(out, action, variant, blocks, tag, dump=None):
    code, log = execute_kernel(out, action, variant, BACKEND, settings(blocks), tag, dump)
    audit(log, blocks)
    return code, log


def screen(out):
    dump_root = ROOT / 'work' / (out.name + '-fixtures')
    dump_root.mkdir(exist_ok=False)
    summary = {'hip_sha256': digest(), 'dump_root': str(dump_root), 'numerical': [], 'performance': []}
    if processes():
        raise RuntimeError('A server is already using the benchmark port')
    for variant in VARIANTS:
        for blocks in SPLITS:
            tag = f'{variant}-{blocks}'
            dump = dump_root/tag
            code, _ = kernel(out, 'test', variant, blocks, tag+'-test', dump)
            if code not in (0, 1):
                raise RuntimeError(f'Unexpected probe exit code {code} for {tag}')
            with (out/(tag+'-oracle.log')).open('w') as stream:
                subprocess.run([str(NUMPY_PYTHON), str(TOOLS/'check_attention_float64.py'),
                                '--dump-dir', str(dump), '--out', str(out/(tag+'.json'))],
                               stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180)
            result = read(out/(tag+'.json'))
            gpu = result['backends']['gpu']['nmse_vs_float64']
            expected_kv = 160000 if '160k' in variant else 103424
            ok = (math.isfinite(gpu) and gpu <= 0.0005 and all(result['inputs_equal'].values())
                  and result['shapes']['k'] == [256, expected_kv, 2, 1])
            row = dict(variant=variant, blocks=blocks, gpu_nmse=gpu,
                       cpu_nmse=result['backends']['cpu']['nmse_vs_float64'],
                       upstream_exit=code, qualified=ok)
            summary['numerical'].append(row)
            save(out/'summary.json', summary)
            print(row, flush=True)
    allowed = qualified(summary)
    for variant in ('strided100k', 'strided160k'):
        for repeat, order in enumerate(ORDERS, 1):
            for blocks in order:
                if blocks not in allowed:
                    continue
                tag = f'{variant}-{blocks}-perf-{repeat}'
                code, log = kernel(out, 'perf', variant, blocks, tag)
                times = re.findall(r'FLASH_ATTN_EXT\([^\n]*nb=3,[^\n]*?([\d.]+) us/run', log)
                if code or len(times) != 1:
                    raise RuntimeError(f'Expected one regular Q3 timing, see {tag}')
                row = dict(variant=variant, repeat=repeat, blocks=blocks, us=float(times[0]))
                summary['performance'].append(row)
                save(out/'summary.json', summary)
                print(row, flush=True)


def qualified(summary):
    return {blocks for blocks in SPLITS if {r['variant'] for r in summary['numerical']
            if r['blocks'] == blocks and r['qualified']} == set(VARIANTS)}


def restored_workloads(case, props, baseline):
    if (props['model_path'], props['build_info'], props['default_generation_settings']['n_ctx']) != (
            baseline['model_path'], baseline['build_info'], baseline['n_ctx']):
        raise RuntimeError('Snapshot/model mismatch')
    probe.OUT = case
    suffix = probe.tokenize(REPETITION)
    for trial, name in enumerate(('103k', '103k', '160k', '160k', '103k', '103k', '160k')):
        filename, cache_n, request = CASES[name]
        prompt = read(request)['prompt']
        repeated = trial >= 5
        if repeated:
            prompt += suffix
        tag = f'{trial}-{name}'
        if any(slot['is_processing'] for slot in get('/slots')):
            raise RuntimeError('Server busy')
        restore = probe.post('/slots/0?action=restore', {'filename': filename})
        save(case/(tag+'-restore.json'), restore)
        if restore['n_restored'] != cache_n:
            raise RuntimeError('Unexpected restored prefix')
        count = 128 if repeated else (48 if trial == 0 else 256)
        started = time.time()
        result = probe.run(tag, prompt, True, count, temperature=0.0 if repeated else 0.6)
        ended = time.time()
        timing = result['final']['timings']
        if (timing['cache_n'], timing['prompt_n']) != (cache_n, len(prompt)-cache_n):
            raise RuntimeError('Unexpected prompt work')
        if trial:
            token_bytes = json.dumps(result['generated_token_ids']).encode()
            yield dict(context=name, trial=trial, workload='repetition' if repeated else 'sampled',
                       tokens=len(prompt), cache_n=cache_n, prompt_n=timing['prompt_n'],
                       predicted_n=timing['predicted_n'], predicted_ms=timing['predicted_ms'],
                       tps=timing['predicted_per_second'], draft=timing.get('draft_n', 0),
                       accepted=timing.get('draft_n_accepted', 0), started_unix=started, ended_unix=ended,
                       ttft_s=result['time_to_first_generated_token_seconds'],
                       tokens_sha256=hashlib.sha256(token_bytes).hexdigest())


def generate(out, screen_path, order):
    summary = read(screen_path/'summary.json')
    allowed = qualified(summary)
    if digest() != summary['hip_sha256'] or not set(order) <= allowed:
        raise RuntimeError('Benchmark DLL and split modes must pass the independent screen')
    baseline = read(TOOLS/'slot-cache-reuse-results/snapshot.json')['server']
    save(out/'manifest.json', dict(order=order, screen=str(screen_path), hip_sha256=digest(),
                                   flags=EXPECTED, requests={k: str(v[2]) for k, v in CASES.items()}))
    samples = []
    for index, blocks in enumerate(order, 1):
        os.environ.update(settings(blocks))
        case = out/f'case-{index:02}-{blocks}'
        with server(case, backend=BACKEND, expected=EXPECTED) as props:
            for row in restored_workloads(case, props, baseline):
                samples.append(dict(case=index, blocks=blocks, **row))
                save(out/'samples.json', samples)
        data = (case/'server.log').read_bytes()
        audit(data.decode('utf-16' if data.startswith(b'\xff\xfe') else 'utf-8'), blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['screen', 'generate'])
    parser.add_argument('out', type=Path)
    parser.add_argument('--screen', type=Path)
    parser.add_argument('--order', type=int, nargs='+', choices=SPLITS, default=[8, 6, 8, 10, 8, 12, 8])
    args = parser.parse_args()
    if args.action == 'generate' and not args.screen:
        parser.error('Generation requires --screen')
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    if args.action == 'screen':
        screen(out)
    else:
        generate(out, args.screen.resolve(), args.order)


if __name__ == '__main__':
    main()
