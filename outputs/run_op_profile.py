"""Diagnostic op timings, HIP vs Vulkan, for one restored-103K MTP 2 continuation per backend.

Both timers synchronize around operations (HIP: GGML_HIP_OP_PROFILE, no graph replay; Vulkan: GGML_VK_PERF_LOGGER),
so absolute speed is distorted. Use the per-operation sums only to compare where each backend spends GPU time.
"""
import argparse
import json
from pathlib import Path

from benchmark_server import server
from run_ceiling_server import BACKENDS, LAUNCHER, STOPPER, parse_case
from run_fine_split import CASES, read, save
from run_thread_sweep import get
import run_mtp_benchmark as probe

TOKENS = 48


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--backends', default='hip,vk', help='hip, vk or x-NAME (work/hip-mmvq/NAME/bin)')
    p.add_argument('--mode', default='1', choices=('1', '2'), help='HIP op profiler mode')
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    filename, cache_n, request = CASES['103k']
    prompt = read(request)['prompt']
    for name in args.backends.split(','):
        if name.startswith('x-'):
            parse_case(f'{name}:2')  # registers the variant runtime
        directory, device, _ = BACKENDS[name]
        case = out/name
        with server(case, ['-DraftTokens', 2, '-Device', device, '-OpTiming', '-OpProfileMode', args.mode], backend=directory,
                    launcher=LAUNCHER, stopper=STOPPER):
            probe.OUT = case
            if any(slot['is_processing'] for slot in get('/slots')):
                raise RuntimeError('Server busy')
            restore = probe.post('/slots/0?action=restore', {'filename': filename})
            save(case/'restore.json', restore)
            if restore['n_restored'] != cache_n:
                raise RuntimeError('Unexpected restored prefix')
            result = probe.run('profile-103k', prompt, True, TOKENS, temperature=0.0)
            t = result['final']['timings']
            save(case/'profile-summary.json', dict(backend=name, cache_n=t['cache_n'], prompt_n=t['prompt_n'],
                                                  predicted_n=t['predicted_n'], predicted_ms=t['predicted_ms'],
                                                  draft=t.get('draft_n'), accepted=t.get('draft_n_accepted')))
            print(json.dumps({'event': 'profiled', 'backend': name, 'tps': t['predicted_per_second']}), flush=True)


if __name__ == '__main__':
    main()
