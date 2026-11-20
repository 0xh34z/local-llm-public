"""Compare GPU workgroup size or query storage in the scoped long-context attention kernel."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import time
from typing import NamedTuple

from benchmark_server import ROOT, TOOLS, server
from run_fine_split import CASES, EXPECTED, VARIANTS, execute_kernel, read, restored_workloads, save, settings
from run_thread_sweep import processes
from screen_kv_split import NUMPY_PYTHON


class Comparison(NamedTuple):
    field: str
    modes: tuple
    backend: Path
    env_key: str

    def config(self, value):
        return {**settings(8), self.env_key: str(value)}

    def perf_order(self):
        return self.modes * 4 + self.modes[:1]


WORKGROUP = Comparison('threads', (256, 128), ROOT/'work/hip-workgroup/bin', 'GGML_HIP_FA_WORKGROUP')
QUERY = Comparison('query_storage', ('registers', 'lds'), ROOT/'work/hip-query-lds/bin', 'GGML_HIP_FA_QUERY_STORAGE')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(log, experiment, mode):
    threads = mode if experiment == WORKGROUP else 256
    marker = (f'HIP_FA_WORKGROUP threads={threads} columns=32' if experiment == WORKGROUP
              else f'HIP_FA_QUERY_STORAGE mode={mode} ')
    if marker not in log:
        raise ValueError(f'Missing dispatch marker {marker}')
    markers = [line for line in log.splitlines() if 'HIP_FA_KV_SPLIT' in line]
    if not markers or 'HIP_FA_MMA_TILE columns=32 ' not in log:
        raise ValueError('Missing Q3 attention split or tile marker')
    for line in markers:
        fields = dict(re.findall(r'(\w+)=(\S+)', line))
        expected = dict(blocks='8', warps=str(threads//32), tiles='2', kv_batch='64')
        if experiment == QUERY:
            expected['shared'] = '51264' if mode == 'lds' else '34816'
        if any(fields.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Unexpected work distribution: {line}')
    return sorted(set(markers))


def run_screen(out, experiment):
    if processes():
        raise RuntimeError('Benchmark port occupied')
    dump_root = ROOT/'work'/(out.name+'-fixtures')
    dump_root.mkdir(exist_ok=False)
    backend = experiment.backend
    result = dict(hip_sha256=sha(backend/'ggml-hip.dll'), executable_sha256=sha(backend/'test-attention-probe.exe'),
                  numerical=[], performance=[], orders=list(experiment.perf_order()), dumps=str(dump_root),
                  environment={str(t): experiment.config(t) for t in experiment.modes})
    for variant in VARIANTS:
        for mode in experiment.modes:
            tag = f'{variant}-{mode}'
            dump = dump_root/tag
            code, log = execute_kernel(out, 'test', variant, backend, experiment.config(mode), tag+'-test', dump)
            if code not in (0, 1):
                raise ValueError(f'{tag} abnormal probe status {code}')
            markers = audit(log, experiment, mode)
            with (out/(tag+'-oracle.log')).open('w') as stream:
                subprocess.run([NUMPY_PYTHON, TOOLS/'check_attention_float64.py', '--dump-dir', dump,
                                '--out', out/(tag+'.json')], stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=180)
            oracle = read(out/(tag+'.json'))
            error = oracle['backends']['gpu']['nmse_vs_float64']
            length = 160000 if '160k' in variant else 103424
            ok = (math.isfinite(error) and error <= 0.0005 and all(oracle['inputs_equal'].values())
                  and oracle['shapes']['k'] == [256, length, 2, 1])
            row = dict(variant=variant, **{experiment.field: mode}, gpu_nmse=error, qualified=ok,
                       cpu_nmse=oracle['backends']['cpu']['nmse_vs_float64'], upstream_exit=code, markers=markers)
            result['numerical'].append(row)
            save(out/'summary.json', result)
            print(row, flush=True)
    if not all(r['qualified'] for r in result['numerical']):
        return
    run_perf(out, result, experiment)


def run_perf(out, result, experiment):
    for variant in ('strided100k', 'strided160k'):
        for position, mode in enumerate(experiment.perf_order(), 1):
            tag = f'{variant}-perf-{position}-{mode}'
            started = time.time()
            code, log = execute_kernel(out, 'perf', variant, experiment.backend, experiment.config(mode), tag)
            ended = time.time()
            markers = audit(log, experiment, mode)
            lines = [line for line in log.splitlines() if 'FLASH_ATTN_EXT(' in line and 'nb=3,' in line
                     and 'q_range=20' not in line and 'us/run' in line]
            if code or len(lines) != 1:
                raise ValueError(f'{tag} unexpected perf result')
            n, us = re.search(r'(\d+) runs\s+-\s+([\d.]+) us/run', lines[0]).groups()
            row = dict(variant=variant, position=position, **{experiment.field: mode}, us=float(us), runs=int(n),
                       started_unix=started, ended_unix=ended, markers=markers)
            result['performance'].append(row)
            save(out/'summary.json', result)
            print(row, flush=True)
    result['median_us'] = {
        v: {str(t): median(r['us'] for r in result['performance'] if r['variant']==v and r[experiment.field]==t)
            for t in experiment.modes} for v in ('strided100k', 'strided160k')}
    save(out/'summary.json', result)


def check_qualification(screen, experiment):
    qualification = read(screen/'summary.json')
    digest = sha(experiment.backend/'ggml-hip.dll')
    passing = {(r[experiment.field], r['variant']) for r in qualification['numerical'] if r['qualified']}
    if (passing != {(t, v) for t in experiment.modes for v in VARIANTS} or qualification['hip_sha256'] != digest
            or not qualification.get('median_us')):
        raise ValueError('Incomplete qualifying screen or different HIP DLL')
    return digest


def run_generation(out, screen, order, experiment):
    digest = check_qualification(screen, experiment)
    manifest = dict(order=order, screen=os.path.relpath(screen, out), hip_sha256=digest,
                    flags=EXPECTED, environment={str(t): experiment.config(t) for t in experiment.modes},
                    original_request_sha256={key: sha(value[2]) for key, value in CASES.items()})
    save(out/'manifest.json', manifest)
    baseline = read(TOOLS/'slot-cache-reuse-results/snapshot.json')['server']
    rows = []
    for position, mode in enumerate(order, 1):
        case = out/f'case-{position:02}-{mode}'
        os.environ.update(experiment.config(mode))
        with server(case, backend=experiment.backend, expected=EXPECTED) as props:
            for row in restored_workloads(case, props, baseline):
                rows.append(dict(case=position, **{experiment.field: mode}, **row))
                save(out/'samples.json', rows)
        raw = (case/'server.log').read_bytes()
        markers = audit(raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8'), experiment, mode)
        save(case/'dispatch.json', markers)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['screen','perf','generate'])
    p.add_argument('out', type=Path)
    p.add_argument('--screen', type=Path)
    p.add_argument('--repeat-candidate', action='store_true')
    p.add_argument('--query-storage', action='store_true', help='compare register and LDS query storage at 256 threads')
    args = p.parse_args()
    if args.action != 'screen' and args.screen is None:
        p.error('Specify the qualifying screen')
    out = args.out.resolve()
    experiment = QUERY if args.query_storage else WORKGROUP
    out.mkdir(exist_ok=False)
    if args.action == 'screen':
        run_screen(out, experiment)
    elif args.action == 'perf':
        if processes():
            raise RuntimeError('Benchmark port occupied')
        digest = check_qualification(args.screen.resolve(), experiment)
        run_perf(out, dict(qualification=os.path.relpath(args.screen, out), hip_sha256=digest,
                           performance=[], orders=list(experiment.perf_order()),
                           environment={str(t):experiment.config(t) for t in experiment.modes}), experiment)
    else:
        order = experiment.modes * (2 if args.repeat_candidate else 1) + experiment.modes[:1]
        run_generation(out, args.screen.resolve(), order, experiment)


if __name__ == '__main__':
    main()
