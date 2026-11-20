"""Correctness (vs CPU) and speed of the model's matrix-vector shapes for isolated HIP runtimes.

Each runtime directory must contain test-matvec-probe.exe (outputs/build_matvec_probe.py). `test` compares the
ROCm0 result with the CPU backend (normalized MSE limit 5e-4); `perf` reports us/run and GB/s per case.
Runtimes are interleaved per repeat so drift affects all of them alike.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT/'work/rocm10-runtime/Lib/site-packages'
PERF = re.compile(r'^\s*(\S+\(.*\)):\s+(\d+) runs -\s+([\d.]+) us/run')
TEST = re.compile(r'^\s*(\S+\(.*\)):\s+.*?\b(OK|FAIL)\b', re.M)


def run(runtime, mode):
    env = os.environ.copy()
    env['LLAMA_HIP_MATVEC_PROBE'] = '1'
    env.pop('LLAMA_HIP_ATTENTION_PROBE', None)
    env.pop('GGML_BACKEND_PATH', None)
    env['PATH'] = os.pathsep.join(map(str, (runtime, SDK/'_rocm_sdk_core/bin', SDK/'_rocm_sdk_libraries/bin', env['PATH'])))
    result = subprocess.run([str(runtime/'test-matvec-probe.exe'), mode, '-b', 'ROCm0'], cwd=runtime, env=env,
                            capture_output=True, text=True, timeout=1800)
    log = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout + result.stderr)  # drop terminal colors around OK/FAIL
    if str(runtime/'ggml-hip.dll').lower() not in log.lower():
        raise RuntimeError(f'{runtime}: probe did not report loading its own ggml-hip.dll')
    return result.returncode, log




def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('runtimes', nargs='+', help='name=directory')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--skip-test', action='store_true')
    p.add_argument('--reparse', action='store_true', help='summarize the logs already in OUT instead of running')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=args.reparse)
    runtimes = dict(r.split('=', 1) for r in args.runtimes)
    runtimes = {k: Path(v).resolve() for k, v in runtimes.items()}
    summary = dict(runtimes={k: str(v) for k, v in runtimes.items()}, test={}, perf={})
    if args.reparse:
        def run(runtime, mode, _names={str(v): k for k, v in runtimes.items()}, _count={}):
            name = _names[str(runtime)]
            if mode == 'test':
                path = args.out/f'{name}-test.log'
            else:
                i = _count[name] = _count.get(name, -1) + 1
                path = args.out/f'{name}-perf-{i}.log'
            return 0, re.sub(r'\x1b\[[0-9;]*m', '', path.read_text(encoding='utf-8'))
    else:
        run_probe = globals()['run']
        def run(runtime, mode):
            return run_probe(runtime, mode)
    if not args.skip_test:
        for name, runtime in runtimes.items():
            code, log = run(runtime, 'test')
            if not args.reparse:
                (args.out/f'{name}-test.log').write_text(log, encoding='utf-8')
            results = TEST.findall(log)
            summary['test'][name] = dict(exit=code, ok=sum(r == 'OK' for _, r in results),
                                         fail=[c for c, r in results if r == 'FAIL'])
            print(name, 'test', summary['test'][name]['ok'], 'ok', len(summary['test'][name]['fail']), 'fail', flush=True)
    for repeat in range(args.repeats):
        for name, runtime in runtimes.items():
            code, log = run(runtime, 'perf')
            if not args.reparse:
                (args.out/f'{name}-perf-{repeat}.log').write_text(log, encoding='utf-8')
            for line in log.splitlines():
                m = PERF.match(line)
                if m:
                    row = summary['perf'].setdefault(m[1], {}).setdefault(name, [])
                    row.append(float(m[3]))
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    names = list(runtimes)
    print('\nmedian us/run' + ''.join(f'{n:>12s}' for n in names) + '   case')
    for case, rows in summary['perf'].items():
        meds = [sorted(rows[n])[len(rows[n])//2] if rows.get(n) else float('nan') for n in names]
        short = re.sub(r'type_b=f32,|bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1,src_overlap=0,m_v=0,pad=0|,m_v=0,ids_offset=0,skip=0', '', case)
        print(' ' * 13 + ''.join(f'{v:12.1f}' for v in meds) + '   ' + short[:120])


if __name__ == '__main__':
    main()
