"""Server decode cost per verify cycle with MTP 2, MTP 1 and speculation off.

Each case runs short-context workloads (the last 1024 tokens of the 103K request) and the restored
103K/160K workloads used by earlier rounds. A cycle is one main-model step: generated tokens minus
accepted draft tokens with MTP, or one per token without it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from benchmark_server import ROOT, TOOLS, server
from run_fine_split import CASES, REPETITION, read, restored_workloads, save
from run_thread_sweep import get
import run_mtp_benchmark as probe

BACKENDS = {'hip': (ROOT/'work/hip-kv-split-bin', 'ROCm0', 'ggml-hip.dll'),
            'vk': (ROOT/'work/vulkan-bin', 'Vulkan0', 'ggml-vulkan.dll')}
LAUNCHER = TOOLS/'Start-CeilingServer.ps1'
STOPPER = TOOLS/'Stop-CeilingServer.ps1'
SHORT_TOKENS = 1024


def parse_case(text):
    """'2' or 'hip2' = HIP MTP 2; 'vk0' = Vulkan without speculation; 'x-NAME:2' = HIP from work/hip-mmvq/NAME/bin.
    Flags after the MTP digit: 't' = trace logging (-lv 4), 'g' = HIP graphs disabled."""
    m = re.fullmatch(r'(?:(hip|vk)|x-(\w+):)?([0-3])([tg]*)', text.strip())
    if not m:
        raise ValueError(f'bad case {text!r}')
    backend = m.group(1) or (f'x-{m.group(2)}' if m.group(2) else 'hip')
    if backend.startswith('x-') and backend not in BACKENDS:
        BACKENDS[backend] = (ROOT/'work/hip-mmvq'/m.group(2)/'bin', 'ROCm0', 'ggml-hip.dll')
    return dict(backend=backend, mtp=int(m.group(3)), trace='t' in m.group(4), nographs='g' in m.group(4))


def label(spec):
    return spec['backend'] + ('-trace' if spec.get('trace') else '') + ('-nographs' if spec.get('nographs') else '')


def cycles(row):
    return row['predicted_n'] - row['accepted'] if row['draft'] else row['predicted_n']


def short_workloads(case):
    probe.OUT = case
    base = read(CASES['103k'][2])['prompt'][-SHORT_TOKENS:]
    suffix = probe.tokenize(REPETITION)
    plan = [('warmup', base, 48, 0.6)] + [('sampled', base, 256, 0.6)] * 3 + [('repetition', base + suffix, 128, 0.0)] * 2
    for trial, (workload, prompt, count, temperature) in enumerate(plan):
        if any(slot['is_processing'] for slot in get('/slots')):
            raise RuntimeError('Server busy')
        started = time.time()
        result = probe.run(f'short-{trial}-{workload}', prompt, True, count, temperature=temperature)
        ended = time.time()
        t = result['final']['timings']
        if trial:
            yield dict(context='short', trial=trial, workload=workload, tokens=len(prompt), cache_n=t['cache_n'],
                       prompt_n=t['prompt_n'], predicted_n=t['predicted_n'], predicted_ms=t['predicted_ms'],
                       tps=t['predicted_per_second'], draft=t.get('draft_n', 0), accepted=t.get('draft_n_accepted', 0),
                       started_unix=started, ended_unix=ended,
                       tokens_sha256=hashlib.sha256(json.dumps(result['generated_token_ids']).encode()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--order', default='2,0,1,2')
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    order = [parse_case(x) for x in args.order.split(',')]
    baseline = read(TOOLS/'slot-cache-reuse-results/snapshot.json')['server']
    used = {c['backend'] for c in order}
    save(out/'manifest.json', dict(order=order, launcher=str(LAUNCHER), short_tokens=SHORT_TOKENS,
                                   backends={b: dict(directory=str(BACKENDS[b][0]), device=BACKENDS[b][1],
                                                     dll_sha256=hashlib.sha256((BACKENDS[b][0]/BACKENDS[b][2]).read_bytes()).hexdigest())
                                             for b in used}))
    rows = []
    for position, spec in enumerate(order, 1):
        draft_tokens = spec['mtp']
        directory, device, _ = BACKENDS[spec['backend']]
        name = label(spec)
        case = out/(f"case-{position:02}-{spec['backend']}-mtp{draft_tokens}" + name[len(spec['backend']):])
        launch = ['-DraftTokens', draft_tokens, '-Device', device, '-LogVerbosity', 4 if spec['trace'] else 3]
        if spec['nographs']:
            launch.append('-NoGraphs')
        tag = dict(case=position, backend=spec['backend'], trace=spec['trace'], nographs=spec['nographs'],
                   label=name, mtp=draft_tokens)
        with server(case, launch, backend=directory, launcher=LAUNCHER, stopper=STOPPER) as props:
            for row in short_workloads(case):
                rows.append(dict(**tag, **row, cycles=cycles(row)))
                save(out/'samples.json', rows)
            try:
                for row in restored_workloads(case, props, baseline):
                    rows.append(dict(**tag, **row, cycles=cycles(row)))
                    save(out/'samples.json', rows)
            except Exception as error:  # restoring an MTP snapshot without speculation may be refused
                save(case/'restored-error.json', dict(error=repr(error)))
                print(json.dumps({'event': 'restored-skipped', 'case': position, 'error': repr(error)}), flush=True)
    summary = {}
    for row in rows:
        key = f"{row.get('label', row['backend'])}/mtp{row['mtp']}/{row['context']}/{row['workload']}"
        group = summary.setdefault(key, dict(ms=0.0, cycles=0, tokens=0, n=0))
        group['ms'] += row['predicted_ms']; group['cycles'] += row['cycles']
        group['tokens'] += row['predicted_n']; group['n'] += 1
    for group in summary.values():
        group['ms_per_cycle'] = group['ms']/group['cycles']
        group['tokens_per_cycle'] = group['tokens']/group['cycles']
        group['tps'] = 1000*group['tokens']/group['ms']
    save(out/'summary.json', summary)
    for key, group in sorted(summary.items()):
        print(f"{key:36s} n={group['n']}  {group['ms_per_cycle']:7.2f} ms/cycle  "
              f"{group['tokens_per_cycle']:.2f} tok/cycle  {group['tps']:6.2f} tok/s")


if __name__ == '__main__':
    main()
