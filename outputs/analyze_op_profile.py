"""Per-graph op-class GPU time, HIP (GGML_HIP_OP_PROFILE) vs Vulkan (GGML_VK_PERF_LOGGER), from run_op_profile.py.

Graph types: verify = main model (has Gated DeltaNet), with its query-token count; catch-up = MTP layer without
top-k sampling; draft = MTP layer with top-k sampling (Vulkan) or a 1-token MTP graph (HIP). Both timers are
diagnostic and synchronized; HIP negative event samples are dropped and counted.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

CLASSES = ('attention', 'experts', 'head', 'dense', 'recurrent', 'other')
HIP = re.compile(r'HIP_OP_PROFILE graph=(\d+) ctx=(\S+) gdn=(\d) q=(\d+) kv=(\d+) first=\d+ last=\d+ gpu_ms=(-?[\d.]+) '
                 r'wall_us=\d+ ops=(\S+) shape=\S+ src0_type=\S+ src0_shape=(\S+)')
VK = re.compile(r'^(.*): (\d+) x ([\d.e+-]+) us = ([\d.e+-]+) us')


def text(path):
    raw = path.read_bytes()
    return raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8', errors='replace')


def classify(ops, head):
    if 'FLASH_ATTN_EXT' in ops:
        return 'attention'
    if 'MUL_MAT_ID' in ops:
        return 'experts'
    if head:
        return 'head'
    if 'MUL_MAT' in ops:
        return 'dense'
    if 'GATED_DELTA_NET' in ops or 'SSM_CONV' in ops:
        return 'recurrent'
    return 'other'


def hip_graphs(log):
    graphs = defaultdict(lambda: dict(classes=defaultdict(float), q=0, gdn=0, negative=0))
    for m in HIP.finditer(text(log)):
        g = graphs[(m[2], int(m[1]))]
        g['q'], g['gdn'] = int(m[4]), int(m[3])
        ms = float(m[6])
        if ms < 0:
            g['negative'] += 1
            continue
        g['classes'][classify(m[7], ',248320,' in f',{m[8]},')] += ms
    out = []
    for g in graphs.values():
        kind = f"verify-q{g['q']}" if g['gdn'] else ('draft' if g['q'] == 1 else f"catch-up-q{g['q']}")
        out.append(dict(kind=kind, negative=g['negative'], **g['classes']))
    return out


def vk_graphs(log):
    out, current = [], None
    for line in text(log).splitlines():
        if line.startswith('Vulkan Timings:'):
            current = dict(classes=defaultdict(float), names=[])
            continue
        if current is None:
            continue
        if line.startswith('Total time:'):
            names = ' '.join(current['names'])
            q = re.search(r'FLASH_ATTN_EXT[^:]*q\(256,(\d+),', names)
            q = int(q[1]) if q else 0
            if 'GATED_DELTA_NET' in names:
                kind = f'verify-q{q}'
            else:
                kind = 'draft' if 'TOP_K' in names else f'catch-up-q{q}'
            out.append(dict(kind=kind, negative=0, **current['classes']))
            current = None
            continue
        m = VK.match(line)
        if m:
            name = m[1]
            current['names'].append(name)
            current['classes'][classify(name, 'm=248320' in name)] += float(m[4]) / 1000
    return out


def summarize(graphs):
    groups = defaultdict(lambda: dict(n=0, negative=0, **{c: 0.0 for c in CLASSES}))
    for g in graphs:
        s = groups[g['kind']]
        s['n'] += 1
        s['negative'] += g['negative']
        for c in CLASSES:
            s[c] += g.get(c, 0.0)
    result = {}
    for kind, s in groups.items():
        per = {c: s[c] / s['n'] for c in CLASSES}
        result[kind] = dict(graphs=s['n'], negative_samples=s['negative'], total_ms=sum(per.values()), **per)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    args = p.parse_args()
    result = dict(kind='diagnostic synchronized timings (not production wall time)',
                  hip=summarize(hip_graphs(args.run/'hip/server.log')),
                  vk=summarize(vk_graphs(args.run/'vk/server.log')))
    (args.run/'op-summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"{'backend':7s} {'graph':12s} {'n':>4s} " + ' '.join(f'{c:>9s}' for c in CLASSES) + f" {'total':>8s}  (ms per graph)")
    for backend in ('hip', 'vk'):
        for kind, s in sorted(result[backend].items()):
            print(f"{backend:7s} {kind:12s} {s['graphs']:4d} " + ' '.join(f'{s[c]:9.2f}' for c in CLASSES)
                  + f" {s['total_ms']:8.2f}" + (f"  neg={s['negative_samples']}" if s['negative_samples'] else ''))


if __name__ == '__main__':
    main()
