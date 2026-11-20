"""Combine the byte-budget floors, llama-bench depth sweeps and server MTP cycles into one decomposition.

Measurements come from bench.jsonl and samples.json files; floors are calculations from byte-budget.json.
Derived quantities (draft cost, overhead, ceilings) are labelled as calculations in the output.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROUND = ROOT/'ceiling-round'
CONTEXT_DEPTH = {'short': 0, '103k': 103194, '160k': 160511}  # short = last 1024 tokens of the 103K prompt


def floors(budget, k, depth):
    """Bytes / bandwidth for a k-token main step (independent routing) and a draft step, in ms."""
    gbps = budget['bandwidth_gbps']
    unique = 256 * (1 - (1 - 8 / 256) ** k)
    per_expert = budget['routed_bytes_per_token_main'] / 8
    main = (budget['nonrouted_main_bytes'] + budget['head_bytes'] + unique * per_expert
            + budget['kv_bytes_per_token_main'] * depth
            + budget['recurrent_state_rw_bytes'])
    draft = (budget['mtp_nonrouted_bytes'] + budget['mtp_routed_bytes'] + budget['head_bytes']
             + budget['kv_bytes_per_token_mtp'] * depth)
    # common_speculative_impl_draft_mtp::process(): the MTP layer also decodes the k verified rows (no head)
    catch_up = (budget['mtp_nonrouted_bytes'] + unique * budget['mtp_routed_bytes'] / 8
                + budget['kv_bytes_per_token_mtp'] * depth)
    return main / gbps / 1e6, draft / gbps / 1e6, catch_up / gbps / 1e6


def labelled(text):
    """'vk=path' -> ('vk', path); a bare path is HIP."""
    label, sep, path = text.partition('=')
    return (label, Path(path)) if sep else ('hip', Path(text))


def bench_rows(paths):
    rows = {}
    for label, path in paths:
        for line in path.read_text(encoding='utf-8').splitlines():
            j = json.loads(line)
            key = (label, j['n_depth'], f"pp{j['n_prompt']}" if j['n_prompt'] else 'tg')
            per_step = j['avg_ns'] / 1e6 / (j['n_gen'] if j['n_gen'] else 1)
            sd = j['stddev_ns'] / 1e6 / (j['n_gen'] if j['n_gen'] else 1)
            rows.setdefault(key, []).append(dict(ms=per_step, sd=sd, source=str(path.relative_to(ROOT))))
    return rows


def server_groups(paths):
    groups = {}
    for path in paths:
        for r in json.loads(path.read_text(encoding='utf-8')):
            key = (r.get('backend', 'hip') + ('-trace' if r.get('trace') else ''), r['mtp'], r['context'], r['workload'])
            g = groups.setdefault(key, dict(ms=0.0, cycles=0, tokens=0, draft=0, accepted=0, n=0))
            g['ms'] += r['predicted_ms']; g['cycles'] += r['cycles']; g['tokens'] += r['predicted_n']
            g['draft'] += r['draft']; g['accepted'] += r['accepted']; g['n'] += 1
    for g in groups.values():
        g['ms_per_cycle'] = g['ms'] / g['cycles']
        g['tokens_per_cycle'] = g['tokens'] / g['cycles']
        g['tps'] = 1000 * g['tokens'] / g['ms']
        g['acceptance'] = g['accepted'] / g['draft'] if g['draft'] else None
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench', nargs='+', default=[str(ROUND/'depth-bench/bench.jsonl')],
                   help='bench.jsonl paths; prefix with vk= for the Vulkan runtime')
    p.add_argument('--server', type=Path, nargs='+', default=[ROUND/'server-cycles/samples.json'])
    p.add_argument('--out', type=Path, default=ROUND/'analysis.json')
    args = p.parse_args()
    budget = json.loads((ROUND/'byte-budget.json').read_text(encoding='utf-8'))
    bench = bench_rows([(label, path.resolve()) for label, path in map(labelled, args.bench) if path.exists()])
    servers = server_groups([s.resolve() for s in args.server if s.exists()])

    def bench_ms(label, depth, test):
        v = bench.get((label, depth, test))
        return sum(x['ms'] for x in v) / len(v) if v else None

    depth_table = []
    for label, depth in sorted({(l, d) for l, d, _ in bench}):
        row = dict(backend=label, depth=depth)
        for test, k in (('tg', 1), ('pp2', 2), ('pp3', 3), ('pp4', 4)):
            ms = bench_ms(label, depth, test)
            if ms is None:
                continue
            floor = floors(budget, k, depth)[0]
            row[test] = dict(measured_ms=ms, floor_ms=floor, ratio=ms / floor,
                             samples=bench[(label, depth, test)])
        depth_table.append(row)

    cycles = []
    for (backend, mtp, context, workload), g in sorted(servers.items()):
        depth = CONTEXT_DEPTH[context]
        main_floor, draft_floor, catch_up_floor = floors(budget, mtp + 1, depth)
        floor_cycle = main_floor + (mtp * draft_floor + catch_up_floor if mtp else 0)
        row = dict(backend=backend, mtp=mtp, context=context, workload=workload, depth_for_bench=depth, **g,
                   floor_cycle_ms=floor_cycle, ceiling_tps_at_floor=1000 * g['tokens_per_cycle'] / floor_cycle)
        verify = bench_ms(backend.split('-')[0], depth, 'tg' if mtp == 0 else f'pp{mtp + 1}')
        if verify is not None:
            row['bench_main_step_ms'] = verify
            residual = g['ms_per_cycle'] - verify
            row['residual_ms'] = residual  # calculation: drafts + server/sampling overhead
            if mtp:
                row['residual_per_draft_ms'] = residual / mtp
                row['draft_floor_ms'] = draft_floor
                row['catch_up_floor_ms'] = catch_up_floor
                row['residual_floor_ms'] = mtp * draft_floor + catch_up_floor
        cycles.append(row)

    result = dict(kind='calculation from measurements', bandwidth_gbps=budget['bandwidth_gbps'],
                  bench_sources=[str(b) for b in args.bench], server_sources=[str(s) for s in args.server],
                  notes=['short context is benchmarked against llama-bench depth 0 (server prompt is ~1.1K tokens)',
                         'residual = server ms/cycle minus llama-bench main step at the same depth; it holds the MTP '
                         'draft passes, sampling and server overhead',
                         'ceiling_tps_at_floor keeps measured tokens/cycle and replaces cycle time by the byte floor'],
                  depth_table=depth_table, cycles=cycles)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')

    print('backend   depth  test  measured  floor  ratio')
    for row in depth_table:
        for test in ('tg', 'pp2', 'pp3', 'pp4'):
            if test in row:
                r = row[test]
                print(f"{row['backend']:7s} {row['depth']:7d}  {test:4s} {r['measured_ms']:8.1f} {r['floor_ms']:6.1f}  {r['ratio']:.2f}")
    print('\nbackend   mtp context workload     n  ms/cycle tok/cyc   tps  bench-step residual/draft  floor  ceiling-tps')
    for r in cycles:
        print(f"{r['backend']:9s} {r['mtp']:3d} {r['context']:6s} {r['workload']:11s} {r['n']:2d} {r['ms_per_cycle']:8.1f} "
              f"{r['tokens_per_cycle']:6.2f} {r['tps']:6.2f} {r.get('bench_main_step_ms', float('nan')):9.1f} "
              f"{r.get('residual_per_draft_ms', r.get('residual_ms', float('nan'))):9.1f} "
              f"{r['floor_cycle_ms']:7.1f} {r['ceiling_tps_at_floor']:8.2f}")


if __name__ == '__main__':
    main()
