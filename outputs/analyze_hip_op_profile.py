"""Summarize the private HIP operation timer; percentages are diagnostic, not production shares."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


def category(ops):
    if 'FLASH_ATTN_EXT' in ops:
        return 'attention'
    if 'MUL_MAT_ID' in ops:
        return 'expert/shared groups'
    if 'GATED_DELTA_NET' in ops:
        return 'recurrent groups'
    if 'MUL_MAT' in ops:
        return 'dense/vocabulary matmul'
    return 'other groups'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    with args.log.open('rb') as raw:
        encoding = 'utf-16' if raw.read(2) in (b'\xff\xfe', b'\xfe\xff') else 'utf-8'
    entries = []
    with args.log.open(encoding=encoding, errors='replace') as log:
        for line in log:
            if 'HIP_OP_PROFILE ' not in line:
                continue
            fields = dict(re.findall(r'(\w+)=([^\s]+)', line.split('HIP_OP_PROFILE ', 1)[1]))
            if not {'graph','ctx','gdn','q','kv','gpu_ms','ops','shape'} <= fields.keys():
                raise RuntimeError('Incomplete operation timing record')
            for key in ('graph','gdn','q','kv','first','last','wall_us'):
                fields[key] = int(fields[key])
            fields['gpu_ms'] = float(fields['gpu_ms'])
            fields['raw_gpu_ms'] = fields['gpu_ms']
            fields['valid_gpu_timing'] = fields['gpu_ms'] >= 0
            if not fields['valid_gpu_timing']:
                fields['gpu_ms'] = 0.0
            if fields['kv'] < 100000 or not 1 <= fields['q'] <= 3:
                raise RuntimeError('Unexpected profiled attention context/query length')
            fields['category'] = category(fields['ops'])
            entries.append(fields)
    if not entries or sum(row['gpu_ms'] for row in entries) <= 0:
        raise RuntimeError('No usable HIP operation timings found')
    contexts = defaultdict(list)
    for row in entries:
        contexts[(row['ctx'], row['gdn'])].append(row)
    summaries = []
    for (context, gdn), rows in contexts.items():
        total = sum(row['gpu_ms'] for row in rows)
        cats = defaultdict(float)
        shapes = defaultdict(list)
        for row in rows:
            cats[row['category']] += row['gpu_ms']
            key = (row['ops'], row['shape'], row.get('src0_type'),
                   row.get('src0_shape'), row.get('src1_shape'))
            shapes[key].append(row)
        groups = []
        for key, grouped in shapes.items():
            elapsed = sum(row['gpu_ms'] for row in grouped)
            groups.append({'ops': key[0], 'shape': key[1], 'src0_type': key[2],
                           'src0_shape': key[3], 'src1_shape': key[4], 'calls': len(grouped),
                           'gpu_ms': elapsed,
                           'valid_samples': sum(row['valid_gpu_timing'] for row in grouped),
                           'invalid_negative_samples': sum(not row['valid_gpu_timing'] for row in grouped),
                           'mean_gpu_ms': elapsed / max(1, sum(row['valid_gpu_timing'] for row in grouped)),
                           'mean_wall_ms': sum(row['wall_us'] for row in grouped) / (1000 * len(grouped)),
                           'percent': 100 * elapsed / total,
                           'example_tensor': grouped[0].get('name')})
        groups.sort(key=lambda group: group['gpu_ms'], reverse=True)
        summaries.append({'context': context, 'contains_gdn': bool(gdn),
                          'graphs': len({row['graph'] for row in rows}), 'groups': len(rows),
                          'query_lengths': sorted({row['q'] for row in rows}),
                          'kv_lengths': sorted({row['kv'] for row in rows}),
                          'recorded_gpu_ms': total,
                          'summed_wall_ms': sum(row['wall_us'] for row in rows) / 1000,
                          'invalid_negative_samples': sum(not row['valid_gpu_timing'] for row in rows),
                          'categories': {key: {'gpu_ms': val, 'percent': val * 100 / total}
                                         for key, val in cats.items()}, 'shapes': groups})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    invalid = [row for row in entries if not row['valid_gpu_timing']]
    quality = {'negative_samples': len(invalid), 'fraction_negative': len(invalid) / len(entries),
               'minimum_raw_gpu_ms': min(row['raw_gpu_ms'] for row in entries),
               'negative_magnitude_ms': -sum(row['raw_gpu_ms'] for row in invalid),
               'aggregation': 'Negative durations are invalid and excluded from GPU sums/means; raw log retained. This does not bound all timing error.'}
    summary = {'source_log': str(args.log.resolve()), 'operation_records': len(entries),
               'timer_quality': quality,
               'contexts': summaries,
               'instrumentation': 'HIP event timing, synchronized per fused operation group; graph replay and concurrent-region scheduling disabled',
               'selection': 'graphs containing Flash Attention with KV >= 100000 and max query length <= 3',
               'limits': 'No individual kernel trace, memory counters or CPU+GPU production wall-time decomposition. Context roles are inferred from GDN presence.'}
    (args.out_dir / 'timing-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['# HIP operation-timing diagnostic', '',
             'These are synchronized operation-group GPU timings from a private diagnostic build. '
             'They are not individual-kernel hardware counters or fractions of production request wall time.', '',
             'The diagnostic disables graph replay and concurrent-region scheduling but retains ordinary operation fusions. '
             'A group containing multiple operations is assigned as a whole to its category. '
             'Only graphs containing Flash Attention at >=100K KV length and <=3 query tokens are sampled; '
             'other graphs and CPU work are outside this summary.', '',
             f'Timer-quality warning: {len(invalid)} of {len(entries)} records have invalid negative GPU durations '
             f'({100 * len(invalid) / len(entries):.2f}%). The minimum is {quality["minimum_raw_gpu_ms"]:.6f} ms. '
             'Negative records are retained in the raw log and excluded from GPU totals/means below. '
             'Their absolute sum does not bound all event-timer error. Treat these as provisional rankings.', '']
    for item in summaries:
        label = 'GDN present (target-like, inferred)' if item['contains_gdn'] else 'GDN absent (draft-like, inferred)'
        lines += [f'## {label}', '',
                  f'Context `{item["context"]}`: {item["graphs"]} graphs, {item["groups"]} timed groups, '
                  f'query lengths {item["query_lengths"]}, KV lengths {item["kv_lengths"]}.', '',
                  '| Category | Recorded GPU ms | Share of this context |', '|---|---:|---:|']
        for key, val in sorted(item['categories'].items(), key=lambda kv: kv[1]['gpu_ms'], reverse=True):
            lines.append(f'| {key} | {val["gpu_ms"]:.3f} | {val["percent"]:.2f}% |')
        lines += ['', '| Top operation group | Source 0 type / shape | Source 1 shape | Calls | Invalid timings | Mean valid GPU ms | Mean synchronized wall ms | Share |',
                  '|---|---|---|---:|---:|---:|---:|---:|']
        for group in item['shapes'][:10]:
            lines.append(f'| {group["ops"]} | {group["src0_type"]} / {group["src0_shape"]} | '
                         f'{group["src1_shape"]} | {group["calls"]} | {group["invalid_negative_samples"]} | '
                         f'{group["mean_gpu_ms"]:.3f} | {group["mean_wall_ms"]:.3f} | {group["percent"]:.2f}% |')
        lines.append('')
    lines += ['## Interpretation limits', '',
              'Device-event waits and logging change scheduling, clock behavior and CPU/GPU overlap. '
              'Kernel launch/host overhead is not isolated by an operation timer. Small operations can be distorted '
              'by event overhead. Context labels above are inferences, and their percentages must not be summed '
              'as if each were a fraction of the whole request. This identifies candidates for further profiling '
              'and isolated benchmarks; it does not establish bandwidth saturation, occupancy limits, register '
              'spills or a speedup achievable by rewriting a kernel.']
    (args.out_dir / 'OPERATION-TIMINGS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'records': len(entries), 'contexts': [
        {'gdn': item['contains_gdn'], 'graphs': item['graphs'], 'categories': item['categories']}
        for item in summaries]}, indent=2))


if __name__ == '__main__':
    main()
