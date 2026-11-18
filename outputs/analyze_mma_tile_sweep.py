"""Summarize completed MMA tile trials without pooling greedy and stochastic runs."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent / 'mma-tile-100k-results-verified')
    parser.add_argument('--parameter-name', default='MMA columns')
    args = parser.parse_args()
    out = args.directory
    records = json.loads((out / 'samples.json').read_text())
    blocks = []
    for case in sorted({r['case'] for r in records}):
        primary = [r for r in records if r['case'] == case and r['temperature'] == 0.6]
        greedy = [r for r in records if r['case'] == case and r['temperature'] == 0.0]
        if not primary:
            continue
        rates = [r['tps'] for r in primary]
        blocks.append({'case': case, 'mode': primary[0]['mode'], 'runs': len(rates),
                       'mean_tps': statistics.mean(rates), 'min_tps': min(rates), 'max_tps': max(rates),
                       'mean_ttft_s': statistics.mean(r['ttft_s'] for r in primary),
                       'output_hashes': [r['output_sha256'] for r in primary],
                       'draft_proposed': [r['draft_n'] for r in primary],
                       'draft_accepted': [r['accepted_drafts'] for r in primary],
                       'greedy': greedy})
    by_case = {b['case']: b for b in blocks}
    for block in blocks:
        if block['mode'] == 'auto':
            continue
        adjacent = [by_case[c] for c in (block['case']-1, block['case']+1) if c in by_case]
        controls = [b for b in adjacent if b['mode'] == 'auto']
        if len(controls) == 2:
            baseline = statistics.mean(b['mean_tps'] for b in controls)
            block['adjacent_control_tps'] = baseline
            block['percent_vs_adjacent_controls'] = 100*(block['mean_tps']/baseline-1)
            block['primary_outputs_match_both_controls'] = all(block['output_hashes'] == c['output_hashes'] for c in controls)
            block['greedy_output_matches_both_controls'] = bool(block['greedy']) and all(
                c['greedy'] and block['greedy'][0]['output_sha256'] == c['greedy'][0]['output_sha256'] for c in controls)
    summary = {'blocks': blocks,
               'primary_request_hashes': sorted({r['request_sha256'] for r in records if r['temperature'] == 0.6}),
               'measured_runs': len(records), 'complete_five_blocks': len(blocks) == 5 and all(b['runs'] == 3 and b['greedy'] for b in blocks)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = [f'| Block | {args.parameter_name} | Primary samples | Mean TPS | Range TPS | Greedy TPS |',
             '|---|---|---:|---:|---:|---:|']
    for b in blocks:
        greedy = f"{b['greedy'][0]['tps']:.3f}" if b['greedy'] else 'pending'
        lines.append(f"| {b['case']} | {b['mode']} | {b['runs']} | {b['mean_tps']:.3f} | {b['min_tps']:.3f}-{b['max_tps']:.3f} | {greedy} |")
    table = '\n'.join(lines)+'\n'
    (out / 'RESULTS.md').write_text(table, encoding='utf-8')
    print(table)


if __name__ == '__main__':
    main()
