"""Summarize bracketed AC-mode generation rates and coarse ADLX samples."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def summarize(out, sensor_file):
    rows = json.loads((out/'samples.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    sensors = list(csv.DictReader(sensor_file.open(encoding='utf-8-sig')))
    if len(rows) != 8 or len({(r['case'], r['length']) for r in rows}) != 8:
        raise ValueError('Expected eight distinct measured requests')
    if any(r['ac_mode'] != manifest['order'][r['case'] - 1] for r in rows):
        raise ValueError('Measured setting differs from trial order')
    results = []
    for length in sorted({r['length'] for r in rows}):
        subset = [r for r in rows if r['length'] == length]
        if {r['case'] for r in subset} != {1, 2, 3, 4}:
            raise ValueError('Incomplete matched set')
        work_groups = {(r['token_ids_sha256'], r['timings']['draft_n'], r['timings']['draft_n_accepted'])
                       for r in subset}
        by_mode = {}
        for mode in dict.fromkeys(manifest['order']):
            group = [r for r in subset if r['ac_mode'] == mode]
            gen_tokens = sum(r['timings']['predicted_n'] - 1 for r in group)
            seconds = sum(r['timings']['predicted_ms'] / 1000 for r in group)
            selected = [s for r in group for s in sensors
                        if r['started_unix'] + r['timings']['prompt_ms']/1000 < int(s['unix_ms']) / 1000 < r['ended_unix']]
            by_mode[mode] = dict(pooled_tps=gen_tokens / seconds,
                                 request_tps=[r['timings']['predicted_per_second'] for r in group],
                                 gpu_sensor_samples=len(selected))
            for key in ['gpu_clock_mhz', 'gpu_temperature_c', 'gpu_power_w']:
                values = [float(s[key]) for s in selected if s[key]]
                if values:
                    by_mode[mode][key] = {'min': min(values), 'mean': mean(values), 'max': max(values)}
        base, candidate = manifest['order'][:2]
        change = 100 * (by_mode[candidate]['pooled_tps'] / by_mode[base]['pooled_tps'] - 1)
        results.append(dict(input_tokens=length, token_hash_and_draft_work_matched=len(work_groups) == 1,
                            modes=by_mode, best_performance_change_pct=change))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('sensors', type=Path)
    args = parser.parse_args()
    result = summarize(args.results, args.sensors)
    (args.results/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
