"""Summarize normal-mode generation, bracketed comparisons and sampled-work differences."""
import argparse
from pathlib import Path

from analyze_fine_split import matched, rate
from run_fine_split import read, save


def analyze(root):
    manifest = read(root/'manifest.json')
    order = manifest['order']
    modes = tuple(order[:2])
    fields = {(256, 128): 'threads', ('registers', 'lds'): 'query_storage'}
    if modes not in fields or order not in (list(modes * 2 + modes[:1]), list(modes + modes[:1])):
        raise ValueError(f'Unexpected bracket order {order}')
    rows = read(root/'samples.json')
    by_case = {(r['case'], r['trial']): r for r in rows}
    if len(rows) != 6*len(order) or len(rows) != len(by_case):
        raise ValueError('Incomplete generation evidence')
    comparisons = []
    for context, workload, trials in (('103k', 'sampled', (1, 4)), ('160k', 'sampled', (2, 3)),
                                      ('103k', 'repetition', (5,)), ('160k', 'repetition', (6,))):
        groups = {}
        for mode in modes:
            subset = [r for r in rows if (r['context'], r['workload'], r[fields[modes]]) == (context, workload, mode)]
            groups[str(mode)] = dict(n=len(subset), pooled_tps=rate(subset),
                                       samples=[r['tps'] for r in subset], min_tps=min(r['tps'] for r in subset))
        gain = 100*(groups[str(modes[1])]['pooled_tps']/groups[str(modes[0])]['pooled_tps']-1)
        brackets = []
        for candidate in range(2, len(order), 2):
            triplets = [[by_case[pos, t] for pos in (candidate-1, candidate, candidate+1)] for t in trials]
            cand_tps = rate([center for _, center, _ in triplets])
            ctrl_tps = rate([r for before, _, after in triplets for r in (before, after)])
            brackets.append(dict(position=candidate, change_pct=100*(cand_tps/ctrl_tps-1),
                                 matched_prefixes=[matched(group, root) for group in triplets]))
        comparisons.append(dict(context=context, workload=workload, groups=groups, change_pct=gain, brackets=brackets))
    save(root/'comparison.json', comparisons)
    return comparisons


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    for row in analyze(args.results.resolve()):
        print(row['context'], row['workload'], row['change_pct'])
