"""Compare elapsed generation time and identify repeatable identical prefixes in bracketed streams."""
import argparse
from itertools import takewhile
import json
from pathlib import Path
from statistics import median

from run_fine_split import read, save


def rate(rows):
    # llama-server excludes the first output token, which belongs to prompt evaluation.
    return 1000 * sum(row['predicted_n'] - 1 for row in rows) / sum(row['predicted_ms'] for row in rows)


def events(root, row):
    setting = next(row[key] for key in ('query_storage', 'threads', 'blocks') if key in row)
    case = root / 'case-{:02}-{}'.format(row['case'], setting)
    path = case / '{}-{}-events.jsonl'.format(row['trial'], row['context'])
    stream = []
    for line in path.read_text(encoding='utf-8').splitlines():
        event = json.loads(line)['response']
        if event.get('stop') or 'prompt_progress' in event:
            continue
        ids = event['tokens']
        timing = event['timings']
        if len(ids) != 1 or timing['predicted_n'] != len(stream) + 1:
            raise RuntimeError(f'Unexpected token stream format in {path}')
        stream.append((ids[0], timing.get('draft_n', 0), timing.get('draft_n_accepted', 0),
                       timing['predicted_ms']))
    if len(stream) != row['predicted_n']:
        raise RuntimeError(f'Incomplete token stream in {path}')
    return stream


def matched(rows, root):
    streams = [events(root, row) for row in rows]
    triples = zip(*streams)
    common = list(takewhile(lambda group: len({event[:3] for event in group}) == 1, triples))
    # A stream can emit several tokens after a single timed verification. Include only entire
    # verification cycles common to all three runs. The first token belongs to prompt evaluation.
    for count in range(len(common), 1, -1):
        if all(count == len(stream) or stream[count][3] > stream[count-1][3] for stream in streams):
            times = [stream[count-1][3] for stream in streams]
            return {'n': count, 'draft': common[count-1][0][1], 'accepted': common[count-1][0][2],
                    'before_candidate_after_ms': times, 'candidate_tps': (count-1)*1000/times[1],
                    'control_tps': (count-1)*2000/(times[0]+times[2])}
    return {'n': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    root = args.results
    rows = read(root/'samples.json')
    manifest = read(root/'manifest.json')
    expected_order = manifest['order']
    if expected_order[::2] != [8]*len(expected_order[::2]) or len(expected_order)%2 != 1:
        raise RuntimeError('Analysis expects eight-block controls before and after every candidate')
    by_case = {(r['case'], r['trial']): r for r in rows}
    if len(by_case) != 6 * len(expected_order):
        raise RuntimeError('Incomplete samples')
    comparisons = []
    for case in range(2, len(expected_order), 2):
        for context, trials in [('103k', (1, 4)), ('160k', (2, 3)), ('103k', (5,)), ('160k', (6,))]:
            triplets = [[by_case[position, trial] for position in (case-1, case, case+1)] for trial in trials]
            controls = [sample for before, _, after in triplets for sample in (before, after)]
            candidates = [r[1] for r in triplets]
            candidate_tps, control_tps = rate(candidates), rate(controls)
            comparisons.append({'blocks': expected_order[case-1], 'case': case, 'context': context,
                'workload': candidates[0]['workload'],
                'candidate_tps': candidate_tps, 'control_tps': control_tps,
                'candidate_tps_change_pct': 100*(candidate_tps/control_tps-1),
                'candidate_samples': [r['tps'] for r in candidates],
                'control_samples': [r['tps'] for r in controls],
                'matched_prefixes': [matched(group, root) for group in triplets]})
    summary = read(Path(manifest['screen'])/'summary.json')
    medians = {v: {b: median(r['us'] for r in summary['performance'] if r['variant'] == v
                and r['blocks'] == b) for b in sorted(set(expected_order))}
               for v in ('strided100k', 'strided160k')}
    result = {'comparisons': comparisons, 'median_kernel_us': medians}
    save(root/'comparison.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
