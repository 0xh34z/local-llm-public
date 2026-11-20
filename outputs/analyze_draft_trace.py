"""Per-request MTP draft() time from trace-level server logs (Start-CeilingServer.ps1 -LogVerbosity 4).

llama.cpp prints cumulative `statistics draft-mtp: ... dur(b,g,a) = begin, draft, accept ms` after each request.
The draft figure covers common_speculative_draft(): the single-token MTP decodes and their on-GPU sampling, plus
waiting for any GPU work still queued before them (the catch-up pass in process() is launched just before and is
not timed separately). Differences between consecutive lines give one value per request; the first request of
each case is the excluded warmup, matching run_ceiling_server.py's sample order.
"""
import argparse
import json
from pathlib import Path
import re

STAT = re.compile(r'statistics\s+draft-mtp: #calls\(b,g,a\) = +(\d+) +(\d+) +(\d+).*dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms')


def per_request(log):
    rows, previous = [], (0, 0.0, 0.0, 0.0)
    raw = log.read_bytes()
    text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8', errors='replace')
    for line in text.splitlines():  # Tee-Object in Windows PowerShell writes UTF-16 LE
        m = STAT.search(line)
        if not m:
            continue
        calls, b, g, a = int(m[2]), float(m[4]), float(m[5]), float(m[6])
        rows.append(dict(draft_calls=calls - previous[0], begin_ms=b - previous[1], draft_ms=g - previous[2],
                         accept_ms=a - previous[3]))
        previous = (calls, b, g, a)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='run_ceiling_server.py output directory')
    args = p.parse_args()
    samples = json.loads((args.run/'samples.json').read_text(encoding='utf-8'))
    out = []
    for case in sorted(c for c in args.run.glob('case-*') if '-trace' in c.name):
        number = int(case.name.split('-')[1])
        rows = [r for r in samples if r['case'] == number]
        timings = per_request(case/'server.log')
        # warmups (the short one and one after the snapshot restore) have no sample row; align greedily:
        # a request with c verify cycles makes c-1 or c-2 draft() calls (the first token is never drafted)
        pairs, queue = [], list(timings)
        for row in rows:
            while queue and not 0 <= row['cycles'] - queue[0]['draft_calls'] <= 3:
                queue.pop(0)
            if not queue:
                raise RuntimeError(f'{case.name}: no statistics line for {row["context"]}/{row["workload"]}')
            pairs.append((row, queue.pop(0)))
        for row, t in pairs:
            cycles = row['cycles']
            out.append(dict(case=number, backend=row.get('label', row['backend']), mtp=row['mtp'], context=row['context'],
                            workload=row['workload'], cycles=cycles,
                            ms_per_cycle=row['predicted_ms']/cycles, draft_ms_per_call=t['draft_ms']/t['draft_calls'],
                            accept_ms_per_call=t['accept_ms']/t['draft_calls'], **t))
    (args.run/'draft-trace.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    groups = {}
    for r in out:
        g = groups.setdefault((r['backend'], r['context'], r['workload']), dict(cycle=0.0, draft=0.0, calls=0, cycles=0))
        g['cycle'] += r['ms_per_cycle']*r['cycles']; g['cycles'] += r['cycles']
        g['draft'] += r['draft_ms']; g['calls'] += r['draft_calls']
    print('backend context workload    ms/cycle  draft()/call  calls/cycle')
    for (backend, context, workload), g in sorted(groups.items()):
        print(f"{backend:7s} {context:7s} {workload:11s} {g['cycle']/g['cycles']:8.1f} {g['draft']/g['calls']:12.1f} "
              f"{g['calls']/g['cycles']:11.2f}")


if __name__ == '__main__':
    main()
