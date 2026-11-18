"""Repeated CPU-worker sweep on the same restored 100K ROCm/MTP prefix."""
import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / 'outputs'
BASE = 'http://127.0.0.1:9931'


def psquote(value):
    return "'" + str(value).replace("'", "''") + "'"


def powershell(command):
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                             "$ErrorActionPreference='Stop'; " + command],
                            text=True, capture_output=True, check=True)
    return result.stdout.strip()


def processes():
    raw = powershell('Get-CimInstance Win32_Process -Filter "Name = \'llama-server.exe\'" | '
                     'Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Depth 3')
    values = json.loads(raw) if raw else []
    if isinstance(values, dict):
        values = [values]
    return [p for p in values if re.search(r'--port\s+9931(?:\s|$)', p['CommandLine'] or '')]


def get(route):
    with urllib.request.urlopen(BASE + route, timeout=10) as response:
        return json.load(response)


def emit(event, **data):
    print(json.dumps({'event': event, **data}), flush=True)


def report(out, records):
    groups = {}
    for row in records:
        groups.setdefault(row['threads'], []).append(row)
    lines = ['# CPU thread sweep at a restored 100K prefix', '',
             '| Main / draft threads | Measured runs | Mean generation TPS | SD | Range TPS | Mean TTFT s |',
             '|---|---:|---:|---:|---:|---:|']
    summary = []
    for threads in sorted(groups):
        rows = groups[threads]
        rates = [r['tps'] for r in rows]
        item = {'threads': threads, 'runs': len(rows), 'mean_tps': statistics.mean(rates),
                'sd_tps': statistics.stdev(rates) if len(rates) > 1 else 0,
                'min_tps': min(rates), 'max_tps': max(rates),
                'mean_ttft_s': statistics.mean(r['ttft_s'] for r in rows)}
        summary.append(item)
        lines.append(f"| {threads} / {threads} | {len(rows)} | {item['mean_tps']:.3f} | "
                     f"{item['sd_tps']:.3f} | {min(rates):.3f}–{max(rates):.3f} | {item['mean_ttft_s']:.3f} |")
    lines += ['', 'Main and MTP-draft generation thread counts change together. Main and draft batch threads '
              'remain fixed at 8. All runs use ROCm0, F16 K/V, MTP 2, context 163840, one slot, '
              'batch 2048, microbatch 512, eight batch threads and log verbosity 3.', '',
              'Each server receives a discarded 64-token warmup, then three measured 256-token continuations. '
              'The same original saved slot is restored before every request. Requests use identical input tokens, '
              'temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repeat penalty 1, presence penalty 0, '
              'completed empty thinking blocks, and ignore_eos=true. Order: 8, 2, 6, 1, 4, 8. '
              'The second 8-thread block brackets time drift. Warmups and reload times are excluded from this table.', '',
              'No affinity pinning, power-limit or temperature control was applied. Repeats within a server are '
              'not independent sessions or diverse coding tasks. Reported range is a range of request averages, '
              'not an instantaneous minimum. This is a synthetic generation test at a populated prefix, '
              'not a cold-prefill or coding/tool-use quality benchmark. Batch-thread optimization is not tested.', '',
              'Each case directory contains verified live launch arguments, properties, the server log, '
              'restore responses, raw requests, streamed events and complete results. See summary.json for per-case '
              'numbers and output token hashes. The snapshot is benchmark state, not a user Pi conversation.']
    analysis = None
    if len(records) == 18 and len(groups) == 5:
        first = statistics.mean(r['tps'] for r in records if r['case'] == 1)
        last = statistics.mean(r['tps'] for r in records if r['case'] == 6)
        baseline = next(item['mean_tps'] for item in summary if item['threads'] == 8)
        best = max(summary, key=lambda item: item['mean_tps'])
        gain = (best['mean_tps'] / baseline - 1) * 100
        drift = (last / first - 1) * 100
        paired_outputs = all(len({r['output_sha256'] for r in records if r['repeat'] == rep}) == 1
                             for rep in (1, 2, 3))
        paired_drafts = all(len({(r['draft_n'], r['accepted_drafts']) for r in records
                                if r['repeat'] == rep}) == 1 for rep in (1, 2, 3))
        analysis = {'baseline_first_mean_tps': first, 'baseline_last_mean_tps': last,
                    'baseline_drift_percent': drift, 'best_threads': best['threads'],
                    'best_mean_gain_over_pooled_baseline_percent': gain,
                    'paired_output_tokens_identical': paired_outputs,
                    'paired_draft_counts_identical': paired_drafts,
                    'unique_measured_request_hashes': len({r['request_sha256'] for r in records}),
                    'measured_requests_averaging_at_least_15_tps': sum(r['tps'] >= 15 for r in records)}
        decision = ('Keep the existing eight-thread default. No convincing generation-speed gain was found.'
                    if gain < 3 or gain < abs(drift) else
                    f'The highest mean is at {best["threads"]} threads; confirm it across workloads before changing defaults.')
        lines[2:2] = [decision, '',
                       f'The highest grouped mean exceeds the pooled eight-thread baseline by {gain:.2f}%. '
                       f'The eight-thread control changed from {first:.3f} to {last:.3f} TPS '
                       f'({drift:+.2f}%) between the first and final blocks.', '']
        lines += ['', '## Matched-work verification', '',
                  f'All measured requests share one SHA256 hash: {records[0]["request_sha256"]}. '
                  f'Corresponding repetition numbers across all six server runs produced identical output '
                  f'token sequences: {paired_outputs}. Corresponding draft proposal/acceptance counts also '
                  f'matched: {paired_drafts}. The three repetitions produce three different output streams; '
                  'the SD in the table therefore includes content/acceptance differences, not just timing noise. '
                  'Comparisons at the same repetition number are paired by output stream.', '',
                  'Every measured request has 103220 input tokens, reuses 103194 tokens, processes 26 tokens, '
                  f'and generates 256 tokens without truncation. {analysis["measured_requests_averaging_at_least_15_tps"]} '
                  'of the 18 measured request averages reached 15 TPS. These are a different continuation '
                  'from earlier 15–16 TPS samples.', '',
                  'Prompt and draft batch threads remained at eight, so this does not evaluate batch-thread '
                  'counts or full cold-prefill speed. Main and draft generation workers were swept together; '
                  'their independent contributions and CPU affinity were not tested. CPU utilization and '
                  'performance with concurrent tools were not measured. The server was shut down after this run.']
    (out / 'summary.json').write_text(json.dumps({'groups': summary, 'records': records, 'analysis': analysis}, indent=2), encoding='utf-8')
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=TOOLS / 'thread-sweep-100k-results')
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('Existing run; choose a new output directory')
    if processes():
        raise RuntimeError('Port 9931 already has an inference server; no changes made')
    snapshot = json.loads((TOOLS / 'slot-cache-reuse-results' / 'snapshot.json').read_text())
    if not (ROOT / 'work' / 'slot-cache' / snapshot['filename']).is_file():
        raise RuntimeError('Saved 100K snapshot file is missing')
    spec = importlib.util.spec_from_file_location('throughput_probe', TOOLS / 'run_mtp_benchmark.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    order = [8, 2, 6, 1, 4, 8]
    manifest = {'order': order, 'batch_threads': 8, 'repeats': 3, 'warmup_output_tokens': 64,
                'measured_output_tokens': 256, 'snapshot': snapshot['filename'],
                'model_and_build': snapshot['server'], 'context': 163840,
                'batch': 2048, 'microbatch': 512, 'kv': 'f16', 'mtp': 2}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    records = []
    prompt = None
    expected_request_hash = None
    own_pid = None
    try:
        for case, threads in enumerate(order, 1):
            case_dir = out / f'case-{case:02d}-t{threads}'
            case_dir.mkdir()
            probe.OUT = case_dir
            launch_args = ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                           '"' + str(TOOLS / 'Start-ROCmServer.ps1') + '"',
                           '-Threads', str(threads), '-BatchThreads', '8', '-DraftThreads', str(threads),
                           '-LogPath', '"' + str(case_dir / 'server.log') + '"']
            command = ('Start-Process -FilePath powershell.exe -ArgumentList @(' +
                       ','.join(psquote(a) for a in launch_args) +
                       ') -WindowStyle Hidden -PassThru -RedirectStandardOutput ' +
                       psquote(case_dir / 'launcher-stdout.log') + ' -RedirectStandardError ' +
                       psquote(case_dir / 'launcher-stderr.log') + ' | Select-Object -ExpandProperty Id')
            # Windows PowerShell descendants can retain captured pipe handles.
            # Real files avoid waiting for a server-long EOF in communicate().
            with (case_dir / 'launch-control-output.txt').open('w') as launch_stdout, \
                    (case_dir / 'launch-control-stderr.log').open('w') as launch_stderr:
                subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                                "$ErrorActionPreference='Stop'; " + command],
                               stdout=launch_stdout, stderr=launch_stderr, check=True, timeout=30)
            launcher_pid = int((case_dir / 'launch-control-output.txt').read_text().strip())
            emit('loading', case=case, cases=len(order), threads=threads, launcher_pid=launcher_pid)
            deadline = time.monotonic() + 120
            while True:
                try:
                    if get('/health').get('status') == 'ok':
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError('Server failed to become healthy; inspect launcher logs')
                time.sleep(1)
            live = processes()
            if len(live) != 1:
                raise RuntimeError('Cannot uniquely identify live server')
            launch = live[0]
            own_pid = launch['ProcessId']
            flags = {'-t': str(threads), '-td': str(threads), '-tb': '8', '-tbd': '8',
                     '-c': '163840', '-b': '2048', '-ub': '512', '-ctk': 'f16', '-ctv': 'f16',
                     '--spec-type': 'draft-mtp', '--spec-draft-n-max': '2', '--device': 'ROCm0'}
            for flag, value in flags.items():
                match = re.search(r'(?:^|\s)' + re.escape(flag) + r'\s+(\S+)', launch['CommandLine'])
                if not match or match[1] != value:
                    raise RuntimeError(f'Live flag differs: {flag} expected {value}')
            props = get('/props')
            fingerprint = {key: props.get(key) for key in ('model_path', 'build_info')} | {
                'n_ctx': props['default_generation_settings']['n_ctx']}
            if fingerprint != snapshot['server']:
                raise RuntimeError('Snapshot model/build/context differs from live server')
            (case_dir / 'server-launch.json').write_text(json.dumps(launch, indent=2), encoding='utf-8')
            (case_dir / 'server-props.json').write_text(json.dumps(props, indent=2), encoding='utf-8')
            if prompt is None:
                suffix = probe.tokenize('\n<|im_end|>\n<|im_start|>user\nExplain how the implementation handles empty inputs.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', True)
                prompt = snapshot['completed_tokens'] + suffix
            for repetition in range(4):
                name = 'warmup' if repetition == 0 else f'repeat-{repetition}'
                if any(slot['is_processing'] for slot in get('/slots')):
                    raise RuntimeError('Server became busy; do not interrupt another request')
                started = time.perf_counter()
                restored = probe.post('/slots/0?action=restore', {'filename': snapshot['filename']})
                restore_wall = time.perf_counter() - started
                if restored.get('n_restored', 0) != 103194:
                    raise RuntimeError('Unexpected restored token count')
                (case_dir / (name + '-restore.json')).write_text(json.dumps({
                    'wall_seconds': restore_wall, 'response': restored}, indent=2), encoding='utf-8')
                result = probe.run(name, prompt, True, 64 if repetition == 0 else 256)
                timing = result['final']['timings']
                if timing['cache_n'] != 103194 or timing['prompt_n'] != 26 or timing.get('draft_n', 0) <= 0:
                    raise RuntimeError('Prefix reuse, new token count or MTP activity differs')
                if repetition == 0:
                    continue
                request_hash = hashlib.sha256((case_dir / (name + '-request.json')).read_bytes()).hexdigest()
                if expected_request_hash is None:
                    expected_request_hash = request_hash
                if request_hash != expected_request_hash:
                    raise RuntimeError('Measured requests differ across repetitions')
                row = {'case': case, 'threads': threads, 'batch_threads': 8, 'repeat': repetition,
                       'tps': timing['predicted_per_second'], 'ttft_s': result['time_to_first_generated_token_seconds'],
                       'wall_s': result['wall_seconds'], 'prompt_ms': timing['prompt_ms'],
                       'cache_n': timing['cache_n'], 'prompt_n': timing['prompt_n'],
                       'draft_n': timing['draft_n'], 'accepted_drafts': timing['draft_n_accepted'],
                       'request_sha256': request_hash,
                       'output_sha256': hashlib.sha256(json.dumps(result['generated_token_ids']).encode()).hexdigest()}
                records.append(row)
                report(out, records)
                emit('sample', **row)
            powershell('& ' + psquote(TOOLS / 'Stop-ROCmServer.ps1'))
            deadline = time.monotonic() + 20
            while processes():
                if time.monotonic() >= deadline:
                    raise RuntimeError('Server did not exit')
                time.sleep(0.5)
            own_pid = None
            emit('case_complete', case=case, threads=threads,
                 mean_tps=statistics.mean(r['tps'] for r in records if r['case'] == case))
        emit('sweep_complete', measured_runs=len(records), report=str(out / 'REPORT.md'))
    finally:
        if own_pid is not None:
            current = processes()
            if any(p['ProcessId'] == own_pid for p in current):
                powershell('& ' + psquote(TOOLS / 'Stop-ROCmServer.ps1'))


if __name__ == '__main__':
    main()
