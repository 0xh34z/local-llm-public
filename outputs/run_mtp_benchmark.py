"""Reproducible synthetic code throughput probe for an existing llama-server."""
import argparse
import hashlib
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

BASE = 'http://127.0.0.1:9931'
OUT = Path(__file__).resolve().parent / 'mtp-100k-results'
OUT.mkdir(exist_ok=True)


def post(route, data):
    req = urllib.request.Request(BASE + route, json.dumps(data).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.load(r)


def tokenize(text, special=False):
    return post('/tokenize', {'content': text, 'add_special': False,
                             'parse_special': special})['tokens']


def run(name, prompt, cache, count, temperature=0.6):
    body = {'prompt': prompt, 'id_slot': 0, 'cache_prompt': cache,
            'n_predict': count, 'temperature': temperature, 'top_k': 20,
            'top_p': 0.95, 'min_p': 0.0, 'seed': 42,
            'repeat_penalty': 1.0, 'presence_penalty': 0.0,
            'ignore_eos': True, 'stream': True, 'return_tokens': True,
            'return_progress': True, 'timings_per_token': True}
    (OUT / (name + '-request.json')).write_text(json.dumps(body), encoding='utf-8')
    started = time.perf_counter()
    print(json.dumps({'event': 'start', 'name': name,
                      'input_tokens': len(prompt), 'output_budget': count}), flush=True)
    req = urllib.request.Request(BASE + '/completion', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    first = None
    last_report = started
    content = []
    ids = []
    final = None
    with (OUT / (name + '-events.jsonl')).open('w', encoding='utf-8') as log:
        with urllib.request.urlopen(req, timeout=7200) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                raw = line[5:].strip()
                if raw == b'[DONE]':
                    continue
                event = json.loads(raw)
                now = time.perf_counter()
                log.write(json.dumps({'elapsed_s': now-started, 'response': event}) + '\n')
                log.flush()
                if 'error' in event:
                    raise RuntimeError(event['error'])
                if not event.get('stop') and 'prompt_progress' not in event:
                    part = event.get('content', '')
                    token_ids = event.get('tokens', [])
                    if first is None and (part or token_ids):
                        first = now-started
                    content.append(part)
                    ids.extend(token_ids)
                if event.get('stop'):
                    final = event
                if now-last_report >= 20 or event.get('stop'):
                    print(json.dumps({'event': 'progress', 'name': name,
                        'elapsed_s': round(now-started, 2), 'generated_ids': len(ids),
                        'prompt_progress': event.get('prompt_progress'),
                        'timings': event.get('timings')}), flush=True)
                    last_report = now
    if final is None:
        raise RuntimeError('Stream ended without final response')
    result = {'name': name, 'input_tokens': len(prompt),
              'wall_seconds': time.perf_counter()-started,
              'time_to_first_generated_token_seconds': first,
              'generated_token_ids': ids, 'content': ''.join(content), 'final': final}
    (OUT / (name + '-result.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'complete', 'name': name, 'wall_seconds': result['wall_seconds'],
                      'ttft_seconds': first, 'generated_ids': len(ids),
                      'timings': final.get('timings'), 'truncated': final.get('truncated')}), flush=True)
    if final.get('truncated'):
        raise RuntimeError('Context truncated')
    if len(ids) != count:
        raise RuntimeError('Unexpected streamed token count; inspect result before continuing')
    return result


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-tokens', type=int, default=102400)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--mtp-draft', type=int, required=True)
    parser.add_argument('--expected-context', type=int, required=True)
    parser.add_argument('--expected-kv', choices=['f16', 'q8_0'])
    args = parser.parse_args()
    if args.mtp_draft < 0:
        parser.error('--mtp-draft must be zero (disabled) or positive')
    if args.out_dir:
        OUT = args.out_dir.resolve()
        OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Output directory already contains a run; choose a new directory')
    proc_result = subprocess.run([
        'powershell.exe', '-NoProfile', '-Command',
        "Get-CimInstance Win32_Process -Filter \"Name = 'llama-server.exe'\" | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Depth 3"
    ], capture_output=True, text=True, check=True)
    processes = json.loads(proc_result.stdout)
    if isinstance(processes, dict):
        processes = [processes]
    matches = [p for p in processes if re.search(r'--port\s+9931(?:\s|$)', p['CommandLine'])]
    if len(matches) != 1:
        raise RuntimeError('Cannot uniquely identify server on port 9931')
    launch = matches[0]
    command = launch['CommandLine']
    draft_match = re.search(r'--spec-draft-n-max\s+(\d+)', command)
    context_match = re.search(r'(?:^|\s)-c\s+(\d+)', command)
    cache_k_match = re.search(r'(?:^|\s)(?:-ctk|--cache-type-k)\s+(\S+)', command)
    cache_v_match = re.search(r'(?:^|\s)(?:-ctv|--cache-type-v)\s+(\S+)', command)
    cache_k = cache_k_match[1] if cache_k_match else 'unspecified'
    cache_v = cache_v_match[1] if cache_v_match else 'unspecified'
    if args.expected_kv and (cache_k != args.expected_kv or cache_v != args.expected_kv):
        raise RuntimeError('Live K/V cache launch arguments differ from expected precision')
    if args.mtp_draft == 0:
        if not re.search(r'--spec-type\s+none(?:\s|$)', command):
            raise RuntimeError('Disabled-drafting benchmark requires --spec-type none')
    elif not draft_match or int(draft_match[1]) != args.mtp_draft or '--spec-type draft-mtp' not in command:
        raise RuntimeError('Live MTP launch arguments do not match requested benchmark label')
    if not context_match or int(context_match[1]) != args.expected_context:
        raise RuntimeError('Live context allocation does not match expectation')
    (OUT / 'server-launch.json').write_text(json.dumps(launch, indent=2), encoding='utf-8')
    with urllib.request.urlopen(BASE + '/props', timeout=30) as response:
        props = json.load(response)
    if props['default_generation_settings']['n_ctx'] != args.expected_context:
        raise RuntimeError('Server-reported context differs from expected allocation')
    if args.prompt_tokens + 512 + 256 + 128 > args.expected_context:
        raise RuntimeError('Insufficient context headroom for this benchmark')
    with urllib.request.urlopen(BASE + '/slots', timeout=30) as response:
        if any(s['is_processing'] for s in json.load(response)):
            raise RuntimeError('Server is busy; no benchmark request was submitted')
    (OUT / 'server-props.json').write_text(json.dumps(props, indent=2), encoding='utf-8')
    # Unique, deterministic modules; no user files or external data are used.
    modules = []
    for i in range(2200):
        modules.append(f'''# file: package/module_{i:04d}.py
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class Record{i}:
    key: str
    amount: int
    enabled: bool = True

def aggregate_{i}(records: Iterable[Record{i}], limit: int = {i+17}) -> dict[str, int]:
    totals: dict[str, int] = {{}}
    for record in records:
        if record.enabled and 0 <= record.amount <= limit:
            totals[record.key] = totals.get(record.key, 0) + record.amount
    return dict(sorted(totals.items()))

def test_aggregate_{i}():
    rows = [Record{i}("alpha", {i % 11}), Record{i}("beta", -1)]
    assert aggregate_{i}(rows) == {{"alpha": {i % 11}}}

''')
    corpus = ''.join(modules)
    (OUT / 'synthetic-corpus.py.txt').write_text(corpus, encoding='utf-8')
    corpus_ids = tokenize(corpus)
    header = tokenize('<|im_start|>system\nYou are a coding assistant.\n<|im_end|>\n<|im_start|>user\nReview this synthetic Python package:\n', True)
    tail = tokenize('\nDesign a robust generic aggregation utility for these records. Write its Python implementation and tests, covering validation, empty inputs, stable ordering and error handling.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', True)
    room = args.prompt_tokens-len(header)-len(tail)
    if room <= 0:
        raise RuntimeError('Prompt length must exceed wrapper token count')
    if len(corpus_ids) < room:
        raise RuntimeError('Insufficient synthetic corpus')
    prompt = header + corpus_ids[:room] + tail
    (OUT / 'manifest.json').write_text(json.dumps({
        'kind': 'synthetic repeated-structure unique Python modules; performance only',
        'input_tokens': len(prompt), 'corpus_sha256': hashlib.sha256(corpus.encode()).hexdigest(),
        'model': props.get('model_path'), 'build': props.get('build_info'),
        'mtp': ('process command line verified --spec-type none' if args.mtp_draft == 0
                else f'process command line verified draft-mtp, n-max={args.mtp_draft}'),
        'allocated_context': args.expected_context,
        'cache_type_k': cache_k, 'cache_type_v': cache_v,
        'server_launch': launch,
        'sampling': 'temperature .6, top_p .95, top_k 20, min_p 0, seed 42',
        'forced_length': True}, indent=2), encoding='utf-8')
    cold = run(f'cold-{args.prompt_tokens}-mtp{args.mtp_draft}', prompt, False, 512)
    if cold['final']['timings']['prompt_n'] != args.prompt_tokens:
        raise RuntimeError('Cold request did not process the full requested input')
    if args.mtp_draft > 0 and cold['final']['timings'].get('draft_n', 0) <= 0:
        raise RuntimeError('No MTP drafting recorded; configuration requires inspection')
    if args.mtp_draft == 0 and cold['final']['timings'].get('draft_n', 0) != 0:
        raise RuntimeError('Drafting recorded in disabled-drafting benchmark')
    followup = tokenize('\n<|im_end|>\n<|im_start|>user\nAdd property-based test examples and explain their invariants.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', True)
    run(f'append-{args.prompt_tokens}-mtp{args.mtp_draft}', prompt+cold['generated_token_ids']+followup, True, 256)


if __name__ == '__main__':
    main()
