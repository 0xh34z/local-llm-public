"""Raw-token llama-server performance probe. Python 3, standard library only.

This is NOT a chat-template, tool-use, or quality benchmark. Supply a substantial
UTF-8 source-code corpus; it is truncated, never repeated/padded. Run locally.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def post(base, route, payload):
    req = urllib.request.Request(
        base.rstrip('/') + route,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=86400) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--label', required=True, help='Build/model/KV/batch/MTP identifier')
    p.add_argument('--contexts', type=int, nargs='+', default=[102400, 131072, 163840])
    p.add_argument('--server-context', type=int, default=196608)
    p.add_argument('--decode', type=int, default=1024)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    raw = a.corpus.read_bytes()
    text = raw.decode('utf-8-sig')
    corpus_hash = hashlib.sha256(raw).hexdigest()
    tokens = post(a.url, '/tokenize', {
        'content': text, 'add_special': False, 'parse_special': False,
    })['tokens']
    suffix = post(a.url, '/tokenize', {
        'content': '\nContinue the code review with concrete changes and tests.\n',
        'add_special': False, 'parse_special': False,
    })['tokens']
    if min(a.contexts) < 100000:
        p.error('This probe requires at least 100,000 populated prompt tokens.')
    if len(tokens) < max(a.contexts):
        p.error(f'Corpus has only {len(tokens)} tokens; supply more real source text.')
    if max(a.contexts) + 2 * a.decode + len(suffix) + 16 > a.server_context:
        p.error('Increase --server-context and the server -c to retain output headroom.')
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def run(prompt, cache, phase, repeat, depth):
        started = time.perf_counter()
        r = post(a.url, '/completion', {
            'prompt': prompt, 'id_slot': 0, 'cache_prompt': cache,
            'n_predict': a.decode, 'temperature': 0, 'seed': 42,
            'ignore_eos': True, 'return_tokens': True, 'stream': False,
        })
        elapsed = time.perf_counter() - started
        if 'error' in r:
            raise RuntimeError(r['error'])
        timing = r.get('timings', {})
        row = {
            'label': a.label, 'phase': phase, 'repeat': repeat,
            'base_depth': depth, 'input_tokens': len(prompt),
            'corpus_sha256': corpus_hash,
            'prompt_tokens_sha256': hashlib.sha256(
                json.dumps(prompt, separators=(',', ':')).encode()).hexdigest(),
            'wall_seconds': elapsed, 'timings': timing,
            'tokens_cached': r.get('tokens_cached'),
            'tokens_evaluated': r.get('tokens_evaluated'),
            'truncated': r.get('truncated'), 'stop_type': r.get('stop_type'),
            'generated_tokens': len(r.get('tokens', [])),
            'content': r.get('content'),
        }
        with a.out.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'content'}), flush=True)
        if r.get('truncated'):
            raise RuntimeError('Context was truncated; reject this run.')
        if len(r.get('tokens', [])) != a.decode:
            raise RuntimeError('Output length differs from requested length; inspect saved row.')
        if not cache and timing.get('prompt_n') != len(prompt):
            raise RuntimeError('Cold prompt_n differs from input length; inspect cache/build behavior.')
        return r

    for depth in a.contexts:
        prefix = tokens[:depth]
        for repeat in range(1, a.repeats + 1):
            cold = run(prefix, False, 'cold', repeat, depth)
            # Exact append of all sampled tokens avoids reconstructing their text.
            # This is an append-only continuation, not a repeated-answer cache hit.
            warm_prompt = prefix + cold['tokens'] + suffix
            run(warm_prompt, True, 'append', repeat, depth)


if __name__ == '__main__':
    main()
