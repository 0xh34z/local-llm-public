"""Save a synthetic benchmark slot, then verify reuse after a real server restart."""
import argparse
import importlib.util
import json
import re
import time
import urllib.request
from pathlib import Path

BASE = 'http://127.0.0.1:9931'


def get(route):
    with urllib.request.urlopen(BASE + route, timeout=30) as response:
        return json.load(response)


def post(route, body):
    request = urllib.request.Request(BASE + route, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def fingerprint(props):
    return {key: props.get(key) for key in ('model_path', 'build_info')} | {
        'n_ctx': props['default_generation_settings']['n_ctx']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['save', 'warm', 'restore', 'rewind', 'deep-prefill'])
    parser.add_argument('--benchmark-dir', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--filename', default='synthetic-100k.bin')
    parser.add_argument('--snapshot-dir', type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.filename) or args.filename in ('.', '..'):
        parser.error('filename must be a simple local file name')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(slot['is_processing'] for slot in get('/slots')):
        raise RuntimeError('Server is busy; no slot action submitted')
    props = get('/props')
    metadata_path = (args.snapshot_dir or args.out_dir) / 'snapshot.json'
    result_path = args.out_dir / (args.action + '-slot-action.json')
    if result_path.exists():
        raise RuntimeError('Action already recorded; use a fresh output directory')

    if args.action == 'save':
        if metadata_path.exists():
            raise RuntimeError('Snapshot metadata already exists')
        request = json.loads((args.benchmark_dir / 'append-102400-mtp2-request.json').read_text())
        result = json.loads((args.benchmark_dir / 'append-102400-mtp2-result.json').read_text())
        if result['final']['truncated'] or len(result['generated_token_ids']) != 256:
            raise RuntimeError('Source benchmark did not complete successfully')
        metadata = {'filename': args.filename, 'server': fingerprint(props),
                    'source_dir': str(args.benchmark_dir.resolve()),
                    'completed_tokens': request['prompt'] + result['generated_token_ids']}
        started = time.perf_counter()
        response = post('/slots/0?action=save', {'filename': args.filename})
        action_result = {'wall_seconds': time.perf_counter() - started, 'response': response}
        if response.get('n_saved', 0) < 102400:
            raise RuntimeError('Saved slot does not contain the expected 100K history')
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
    else:
        metadata = json.loads(metadata_path.read_text())
        if metadata['filename'] != args.filename or metadata['server'] != fingerprint(props):
            raise RuntimeError('Snapshot model/build/context fingerprint does not match server')
        if args.action == 'warm':
            action_result = {'wall_seconds': 0, 'response': {'source': 'live slot; no disk restore'}}
        else:
            started = time.perf_counter()
            response = post('/slots/0?action=restore', {'filename': args.filename})
            action_result = {'wall_seconds': time.perf_counter() - started, 'response': response}
            if response.get('n_restored', 0) < 102400:
                raise RuntimeError('Restored slot does not contain the expected 100K history')
    result_path.write_text(json.dumps(action_result, indent=2), encoding='utf-8')
    print(json.dumps({'action': args.action, **action_result}), flush=True)

    if args.action != 'save':
        spec = importlib.util.spec_from_file_location('throughput_probe',
                                                     Path(__file__).with_name('run_mtp_benchmark.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.OUT = args.out_dir
        if args.action == 'deep-prefill':
            # Fixed-length new prompt work at a populated 100K prefix; no claim
            # that this substitutes for a complete cold 100K benchmark.
            source = json.loads((args.benchmark_dir / 'cold-102400-mtp2-request.json').read_text())
            header = module.tokenize('\n<|im_end|>\n<|im_start|>user\nReview these additional synthetic modules:\n', True)
            tail = module.tokenize('\nSummarize the validation requirements.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', True)
            extra = header + source['prompt'][100:100 + 8192 - len(header) - len(tail)] + tail
            assert len(extra) == 8192
            prompt = metadata['completed_tokens'] + extra
        elif args.action in ('restore', 'warm'):
            suffix = module.tokenize('\n<|im_end|>\n<|im_start|>user\nExplain how the implementation handles empty inputs.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', True)
            prompt = metadata['completed_tokens'] + suffix
        else:
            # Controlled edit near the end of the saved generation. This probes
            # hybrid-state rewind; it is not claimed to reproduce a Pi request.
            prompt = metadata['completed_tokens'].copy()
            position = len(prompt) - 128
            replacement = module.tokenize(' changed ')[0]
            if prompt[position] == replacement:
                replacement = module.tokenize(' different ')[0]
            prompt[position] = replacement
            (args.out_dir / 'rewind-edit.json').write_text(json.dumps({
                'first_changed_token': position, 'input_tokens': len(prompt)}))
        result = module.run(args.action + '-continuation', prompt, True,
                            1 if args.action == 'deep-prefill' else 64)
        timings = result['final']['timings']
        if args.action in ('restore', 'warm', 'deep-prefill') and timings['cache_n'] < 102400:
            raise RuntimeError('Disk restoration did not preserve the expected 100K prefix')


if __name__ == '__main__':
    main()
