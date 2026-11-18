"""Compare eight-row vocabulary projection with stock at two actual prefix lengths."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from benchmark_server import ROOT, TOOLS, server
from run_thread_sweep import get
import run_mtp_benchmark as probe


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('screen', type=Path)
    args = parser.parse_args()
    screen = read(args.screen / 'samples.json')
    if [r['mode'] for r in screen] != ['auto','4','8','auto','8','4','auto'] or any(
            r['incorrect'] or r['checked_logits'] != 248320 for r in screen):
        raise RuntimeError('Incomplete projection screen')
    cases = {
        '103k': ('synthetic-helper-check.bin', 103194,
                 TOOLS / 'kv-split-100k-results-verified/case-02-double/repeat-1-request.json'),
        '160k': ('synthetic-160k-kv-split.bin', 160511,
                 TOOLS / 'populated-160k-kv-split-results/case-02-double/measured-request.json')}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    baseline = read(TOOLS / 'slot-cache-reuse-results/snapshot.json')['server']
    backend = ROOT / 'work/hip-vocab-probe/bin'
    samples = []
    order = ['auto','8','auto']
    (out/'manifest.json').write_text(json.dumps({'order': order, 'cases': {
        name: str(file.relative_to(TOOLS)) for name, (_, _, file) in cases.items()},
        'hip_sha256': hashlib.sha256((backend/'ggml-hip.dll').read_bytes()).hexdigest(),
        'screen': str(args.screen), 'kv_split': 'double'}, indent=2))
    for index, mode in enumerate(order):
        os.environ['GGML_HIP_VOCAB_ROWS'] = mode
        case = out / f'case-{index+1}-{mode}'
        with server(case, backend=backend, launcher=TOOLS/'Start-KVSplitServer.ps1',
                    arguments=['-KvSplit', 'double'], expected={'-ctk':'f16','-ctv':'f16',
                    '-c':163840,'-np':1,'-b':2048,'-ub':512,'-t':8,'-td':8,'-tb':8,'-tbd':8,
                    '--device':'ROCm0','--spec-type':'draft-mtp','--spec-draft-n-max':2}) as props:
            if (props['model_path'], props['build_info'], props['default_generation_settings']['n_ctx']) != (
                baseline['model_path'], baseline['build_info'], baseline['n_ctx']):
                raise RuntimeError('Snapshot/model mismatch')
            probe.OUT = case
            for trial, name in enumerate(['103k','103k','160k','160k','103k']):
                filename, cached, request = cases[name]
                prompt = read(request)['prompt']
                tag = f'{trial}-{name}'
                if any(s['is_processing'] for s in get('/slots')):
                    raise RuntimeError('Server busy')
                restore = probe.post('/slots/0?action=restore', {'filename': filename})
                (case/(tag+'-restore.json')).write_text(json.dumps(restore))
                if restore['n_restored'] != cached:
                    raise RuntimeError('Unexpected restore length')
                result = probe.run(tag, prompt, True, 48 if trial == 0 else 256)
                timing = result['final']['timings']
                if (timing['cache_n'], timing['prompt_n']) != (cached, len(prompt)-cached):
                    raise RuntimeError('Unexpected prompt work')
                if trial:
                    samples.append({'case':index+1, 'mode':mode, 'tokens':len(prompt), 'trial':trial,
                        'tps':timing['predicted_per_second'], 'ttft_s':result['time_to_first_generated_token_seconds'],
                        'draft':timing['draft_n'], 'accepted':timing['draft_n_accepted'],
                        'tokens_sha256':hashlib.sha256(json.dumps(result['generated_token_ids']).encode()).hexdigest()})
                    (out/'samples.json').write_text(json.dumps(samples, indent=2))
        log_bytes = (case/'server.log').read_bytes()
        log = log_bytes.decode('utf-16' if log_bytes.startswith(b'\xff\xfe') else 'utf-8')
        if f'HIP_VOCAB_ROWS mode={mode}' not in log or 'HIP_FA_KV_SPLIT mode=double blocks=8' not in log:
            raise RuntimeError('Dispatch markers missing in server log')


if __name__ == '__main__':
    main()
