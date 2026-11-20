"""Run four fresh-server 102400-token prefills in 512/1024/1024/512 order."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark_server import ROOT, TOOLS, server
import run_mtp_benchmark as probe


SOURCE_SHA256 = '572a0e822a86213d60e076b486c7f72a22d8040a322d69605b487130c63c31cd'


def file_sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    source = TOOLS / 'rocm-100k-mtp2-f16-results/cold-102400-mtp2-request.json'
    digest = file_sha256(source)
    if digest != SOURCE_SHA256:
        raise RuntimeError('Different source request')
    prompt = json.loads(source.read_text())['prompt']
    backend = ROOT / 'work/hip-kv-split-bin'
    order = [512, 1024, 1024, 512]
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        'order': order,
        'prompt_file': str(source.relative_to(ROOT)),
        'request_sha256': digest,
        'hip_sha256': file_sha256(backend / 'ggml-hip.dll'),
        'kv_split': 'double',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for position, microbatch in enumerate(order, 1):
        case = out / f'case-{position}-{microbatch}'
        expected = {
            '-c': 163840, '-np': 1, '-ngl': 99, '-fa': 'on', '-ctk': 'f16', '-ctv': 'f16',
            '-b': 2048, '-ub': microbatch, '-t': 8, '-tb': 8, '-td': 8, '-tbd': 8,
            '--spec-type': 'draft-mtp', '--spec-draft-n-max': 2, '--device': 'ROCm0',
        }
        with server(
            case,
            backend=backend,
            launcher=TOOLS / 'Start-KVSplitServer.ps1',
            arguments=['-KvSplit', 'double', '-MicroBatchSize', microbatch],
            expected=expected,
        ) as props:
            if props['default_generation_settings']['n_ctx'] != 163840:
                raise RuntimeError('Context mismatch')
            probe.OUT = case
            result = probe.run('cold', prompt, False, 512)
            timing = result['final']['timings']
            if file_sha256(case / 'cold-request.json') != digest:
                raise RuntimeError('Sent request differs from the original')
            if timing['cache_n'] != 0 or timing['prompt_n'] != 102400:
                raise RuntimeError('Cold request or prompt work differs')
            token_ids = json.dumps(result['generated_token_ids']).encode()
            rows.append({
                'position': position,
                'microbatch': microbatch,
                'timings': timing,
                'ttft_s': result['time_to_first_generated_token_seconds'],
                'wall_s': result['wall_seconds'],
                'output_sha256': hashlib.sha256(token_ids).hexdigest(),
            })
            (out / 'samples.json').write_text(json.dumps(rows, indent=2))
        log = (case / 'server.log').read_bytes()
        log = log.decode('utf-16' if log.startswith(b'\xff\xfe') else 'utf-8')
        if 'HIP_FA_KV_SPLIT mode=double blocks=8' not in log:
            raise RuntimeError('Missing eight-block dispatch marker')


if __name__ == '__main__':
    main()
