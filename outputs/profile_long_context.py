"""Collect operation timings at the target depth; instrumented speed is diagnostic."""
import importlib.util
import json
import shutil
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bench', root/'run_mtp_benchmark.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.OUT = root/'vulkan-profile-100k-diagnostic'
bench.OUT.mkdir(exist_ok=True)
if (bench.OUT/'manifest.json').exists():
    raise RuntimeError('Diagnostic output exists; preserve it and select a new directory')
with urllib.request.urlopen(bench.BASE+'/slots', timeout=20) as r:
    if any(s['is_processing'] for s in json.load(r)):
        raise RuntimeError('Server is busy')
with urllib.request.urlopen(bench.BASE+'/props', timeout=20) as r:
    props = json.load(r)
assert props['default_generation_settings']['n_ctx'] == 163840
log = root/'vulkan-profile.log'
manifest = {'kind': 'instrumented operation profile, not an ordinary throughput benchmark',
            'populated_input_tokens': 102400, 'log_start_bytes': log.stat().st_size,
            'initial_output_tokens': 128, 'followup_output_tokens': 128,
            'server_props': props}
(bench.OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
previous = json.loads((root/'mtp-100k-mtp1-f16-results/cold-102400-mtp1-request.json').read_text())
prompt = previous['prompt']
assert len(prompt) == 102400
cold = bench.run('profile-102400', prompt, False, 128)
assert cold['final']['timings']['prompt_n'] == 102400
followup = bench.tokenize('\n<|im_end|>\n<|im_start|>user\nAdd property-based test examples and explain their invariants.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',True)
bench.run('profile-102400-followup',prompt+cold['generated_token_ids']+followup,True,128)
shutil.copyfile(log,bench.OUT/'vulkan-profile-snapshot.log')
print('Long-context profiling complete; raw timing log saved.',flush=True)
