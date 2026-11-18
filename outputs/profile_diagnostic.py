"""Short GPU timing diagnostic; instrumented throughput is not a benchmark."""
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bench', root/'run_mtp_benchmark.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.OUT = root/'vulkan-profile-diagnostic'
bench.OUT.mkdir(exist_ok=True)
previous = json.loads((root/'mtp-100k-mtp1-f16-results/cold-102400-mtp1-request.json').read_text())
tail = bench.tokenize('\nWrite a Python aggregation function with validation and tests.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', True)
prompt = previous['prompt'][:1024-len(tail)] + tail
result = bench.run('short-profile-1024', prompt, False, 128)
assert len(prompt) == 1024
assert result['final']['timings']['prompt_n'] == 1024
assert result['final']['timings'].get('draft_n', 0) > 0
print('Short profiling request complete. Inspect vulkan-profile.log; do not compare instrumented tokens/s to normal benchmarks.', flush=True)
