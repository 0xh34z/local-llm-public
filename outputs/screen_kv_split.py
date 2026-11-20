"""Numerical gates and repeated isolated attention timings for KV work splitting."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/kv-split-kernel-tests'
BACKEND = ROOT / 'work/hip-kv-split-bin'
NUMPY_PYTHON = Path(os.environ.get('NUMPY_PYTHON', sys.executable))


def main():
    if OUT.exists():
        raise RuntimeError('Screen output exists; preserve it and use a new path in the script')
    OUT.mkdir()
    env = os.environ.copy()
    env.update(PATH=';'.join(str(p) for p in (BACKEND,
        ROOT/'work/rocm10-runtime/Lib/site-packages/_rocm_sdk_core/bin',
        ROOT/'work/rocm10-runtime/Lib/site-packages/_rocm_sdk_libraries/bin'))+';'+env['PATH'],
        LLAMA_HIP_ATTENTION_PROBE='1', GGML_HIP_FA_EXPERIMENT='auto',
        GGML_HIP_FA_MMA_TILE='auto', GGML_HIP_OP_PROFILE='0')
    env.pop('GGML_BACKEND_PATH', None)
    modes = ('auto','half','double','quad')
    numerical = {}
    for mode in modes:
        env['GGML_HIP_FA_KV_SPLIT'] = mode
        dump = ROOT / ('work/kv-split-float64-'+mode)
        dump.mkdir(exist_ok=False)
        env['LLAMA_HIP_ATTENTION_DUMP'] = str(dump)
        with (OUT/(mode+'-test.log')).open('w') as log:
            completed = subprocess.run([str(BACKEND/'test-attention-probe.exe'),'test','-b','ROCm0',
                '-o','FLASH_ATTN_EXT'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        if completed.returncode not in (0,1):
            raise RuntimeError('Probe crashed: '+mode)
        with (OUT/(mode+'-oracle.log')).open('w') as log:
            subprocess.run([str(NUMPY_PYTHON),str(ROOT/'outputs/check_attention_float64.py'),
                '--dump-dir',str(dump),'--out',str(OUT/(mode+'-float64.json'))],stdout=log,
                stderr=subprocess.STDOUT,check=True,timeout=120)
        oracle = json.loads((OUT/(mode+'-float64.json')).read_text())
        error = oracle['backends']['gpu']['nmse_vs_float64']
        numerical[mode] = {'upstream_exit_code':completed.returncode, 'oracle_nmse':error,
                          'qualified':math.isfinite(error) and error<=0.0005 and all(oracle['inputs_equal'].values())}
        print(json.dumps({'event':'numerical', 'mode':mode, **numerical[mode]}),flush=True)
    env.pop('LLAMA_HIP_ATTENTION_DUMP',None)
    records=[]
    orders = [modes,tuple(reversed(modes)),('double','auto','quad','half')]
    for repeat, order in enumerate(orders,1):
        for mode in order:
            if not numerical[mode]['qualified']:
                continue
            env['GGML_HIP_FA_KV_SPLIT'] = mode
            path=OUT/f'{mode}-perf-{repeat}.log'
            with path.open('w') as log:
                subprocess.run([str(BACKEND/'test-attention-probe.exe'),'perf','-b','ROCm0',
                    '-o','FLASH_ATTN_EXT'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            log=path.read_text()
            lines=[line for line in log.splitlines() if 'FLASH_ATTN_EXT(' in line and 'nb=3,' in line and 'q_range=20' not in line]
            if len(lines)!=1:
                raise RuntimeError('Expected one regular Q3 timing')
            match=re.search(r'([0-9.]+) us/run',lines[0])
            if not match or 'HIP_FA_KV_SPLIT mode='+mode not in log:
                raise RuntimeError('Missing timing or split audit')
            records.append({'mode':mode,'repeat':repeat,'regular_q3_us':float(match[1])})
            print(json.dumps({'event':'perf',**records[-1]}),flush=True)
    summary={'numerical':numerical,'timings':records,'orders':orders,
        'backend_sha256':hashlib.sha256((BACKEND/'ggml-hip.dll').read_bytes()).hexdigest(),
        'median_q3_us':{m:statistics.median(r['regular_q3_us'] for r in records if r['mode']==m)
                        for m in modes if numerical[m]['qualified']}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'event':'screen_complete','medians_us':summary['median_q3_us']}),flush=True)


if __name__=='__main__':
    main()
