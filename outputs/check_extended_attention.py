"""Independent numerical checks for strided and 160K KV attention fixtures."""
import json
import math
import os
import subprocess

from screen_kv_split import ROOT, BACKEND, NUMPY_PYTHON


def main():
    out=ROOT/'outputs/kv-split-extended-numerics'
    out.mkdir(exist_ok=False)
    env=os.environ.copy()
    env.update(PATH=';'.join(str(p) for p in (BACKEND,
        ROOT/'work/rocm10-runtime/Lib/site-packages/_rocm_sdk_core/bin',
        ROOT/'work/rocm10-runtime/Lib/site-packages/_rocm_sdk_libraries/bin'))+';'+env['PATH'],
        LLAMA_HIP_ATTENTION_PROBE='1', GGML_HIP_FA_EXPERIMENT='auto',
        GGML_HIP_FA_MMA_TILE='auto', GGML_HIP_OP_PROFILE='0')
    env.pop('GGML_BACKEND_PATH',None)
    records=[]
    for variant in ('strided100k','strided160k','contiguous160k'):
        for mode in ('auto','double','quad'):
            name=variant+'-'+mode
            dump=ROOT/('work/kv-split-extended-'+name)
            dump.mkdir(exist_ok=False)
            env.update(LLAMA_HIP_ATTENTION_VARIANT=variant, GGML_HIP_FA_KV_SPLIT=mode,
                       LLAMA_HIP_ATTENTION_DUMP=str(dump))
            with (out/(name+'-test.log')).open('w') as log:
                completed=subprocess.run([str(BACKEND/'test-attention-probe.exe'),'test','-b','ROCm0',
                    '-o','FLASH_ATTN_EXT'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            if completed.returncode not in (0,1):
                raise RuntimeError('Probe crashed: '+name)
            with (out/(name+'-oracle.log')).open('w') as log:
                subprocess.run([str(NUMPY_PYTHON),str(ROOT/'outputs/check_attention_float64.py'),
                    '--dump-dir',str(dump),'--out',str(out/(name+'-float64.json'))],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
            oracle=json.loads((out/(name+'-float64.json')).read_text())
            error=oracle['backends']['gpu']['nmse_vs_float64']
            audit='HIP_FA_KV_SPLIT mode='+mode+' ' in (out/(name+'-test.log')).read_text()
            records.append({'variant':variant,'mode':mode,'upstream_exit_code':completed.returncode,
                'gpu_nmse':error,'cpu_nmse':oracle['backends']['cpu']['nmse_vs_float64'],
                'split_audit_present':audit,
                'qualified':audit and math.isfinite(error) and error<=0.0005 and all(oracle['inputs_equal'].values())})
            (out/'summary.json').write_text(json.dumps(records,indent=2))
            print(json.dumps(records[-1]),flush=True)


if __name__=='__main__':
    main()
