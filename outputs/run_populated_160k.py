"""Populate 128K or 160K actual input tokens from the saved 103K synthetic prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from benchmark_server import server, ROOT, TOOLS
from run_thread_sweep import get, emit
import run_mtp_benchmark as probe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['auto','double'],default='double')
    parser.add_argument('--input-tokens',type=int,choices=[128000,160000],default=160000)
    parser.add_argument('--out-dir',type=Path)
    parser.add_argument('--slot-filename')
    args=parser.parse_args()
    target=args.input_tokens
    label=f'{target//1000}k'
    out=(args.out_dir or TOOLS/f'populated-{label}-kv-split-results').resolve()
    out.mkdir(exist_ok=False,parents=True)
    saved_filename=args.slot_filename or f'synthetic-{label}-kv-split.bin'
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',saved_filename) or saved_filename in ('.','..'):
        raise RuntimeError('Slot filename must be a simple local filename')
    if (ROOT/'work/slot-cache'/saved_filename).exists():
        raise RuntimeError('Snapshot filename already exists; preserve it and choose another')
    snapshot=json.loads((TOOLS/'slot-cache-reuse-results/snapshot.json').read_text())
    source=json.loads((TOOLS/'rocm-100k-mtp2-f16-results/cold-102400-mtp2-request.json').read_text())
    if args.split=='double':
        checks=json.loads((TOOLS/'kv-split-extended-numerics/summary.json').read_text())
        relevant=[r for r in checks if r['mode']=='double']
        if len(relevant)!=3 or not all(r['qualified'] for r in relevant):
            raise RuntimeError('Extended numerical gate did not qualify')
    backend=ROOT/'work/hip-kv-split-bin'
    expected={'-c':163840,'-b':2048,'-ub':512,'-ctk':'f16','-ctv':'f16','-t':8,'-td':8,
              '-tb':8,'-tbd':8,'--spec-draft-n-max':2,'--device':'ROCm0'}
    with server(out/'server',backend=backend,launcher=TOOLS/'Start-KVSplitServer.ps1',
                arguments=['-KvSplit',args.split],expected=expected) as props:
        fingerprint={k:props.get(k) for k in ('model_path','build_info')}|{'n_ctx':props['default_generation_settings']['n_ctx']}
        if fingerprint!=snapshot['server']:
            raise RuntimeError('Snapshot fingerprint mismatch')
        probe.OUT=out
        header=probe.tokenize('\n<|im_end|>\n<|im_start|>user\nReview these additional synthetic modules:\n',True)
        tail=probe.tokenize('\nDesign validation and error-handling tests for this extended package.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',True)
        room=target-len(snapshot['completed_tokens'])-len(header)-len(tail)
        prompt=snapshot['completed_tokens']+header+source['prompt'][100:100+room]+tail
        assert len(prompt)==target
        manifest={'input_tokens':len(prompt),'allocated_context':163840,'initial_output_tokens':512,
            'followup_output_tokens':256,'reused_prefix_expected':103194,'split':args.split,
            'model_and_build':snapshot['server'],'kv':'f16','mtp':2,'batch':2048,'microbatch':512,
            'threads':8,'scope':'Synthetic performance only; extension from saved prefix, not full cold prefill',
            'backend_sha256':hashlib.sha256((backend/'ggml-hip.dll').read_bytes()).hexdigest(),
            'bracket_order':['auto','double','auto'],'bracket_warmup_tokens':64,'bracket_output_tokens':512}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        restored=probe.post('/slots/0?action=restore',{'filename':snapshot['filename']})
        if restored.get('n_restored')!=103194:
            raise RuntimeError('Unexpected restored prefix')
        (out/'restore.json').write_text(json.dumps(restored,indent=2))
        first=probe.run(f'populate-{target}',prompt,True,512)
        timing=first['final']['timings']
        if timing['cache_n']!=103194 or timing['prompt_n']!=target-103194 or first['final'].get('truncated'):
            raise RuntimeError('Input context was not populated as expected')
        if any(s['is_processing'] for s in get('/slots')):
            raise RuntimeError('Server busy; no snapshot submitted')
        started=time.perf_counter()
        saved=probe.post('/slots/0?action=save',{'filename':saved_filename})
        if saved.get('n_saved')!=target+511:
            raise RuntimeError('Unexpected saved populated prefix')
        (out/'saved-prefix.json').write_text(json.dumps({'filename':saved_filename,
            'completed_tokens':prompt+first['generated_token_ids'],'response':saved,
            'save_wall_s':time.perf_counter()-started,'server':fingerprint},indent=2))
        suffix=probe.tokenize('\n<|im_end|>\n<|im_start|>user\nExplain the edge cases in your implementation.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',True)
        follow_prompt=prompt+first['generated_token_ids']+suffix
        follow=probe.run(f'cached-{label}-followup',follow_prompt,True,256)
        if follow['final']['timings']['cache_n']<target or follow['final'].get('truncated'):
            raise RuntimeError('Followup did not preserve the populated prefix')
        summary={'first':{'timings':timing,'ttft_s':first['time_to_first_generated_token_seconds'],
                          'wall_s':first['wall_seconds']},
                 'followup':{'input_tokens':len(follow_prompt),'timings':follow['final']['timings'],
                             'ttft_s':follow['time_to_first_generated_token_seconds'],'wall_s':follow['wall_seconds']}}
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        emit(f'populated_{label}_complete',**summary)
    records=[]
    for case,mode in enumerate(manifest['bracket_order'],1):
        directory=out/f'case-{case:02d}-{mode}'
        emit(f'{label}_control_loading',case=case,mode=mode)
        with server(directory,backend=backend,launcher=TOOLS/'Start-KVSplitServer.ps1',
                    arguments=['-KvSplit',mode],expected=expected):
            probe.OUT=directory
            for name,count in [('warmup',64),('measured',512)]:
                if any(s['is_processing'] for s in get('/slots')):
                    raise RuntimeError('Server busy; no restore submitted')
                restored=probe.post('/slots/0?action=restore',{'filename':saved_filename})
                (directory/(name+'-restore.json')).write_text(json.dumps(restored,indent=2))
                if restored.get('n_restored')!=target+511:
                    raise RuntimeError('Unexpected restored populated prefix')
                result=probe.run(name,follow_prompt,True,count)
                t=result['final']['timings']
                if t['cache_n']!=target+511 or t['prompt_n']!=len(follow_prompt)-(target+511):
                    raise RuntimeError('Unexpected populated-context workload')
                if name=='warmup':
                    continue
                records.append({'case':case,'mode':mode,'input_tokens':len(follow_prompt),
                    'timings':t,'ttft_s':result['time_to_first_generated_token_seconds'],
                    'wall_s':result['wall_seconds'],
                    'request_sha256':hashlib.sha256((directory/(name+'-request.json')).read_bytes()).hexdigest(),
                    'output_sha256':hashlib.sha256(json.dumps(result['generated_token_ids']).encode()).hexdigest()})
                (out/'bracket-samples.json').write_text(json.dumps(records,indent=2))
                emit(f'{label}_sample',**records[-1])
    summary['bracket']=records
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
