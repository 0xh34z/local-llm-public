"""Bracketed 103K-prefix throughput checks for code and open-thinking prompts."""
import json
from pathlib import Path
import statistics

from benchmark_server import server, ROOT, TOOLS
from run_thread_sweep import get, emit
import run_mtp_benchmark as probe


def main():
    out=TOOLS/'kv-split-workload-validation'
    out.mkdir(exist_ok=False)
    snapshot=json.loads((TOOLS/'slot-cache-reuse-results/snapshot.json').read_text())
    tasks={
        'code':('Implement a bounded thread-safe LRU cache in Python with get and put operations. '
                'Specify capacity validation and missing-key behavior, and include tests for replacement, '
                'recency updates, empty inputs and concurrent access. Explain the locking invariant.',False),
        'reasoning':('A shared balance starts at 100. Two threads each read it, subtract 10 locally, and write '
                     'the local result back, with each read and write atomic but the pair unprotected. '
                     'Enumerate all distinct final balances and justify the interleavings. Generalize the '
                     'possible balances to N threads, then explain the minimal correct locking change.',True)}
    manifest={'order':['auto','double','auto'],'context':163840,'cached_prefix':103194,
              'tasks':tasks,'output_tokens_per_task':512,'temperature':0.6,
              'precision':'F16 K/V, unchanged Q4 weights','mtp':2,'batch':2048,'microbatch':512,'threads':8,
              'scope':'Synthetic throughput across two additional prompts; forced output, not quality or tool-call evaluation'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records=[]
    prompts=None
    for case,mode in enumerate(manifest['order'],1):
        directory=out/f'case-{case:02d}-{mode}'
        expected={'-c':163840,'-b':2048,'-ub':512,'-ctk':'f16','-ctv':'f16','-t':8,'-tb':8,
                  '-td':8,'-tbd':8,'--spec-draft-n-max':2}
        emit('workload_loading',case=case,mode=mode)
        with server(directory,backend=ROOT/'work/hip-kv-split-bin',launcher=TOOLS/'Start-KVSplitServer.ps1',
                    arguments=['-KvSplit',mode],expected=expected) as props:
            fingerprint={k:props.get(k) for k in ('model_path','build_info')}|{'n_ctx':props['default_generation_settings']['n_ctx']}
            if fingerprint!=snapshot['server']:
                raise RuntimeError('Snapshot fingerprint mismatch')
            if prompts is None:
                prompts={}
                for name,(question,thinking) in tasks.items():
                    suffix='\n<|im_end|>\n<|im_start|>user\n'+question+'\n<|im_end|>\n<|im_start|>assistant\n<think>\n'
                    if not thinking:
                        suffix+='\n</think>\n\n'
                    prompts[name]=snapshot['completed_tokens']+probe.tokenize(suffix,True)
            probe.OUT=directory
            for name,prompt in [('warmup',prompts['code']),*prompts.items()]:
                if any(s['is_processing'] for s in get('/slots')):
                    raise RuntimeError('Server busy; no request submitted')
                restored=probe.post('/slots/0?action=restore',{'filename':snapshot['filename']})
                if restored.get('n_restored')!=103194:
                    raise RuntimeError('Unexpected restored prefix')
                (directory/(name+'-restore.json')).write_text(json.dumps(restored,indent=2))
                result=probe.run(name,prompt,True,64 if name=='warmup' else 512)
                timing=result['final']['timings']
                if timing['cache_n']!=103194 or timing['prompt_n']!=len(prompt)-103194 or result['final'].get('truncated'):
                    raise RuntimeError('Unexpected prefix reuse or truncation')
                if name=='warmup':
                    continue
                records.append({'case':case,'mode':mode,'task':name,'input_tokens':len(prompt),
                    'tps':timing['predicted_per_second'],'ttft_s':result['time_to_first_generated_token_seconds'],
                    'wall_s':result['wall_seconds'],'draft_n':timing.get('draft_n'),
                    'draft_n_accepted':timing.get('draft_n_accepted')})
                (out/'samples.json').write_text(json.dumps(records,indent=2))
                emit('workload_sample',**records[-1])
    summary={}
    for name in tasks:
        control=[r['tps'] for r in records if r['mode']=='auto' and r['task']==name]
        candidate=[r['tps'] for r in records if r['mode']=='double' and r['task']==name][0]
        summary[name]={'control_mean_tps':statistics.mean(control),'control_samples':control,
                       'candidate_tps':candidate,'percent_gain':100*(candidate/statistics.mean(control)-1)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    emit('workload_validation_complete',summary=summary)


if __name__=='__main__':
    main()
