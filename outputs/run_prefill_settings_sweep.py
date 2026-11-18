"""Matched 4096-token suffix processing at a restored 103K prefix; not full cold prefill."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from benchmark_server import server, BASE_BACKEND, TOOLS, ROOT
from run_thread_sweep import get, emit
import run_mtp_benchmark as probe


def summarize(out, records):
    groups=[]
    for case in sorted({r['case'] for r in records}):
        rows=[r for r in records if r['case']==case]
        groups.append({'case':case,'batch':rows[0]['batch'],'batch_threads':rows[0]['batch_threads'],
            'microbatch':rows[0].get('microbatch',512),
            'runs':len(rows),'mean_prompt_s':statistics.mean(r['prompt_s'] for r in rows),
            'mean_ttft_s':statistics.mean(r['ttft_s'] for r in rows),
            'min_prompt_s':min(r['prompt_s'] for r in rows),'max_prompt_s':max(r['prompt_s'] for r in rows)})
    expected_samples=2*len(json.loads((out/'manifest.json').read_text())['order'])
    summary={'groups':groups,'records':records,'complete':len(records)==expected_samples}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['| Case | Logical batch | Microbatch | Prompt threads | Samples | Mean prompt seconds | Range seconds |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['case']} | {g['batch']} | {g['microbatch']} | {g['batch_threads']} | {g['runs']} | {g['mean_prompt_s']:.3f} | {g['min_prompt_s']:.3f}-{g['max_prompt_s']:.3f} |")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path)
    parser.add_argument('--microbatch-sweep',action='store_true')
    args=parser.parse_args()
    out=(args.out_dir or TOOLS/('microbatch-small-100k-results' if args.microbatch_sweep else 'prefill-settings-100k-results')).resolve()
    out.mkdir(exist_ok=False,parents=True)
    snapshot=json.loads((TOOLS/'slot-cache-reuse-results/snapshot.json').read_text())
    source=json.loads((TOOLS/'rocm-100k-mtp2-f16-results/cold-102400-mtp2-request.json').read_text())
    order=([(2048,8,512),(2048,8,256),(2048,8,128),(2048,8,512)] if args.microbatch_sweep else
           [(2048,8,512),(512,8,512),(2048,4,512),(4096,8,512),(2048,16,512),(2048,8,512)])
    manifest={'order':order,'context':163840,'microbatch':'see order triples: batch, threads, microbatch','kv':'f16','mtp':2,
              'generation_threads':8,'repeats':2,'measured_suffix_tokens':4096,'warmup_suffix_tokens':1024,
              'output_tokens':1,'snapshot':snapshot['filename'],'server':snapshot['server'],
              'scope':'Native completion from baseline slot; not a fresh cold 100K input or generation benchmark'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    prompt=None
    records=[]
    for case,(batch,batch_threads,microbatch) in enumerate(order,1):
        directory=out/(f'case-{case:02d}-ub{microbatch}' if args.microbatch_sweep else f'case-{case:02d}-b{batch}-tb{batch_threads}')
        expected={'-c':'163840','-ctk':'f16','-ctv':'f16','-b':batch,'-ub':microbatch,'-t':8,'-td':8,
                  '-tb':batch_threads,'-tbd':batch_threads,'--spec-draft-n-max':2,'--device':'ROCm0'}
        emit('prefill_loading',case=case,batch=batch,batch_threads=batch_threads,microbatch=microbatch)
        with server(directory,arguments=['-BatchSize',batch,'-BatchThreads',batch_threads,'-MicroBatchSize',microbatch],expected=expected) as props:
            fingerprint={k:props.get(k) for k in ('model_path','build_info')}|{'n_ctx':props['default_generation_settings']['n_ctx']}
            if fingerprint!=snapshot['server']:
                raise RuntimeError('Snapshot fingerprint differs')
            if prompt is None:
                header=probe.tokenize('\n<|im_end|>\n<|im_start|>user\nReview these additional synthetic modules:\n',True)
                tail=probe.tokenize('\nSummarize the validation requirements.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',True)
                suffix=header+source['prompt'][100:100+4096-len(header)-len(tail)]+tail
                assert len(suffix)==4096
                prompt=snapshot['completed_tokens']+suffix
                warmup=snapshot['completed_tokens']+suffix[:1024]
            probe.OUT=directory
            for name,request in [('warmup',warmup),('repeat-1',prompt),('repeat-2',prompt)]:
                if any(s['is_processing'] for s in get('/slots')):
                    raise RuntimeError('Server is busy; no slot restore submitted')
                restored=probe.post('/slots/0?action=restore',{'filename':snapshot['filename']})
                (directory/(name+'-restore.json')).write_text(json.dumps(restored,indent=2))
                if restored.get('n_restored')!=103194:
                    raise RuntimeError('Unexpected restored prefix')
                result=probe.run(name,request,True,1,temperature=0)
                timing=result['final']['timings']
                if timing['cache_n']!=103194 or timing['prompt_n']!=len(request)-103194 or result['final'].get('truncated'):
                    raise RuntimeError('Unexpected cache, prefill count or truncation')
                if name=='warmup':
                    continue
                records.append({'case':case,'batch':batch,'batch_threads':batch_threads,'microbatch':microbatch,'name':name,
                    'prompt_s':timing['prompt_ms']/1000,'prompt_tps':timing['prompt_per_second'],
                    'ttft_s':result['time_to_first_generated_token_seconds'],'wall_s':result['wall_seconds'],
                    'cache_n':timing['cache_n'],'prompt_n':timing['prompt_n'],'input_tokens':len(request),
                    'request_sha256':hashlib.sha256((directory/(name+'-request.json')).read_bytes()).hexdigest(),
                    'generated_token_ids':result['generated_token_ids']})
                summarize(out,records)
                emit('prefill_sample',**records[-1])
        emit('prefill_case_complete',case=case)
    if len({r['request_sha256'] for r in records})!=1:
        raise RuntimeError('Measured requests differ')
    emit('prefill_settings_complete',requests=len(records))


if __name__=='__main__':
    main()
