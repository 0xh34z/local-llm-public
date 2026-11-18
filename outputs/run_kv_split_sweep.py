"""Bracketed KV work-splitting experiment at a restored 103K prefix."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=TOOLS / 'kv-split-100k-results-verified')
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('Existing results; choose a fresh output directory')
    utility = load('sweep_utils', TOOLS / 'run_thread_sweep.py')
    probe = load('throughput_probe', TOOLS / 'run_mtp_benchmark.py')
    if utility.processes():
        raise RuntimeError('Server already running; no changes made')
    oracles = {}
    for columns in ('double','quad'):
        oracle = json.loads((TOOLS / 'kv-split-kernel-tests' / (columns+'-float64.json')).read_text())
        error = oracle['backends']['gpu']['nmse_vs_float64']
        if not math.isfinite(error) or error > 0.0005 or not all(oracle['inputs_equal'].values()):
            raise RuntimeError('MMA numerical check did not qualify: '+columns)
        oracles[columns] = oracle
    snapshot = json.loads((TOOLS / 'slot-cache-reuse-results' / 'snapshot.json').read_text())
    backend = ROOT / 'work' / 'hip-kv-split-bin'
    manifest = {'order': ['auto','double','auto','quad','auto'], 'primary_temperature': 0.6,
                'primary_repetitions': 3, 'primary_output_tokens': 256,
                'greedy_repetitions': 1, 'warmup_output_tokens': 64,
                'model_build': snapshot['server'], 'context': 163840,
                'batch': 2048, 'microbatch': 512, 'threads': 8, 'mtp': 2, 'kv': 'f16',
                'override': 'gfx1152, Q3 D256 H16, KV heads2 F16, >=100K padded KV; other shapes unchanged',
                'normal_execution': 'Profiling disabled; graph replay remains enabled',
                'oracles': oracles, 'backend_sha256': hashlib.sha256((backend/'ggml-hip.dll').read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    records = []
    prompt = None
    own_pid = None
    try:
        for case, mode in enumerate(manifest['order'], 1):
            case_dir = out / f'case-{case:02d}-{mode}'
            case_dir.mkdir()
            probe.OUT = case_dir
            launch_args = ['-NoProfile','-ExecutionPolicy','Bypass','-File',
                           '"'+str(TOOLS/'Start-AttentionExperiment.ps1')+'"',
                           '-AttentionKernel','auto','-MmaColumns','auto','-KvSplit',mode,'-BackendDirectory','"'+str(backend)+'"','-LogPath','"'+str(case_dir/'server.log')+'"']
            command = ('Start-Process -FilePath powershell.exe -ArgumentList @('+
                       ','.join(utility.psquote(a) for a in launch_args)+') -WindowStyle Hidden '
                       '-PassThru -RedirectStandardOutput '+utility.psquote(case_dir/'launcher-stdout.log')+
                       ' -RedirectStandardError '+utility.psquote(case_dir/'launcher-stderr.log')+
                       ' | Select-Object -ExpandProperty Id')
            with (case_dir/'launch-control.txt').open('w') as stdout, (case_dir/'launch-control-stderr.log').open('w') as stderr:
                subprocess.run(['powershell.exe','-NoProfile','-Command',"$ErrorActionPreference='Stop'; "+command],
                               stdout=stdout, stderr=stderr, check=True, timeout=30)
            utility.emit('loading', case=case, mode=mode)
            deadline = time.monotonic()+120
            while True:
                try:
                    if utility.get('/health').get('status') == 'ok':
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('Experiment server did not load')
                time.sleep(1)
            live = utility.processes()
            if len(live) != 1 or Path(live[0]['ExecutablePath']).resolve() != (backend/'llama-server.exe').resolve():
                raise RuntimeError('Unexpected server executable')
            own_pid = live[0]['ProcessId']
            for flag, expected in {'-c':'163840','-ctk':'f16','-ctv':'f16','-b':'2048','-ub':'512',
                                   '-t':'8','-tb':'8','-td':'8','-tbd':'8','--spec-type':'draft-mtp',
                                   '--spec-draft-n-max':'2','--device':'ROCm0'}.items():
                match = re.search(r'(?:^|\s)'+re.escape(flag)+r'\s+(\S+)',live[0]['CommandLine'])
                if not match or match[1] != expected:
                    raise RuntimeError('Unexpected launch flag: '+flag)
            (case_dir/'server-launch.json').write_text(json.dumps(live[0],indent=2),encoding='utf-8')
            props = utility.get('/props')
            fingerprint = {key:props.get(key) for key in ('model_path','build_info')} | {
                'n_ctx':props['default_generation_settings']['n_ctx']}
            if fingerprint != snapshot['server']:
                raise RuntimeError('Snapshot/server fingerprint mismatch')
            (case_dir/'server-props.json').write_text(json.dumps(props,indent=2),encoding='utf-8')
            if prompt is None:
                suffix = probe.tokenize('\n<|im_end|>\n<|im_start|>user\nExplain how the implementation handles empty inputs.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', True)
                prompt = snapshot['completed_tokens']+suffix
            trials = [('warmup',64,0.6)]+[(f'repeat-{r}',256,0.6) for r in (1,2,3)]+[('greedy',256,0.0)]
            for name, count, temperature in trials:
                if any(slot['is_processing'] for slot in utility.get('/slots')):
                    raise RuntimeError('Server busy; no request submitted')
                restored = probe.post('/slots/0?action=restore',{'filename':snapshot['filename']})
                if restored.get('n_restored') != 103194:
                    raise RuntimeError('Unexpected restored prefix')
                (case_dir/(name+'-restore.json')).write_text(json.dumps(restored,indent=2),encoding='utf-8')
                result = probe.run(name,prompt,True,count,temperature=temperature)
                timing = result['final']['timings']
                if timing['cache_n'] != 103194 or timing['prompt_n'] != 26:
                    raise RuntimeError('Prefix reuse differs')
                if name != 'warmup':
                    row = {'case':case,'mode':mode,'trial':name,'temperature':temperature,
                           'tps':timing['predicted_per_second'],'ttft_s':result['time_to_first_generated_token_seconds'],
                           'wall_s':result['wall_seconds'],'draft_n':timing.get('draft_n',0),
                           'accepted_drafts':timing.get('draft_n_accepted',0),
                           'request_sha256':hashlib.sha256((case_dir/(name+'-request.json')).read_bytes()).hexdigest(),
                           'output_sha256':hashlib.sha256(json.dumps(result['generated_token_ids']).encode()).hexdigest()}
                    records.append(row)
                    (out/'samples.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
                    utility.emit('sample',**row)
                    control = [r['tps'] for r in records if r['case']==1 and r['temperature']==0.6]
                    if mode != 'auto' and name == 'repeat-1' and row['tps'] < statistics.mean(control)*0.5:
                        (out/'early-stop.json').write_text(json.dumps({
                            'Reason':'Severe MMA candidate regression; remaining candidate repetitions skipped',
                            'CompletedCandidateTPS':row['tps'],'CompleteCandidateOutputTokens':count,
                            'UnfinishedRequestsExcluded':True},indent=2),encoding='utf-8')
                        utility.emit('candidate_rejected',mode=mode,tps=row['tps'])
                        break
            with (case_dir/'server.log').open('rb') as f:
                encoding = 'utf-16' if f.read(2) in (b'\xff\xfe',b'\xfe\xff') else 'utf-8'
            log = (case_dir/'server.log').read_text(encoding=encoding,errors='replace')
            if 'HIP_FA_MMA_TILE columns=32 q=3 ' not in log:
                raise RuntimeError('Automatic 32-column MMA configuration was not observed')
            if 'HIP_FA_KV_SPLIT mode='+mode+' ' not in log:
                raise RuntimeError('Expected KV split mode was not observed')
            if 'HIP_FA_EXPERIMENT mode=auto kernel=400' not in log:
                raise RuntimeError('MMA family was not selected')
            if 'HIP_OP_PROFILE ' in log:
                raise RuntimeError('Timing profiler was unexpectedly enabled')
            utility.powershell('& '+utility.psquote(TOOLS/'Stop-ROCmServer.ps1')+' -BackendDirectory '+utility.psquote(backend))
            deadline = time.monotonic()+20
            while utility.processes():
                if time.monotonic()>deadline:
                    raise RuntimeError('Server did not exit')
                time.sleep(0.5)
            own_pid = None
            utility.emit('case_complete',case=case,mode=mode,
                         mean_tps=statistics.mean(r['tps'] for r in records if r['case']==case and r['temperature']==0.6))
        utility.emit('attention_sweep_complete',measured_runs=len(records))
    finally:
        if own_pid is not None and any(p['ProcessId']==own_pid for p in utility.processes()):
            utility.powershell('& '+utility.psquote(TOOLS/'Stop-ROCmServer.ps1')+' -BackendDirectory '+utility.psquote(backend))


if __name__ == '__main__':
    main()
