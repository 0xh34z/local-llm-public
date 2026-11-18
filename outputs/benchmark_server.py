"""Owned server lifecycle for sequential, local inference experiments."""
from contextlib import contextmanager
import json
from pathlib import Path
import re
import subprocess
import time

from run_thread_sweep import processes, get, psquote, powershell

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
BASE_BACKEND = ROOT / 'work/backends/llama-b11888-bin-win-rocm-10.0-x64'


@contextmanager
def server(directory, arguments=(), backend=BASE_BACKEND, launcher=None, expected=None):
    """Refuse occupied port; stop only this experiment's identified, idle server."""
    directory = Path(directory).resolve()
    backend = Path(backend).resolve()
    launcher = Path(launcher or TOOLS/'Start-ROCmServer.ps1').resolve()
    if processes():
        raise RuntimeError('Port 9931 already has a server; no changes made')
    directory.mkdir(exist_ok=False, parents=True)
    args = ['-NoProfile','-ExecutionPolicy','Bypass','-File',str(launcher),
            '-BackendDirectory',str(backend),'-LogPath',str(directory/'server.log'),*map(str,arguments)]
    quoted = ['"'+a+'"' if ' ' in a else a for a in args]
    command = ('Start-Process -FilePath powershell.exe -ArgumentList @('+
        ','.join(psquote(a) for a in quoted)+') -WindowStyle Hidden -PassThru '
        '-RedirectStandardOutput '+psquote(directory/'launcher-stdout.log')+
        ' -RedirectStandardError '+psquote(directory/'launcher-stderr.log')+
        ' | Select-Object -ExpandProperty Id')
    own_pid = None
    try:
        with (directory/'launch-control.txt').open('w') as stdout, (directory/'launch-control-error.log').open('w') as stderr:
            subprocess.run(['powershell.exe','-NoProfile','-Command',"$ErrorActionPreference='Stop'; "+command],
                           stdout=stdout,stderr=stderr,timeout=30,check=True)
        deadline = time.monotonic()+150
        while True:
            live = processes()
            owned = [p for p in live if p['ExecutablePath'] and Path(p['ExecutablePath']).resolve()==backend/'llama-server.exe']
            if len(live)>1 or len(owned)>1:
                raise RuntimeError('Ambiguous server processes')
            if owned:
                own_pid=owned[0]['ProcessId']
            try:
                ready=get('/health').get('status')=='ok'
            except (OSError,ValueError):
                ready=False
            if ready:
                if len(owned)!=1 or len(live)!=1:
                    raise RuntimeError('Health endpoint belongs to an unexpected process')
                launch=owned[0]
                break
            if time.monotonic()>deadline:
                raise RuntimeError('Server startup timed out; inspect launcher logs')
            time.sleep(1)
        for flag,value in (expected or {}).items():
            match=re.search(r'(?:^|\s)'+re.escape(flag)+r'\s+(\S+)',launch['CommandLine'])
            if not match or match[1]!=str(value):
                raise RuntimeError('Unexpected launch flag: '+flag)
        props=get('/props')
        (directory/'server-launch.json').write_text(json.dumps(launch,indent=2))
        (directory/'server-props.json').write_text(json.dumps(props,indent=2))
        if any(s['is_processing'] for s in get('/slots')):
            raise RuntimeError('Server is unexpectedly busy')
        yield props
    finally:
        if own_pid is not None and any(p['ProcessId']==own_pid for p in processes()):
            powershell('& '+psquote(TOOLS/'Stop-ROCmServer.ps1')+' -BackendDirectory '+psquote(backend))
            deadline=time.monotonic()+20
            while any(p['ProcessId']==own_pid for p in processes()):
                if time.monotonic()>deadline:
                    raise RuntimeError('Owned server did not exit')
                time.sleep(0.5)
