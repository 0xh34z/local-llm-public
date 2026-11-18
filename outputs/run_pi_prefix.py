"""Measure a local Pi read-tool cycle on an owned Windows eight-block server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
PORT = 19931


def get(route):
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}{route}', timeout=5) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--read', choices=['small', 'full', 'range', 'template', 'automatic'], default='small')
    parser.add_argument('--long-prefix', action='store_true',
                        help='Append about 100K tokens of synthetic project context to the Pi system prompt')
    parser.add_argument('--default-tools', action='store_true', help='Use the installed Pi default tools for the automatic lookup')
    args = parser.parse_args()
    if args.default_tools and args.read != 'automatic':
        parser.error('--default-tools applies to automatic lookups')
    if args.long_prefix and args.read == 'automatic':
        parser.error('The long-prefix test supplies its own system prompt appendix')
    out = args.out.resolve()
    for port in (9931, PORT):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} is occupied')
    out.mkdir(parents=True, exist_ok=False)
    scratch = ROOT / 'work' / out.name
    scratch.mkdir(exist_ok=False)
    profile = scratch / 'profile'
    profile.mkdir()
    fixture = scratch / 'fixture'
    fixture.mkdir()
    source = 'def delay(attempt):\n    return min(37, 3 ** attempt)\n'
    if args.read != 'small':
        source += '\n' + ''.join(f'LIMIT_{i:03} = {i * 13 + 17}\n' for i in range(600))
    if args.read == 'automatic':
        lines = source.splitlines(keepends=True)
        source = ''.join(lines[3:303] + lines[:3] + lines[303:])
        project_config = fixture / '.pi'
        project_config.mkdir()
        shutil.copyfile(ROOT/'.pi/APPEND_SYSTEM.md', project_config/'APPEND_SYSTEM.md')
        (project_config/'extensions').mkdir()
        shutil.copyfile(ROOT/'.pi/extensions/local-read.js', project_config/'extensions/local-read.js')
    (fixture / 'retry.py').write_text(source)
    config = json.loads((TOOLS / 'pi-models-recommended.json').read_text())
    provider = config['providers']['llama_cpp']
    provider['baseUrl'] = f'http://127.0.0.1:{PORT}/v1'
    model = provider['models'][0]
    model['maxTokens'] = 384
    (profile / 'models.json').write_text(json.dumps(config))
    if args.long_prefix:
        (profile / 'settings.json').write_text(json.dumps({'httpIdleTimeoutMs':1800000,
                                                            'retry':{'enabled':False}}))

    backend = ROOT / 'work/hip-kv-split-bin'
    sdk = ROOT / 'work/rocm10-runtime/Lib/site-packages'
    env = os.environ.copy()
    env.update(GGML_HIP_FA_KV_SPLIT='double', GGML_HIP_OP_PROFILE='0',
               GGML_HIP_FA_EXPERIMENT='auto', GGML_HIP_FA_MMA_TILE='auto')
    env['PATH'] = os.pathsep.join(str(p) for p in
        (sdk / '_rocm_sdk_core/bin', sdk / '_rocm_sdk_libraries/bin', env['PATH']))
    server = [backend / 'llama-server.exe', '-m', Path.home() / 'Desktop/Models' / model['id'],
              '--device', 'ROCm0', '-c', '163840', '-np', '1', '-ngl', '99', '-fa', 'on', '--jinja',
              '-ctk', 'f16', '-ctv', 'f16', '-b', '2048', '-ub', '512', '-t', '8', '-tb', '8',
              '-td', '8', '-tbd', '8', '--spec-type', 'draft-mtp', '--spec-draft-n-max', '2',
              '--host', '127.0.0.1', '--port', str(PORT), '-lm', 'auto', '-lv', '3']
    package = Path.home() / '.pi/agent'
    node = shutil.which('node') or str(Path(os.environ['LOCALAPPDATA']) / 'pi-node/current/node.exe')
    prompt = 'Use the read tool on retry.py. From its implementation give the maximum delay and the first attempt number that reaches it. Respond in one sentence.'
    if args.read != 'small':
        scope = ('read the entire file with no offset or limit' if args.read == 'full'
                 else 'read just lines 1 and 2 with offset 1 and limit 2')
        prompt = f'Use the read tool on retry.py exactly once. For this benchmark, {scope}. From delay(attempt), give the maximum delay and the first attempt number that reaches it. Respond in one sentence.'
    pi = [node, package / 'bin/pi-launcher.js', '--offline', '--no-session', '--no-extensions',
          '--no-skills', '--no-prompt-templates', '--no-themes', '--no-context-files',
          '-e', TOOLS / 'pi-prefix-meter.mjs', '--tools', 'read', '--thinking', 'off',
          '--provider', 'llama_cpp', '--model', model['id'], '--mode', 'json', '-p', prompt]
    if args.read == 'template':
        pi.remove('--no-prompt-templates')
        pi[2:2] = ['--prompt-template', ROOT / '.pi/prompts/read-range.md']
        pi[-1] = '/read-range retry.py 1 2 From delay(attempt), give the maximum delay and the first attempt number that reaches it. Respond in one sentence.'
    if args.read == 'automatic':
        pi[2:2] = ['--approve']
        pi.remove('--no-extensions')
        pi[pi.index('--tools') + 1] = 'read,grep'
        if args.default_tools:
            index = pi.index('--tools')
            del pi[index:index + 2]
        pi[-1] = 'In retry.py, what is the maximum delay and which attempt first reaches it?'
        env.update(PI_PREFIX_GUIDANCE=str(project_config/'APPEND_SYSTEM.md'),
                   PI_PREFIX_GUIDANCE_AUDIT=str(out/'guidance.json'))
    context_file = scratch / 'context.txt'
    if args.long_prefix:
        pi[2:2] = ['--append-system-prompt', context_file]
    env.update(PI_CODING_AGENT_DIR=str(profile), PI_PREFIX_EVIDENCE=str(out / 'requests.jsonl'),
               PI_PREFIX_LONG='1' if args.long_prefix else '0')
    (out / 'launch.json').write_text(json.dumps({
        'server': list(map(str, server)), 'pi': list(map(str, pi)), 'split': 'double',
        'pi_version': (package / 'install/current-version').read_text().strip()}, indent=2))
    with (out / 'server.log').open('wb') as log:
        proc = subprocess.Popen(server, cwd=backend, env=env, stdout=log, stderr=subprocess.STDOUT)
        ready = False
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    ready = get('/health').get('status') == 'ok'
                except OSError:
                    pass
                if ready:
                    break
                time.sleep(1)
            if not ready:
                raise RuntimeError('Server failed startup, see server.log')
            (out / 'props.json').write_text(json.dumps(get('/props'), indent=2))
            if args.long_prefix:
                corpus = (TOOLS/'rocm-100k-mtp2-f16-results/synthetic-corpus.py.txt').read_text()
                def post(route, body):
                    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/{route}',
                        json.dumps(body).encode(), {'Content-Type':'application/json'})
                    with urllib.request.urlopen(req, timeout=90) as response:
                        return json.load(response)
                ids = post('tokenize', {'content':corpus[:600000], 'add_special':False})['tokens']
                if len(ids) < 102400:
                    raise RuntimeError('Long prefix corpus too short')
                excerpt = post('detokenize', {'tokens':ids[:102400]})['content']
                context_file.write_text('Synthetic project background for benchmarking:\n' + excerpt, encoding='utf-8')
                (out/'context.json').write_text(json.dumps({'corpus_tokens':102400,
                    'file_sha256':hashlib.sha256(context_file.read_bytes()).hexdigest()}))
            with (out / 'pi-events.jsonl').open('wb') as events, (out / 'pi-stderr.log').open('wb') as err:
                subprocess.run(pi, cwd=fixture, env=env, stdout=events, stderr=err,
                               timeout=2400 if args.long_prefix else 240, check=True)
            rows = [json.loads(s) for s in (out / 'requests.jsonl').read_text().splitlines()]
            allowed_tools = {'grep', 'read', 'bash', 'powershell'} if args.read == 'automatic' else {'read'}
            if len(rows) < 2 or not any(allowed_tools.intersection(r['calls']) for r in rows) or rows[-1]['stop'] != 'stop':
                raise RuntimeError('Incomplete tool cycle')
            if args.long_prefix and rows[0]['usage']['prompt_tokens'] < 100000:
                raise RuntimeError('Pi did not send the requested long context')
            transcript = [json.loads(s) for s in (out/'pi-events.jsonl').read_text(encoding='utf-8').splitlines()]
            if args.read == 'automatic':
                calls = [{'tool':e['toolName'], 'args':e['args']} for e in transcript if e['type'] == 'tool_execution_start']
                answer = next(e['message']['content'] for e in reversed(transcript)
                              if e['type'] == 'message_end' and e['message']['role'] == 'assistant')
                (out/'tool-result.json').write_text(json.dumps({'calls':calls, 'answer':answer}, indent=2))
                if not json.loads((out/'guidance.json').read_text())['loaded']:
                    raise RuntimeError('The provider did not receive the project instructions')
                # Shell grep returns exit 1 on a miss. Pi can refine the search within the same turn.
                results = [e for e in transcript if e['type'] == 'tool_execution_end' and not e.get('isError')]
                excerpts = '\n'.join(block['text'] for e in results for block in e['result']['content']
                                     if block['type'] == 'text')
                if all(f'LIMIT_{i:03} =' in excerpts for i in range(600)):
                    raise RuntimeError('The lookup returned the whole fixture across its calls')
                if 'return min(37, 3 ** attempt)' not in excerpts:
                    raise RuntimeError('The lookup missed the relevant implementation')
                print(json.dumps({'calls':calls, 'requests':rows}, indent=2))
                return
            call, = (e['args'] for e in transcript if e['type'] == 'tool_execution_start' and e['toolName'] == 'read')
            if Path(call['path']).name != 'retry.py':
                raise RuntimeError('Unexpected file')
            if args.read in ('range','template') and (call.get('offset',1),call.get('limit')) != (1,2):
                raise RuntimeError('Tool did not use the requested range')
            if args.read == 'full' and ('limit' in call or call.get('offset',1) != 1):
                raise RuntimeError('Tool did not read the whole file')
            answer = next(e['message']['content'] for e in reversed(transcript)
                          if e['type'] == 'message_end' and e['message']['role'] == 'assistant')
            (out/'tool-result.json').write_text(json.dumps({'read_arguments':call, 'answer':answer}, indent=2))
            print(json.dumps(rows, indent=2))
        finally:
            if proc.poll() is None:
                if ready:
                    while any(s['is_processing'] for s in get('/slots')):
                        time.sleep(1)
                proc.terminate()
                proc.wait(timeout=30)


if __name__ == '__main__':
    main()
