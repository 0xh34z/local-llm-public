"""Build isolated 6, 10 and 12 block long-context attention variants on gfx1152."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

TOOLS = Path(__file__).resolve().parent
WORK = TOOLS.parent / 'work'
BUILD = WORK / 'hip-diagnostic-build'
DEST = WORK / 'hip-kv-fine'
BACKEND = DEST / 'bin'

EXTRA_MODES = """    } else if (mode == "6") {
        blocks = 6;
    } else if (mode == "10") {
        blocks = 10;
    } else if (mode == "12") {
        blocks = 12;
"""


def main():
    expected = '51230c04fe096226e2a8cd5fc3a37cfc4930c13fc23e086c4c2771bf7343b4b5'
    for dll in (BUILD/'bin/ggml-hip.dll', WORK/'hip-kv-split-bin/ggml-hip.dll'):
        with dll.open('rb') as artifact:
            if hashlib.file_digest(artifact, 'sha256').hexdigest() != expected:
                raise RuntimeError('This experiment requires the measured eight-block build: ' + str(dll))
    BACKEND.mkdir(parents=True, exist_ok=True)
    source = WORK / 'llama.cpp/ggml/src/ggml-cuda/fattn.cu'
    body = source.read_text()
    anchor = '    } else if (mode != "auto") {\n        GGML_ABORT("invalid GGML_HIP_FA_KV_SPLIT mode");'
    if body.count(anchor) != 1:
        raise RuntimeError('Unexpected attention mode dispatch')
    (DEST / 'fattn.cu').write_text(body.replace(anchor, EXTRA_MODES + anchor))
    commands = json.loads((BUILD / 'compile_commands.json').read_text())
    entry, = (c for c in commands if Path(c['file']).name == 'fattn.cu')
    prefix = entry['command'].rsplit(' -o ', 1)[0]
    sdk = WORK / 'rocm10-build-tools/Lib/site-packages'
    compiler = sdk / '_rocm_sdk_devel/lib/llvm/bin/clang++.exe'
    vs = Path(os.environ['ProgramFiles(x86)']) / 'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
    raw = subprocess.check_output(f'cmd.exe /d /c "call "{vs}" >nul && set"', text=True)
    env = dict(os.environ)
    env.update(line.split('=', 1) for line in raw.splitlines() if '=' in line and not line.startswith('='))
    path = next(v for k,v in env.items() if k.upper() == 'PATH' and 'MSVC' in v)
    env = {k:v for k,v in env.items() if k.upper() != 'PATH'}
    env['PATH'] = os.pathsep.join(map(str, [sdk/'_rocm_sdk_core/bin', sdk/'_rocm_sdk_libraries/bin', path]))
    obj = DEST / 'fattn.obj'
    compile_cmd = prefix + ' ' + subprocess.list2cmdline(['-I', source.parent, '-o', obj, '-c', DEST / 'fattn.cu'])
    ninja = (BUILD / 'build.ninja').read_text()
    target = re.search(r'^build bin/ggml-hip.dll: \S+ (.+?) \|.*?\n((?:  .*\n)+)', ninja, re.M)
    objects = [obj.as_posix() if x.endswith('/fattn.cu.obj') else x for x in target[1].split()]
    assert objects.count(obj.as_posix()) == 1
    fields = dict(line.strip().split(' = ', 1) for line in target[2].splitlines() if ' = ' in line)
    rsp = DEST / 'link.rsp'
    rsp.write_text(subprocess.list2cmdline(objects) + ' ' + fields['LINK_LIBRARIES'])
    link_cmd = [compiler, '-nostartfiles', '-nostdlib', *fields['LANGUAGE_COMPILE_FLAGS'].split(),
                *fields['LINK_FLAGS'].split(), '-o', BACKEND / 'ggml-hip.dll', '-Xlinker', '/MANIFEST:EMBED',
                '-Xlinker', '/implib:' + str(DEST / 'hip.lib'), '-Xlinker', '/version:0.27', '@' + str(rsp)]
    with (DEST / 'build.log').open('w') as log:
        subprocess.run(compile_cmd, cwd=BUILD, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(link_cmd, cwd=BUILD, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for p in (WORK / 'hip-kv-split-bin').iterdir():
        if p.is_file() and p.name != 'ggml-hip.dll' and not (BACKEND / p.name).exists():
            shutil.copy2(p, BACKEND / p.name)
    print(BACKEND)


if __name__ == '__main__':
    main()
