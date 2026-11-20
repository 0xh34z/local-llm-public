"""Compile the guarded-dispatch object's D256 / 4x8 query-in-LDS alternative for the Radeon 860M."""
import json
import os
from pathlib import Path
import subprocess

from build_attention_workgroup import BUILD, WORK, SOURCE, digest, replace_once


OUT = WORK / 'hip-wg-metadata/qlds'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = (SOURCE / 'fattn-mma-f16.cuh').read_text()
    header = original[:original.index('#define DECL_FATTN_MMA_F16_CASE(')]
    config = 'GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 256, 2,  64, 128, 128,  64, 1, true);'
    header = replace_once(header, config, config.replace('true', 'false'))
    (OUT / 'qlds-256.cuh').write_text(header)
    (OUT / 'qlds-256.cu').write_text('''#define flash_attn_ext_f16 flash_attn_ext_f16_qlds_256
#define ggml_cuda_flash_attn_ext_mma_f16_case ggml_cuda_flash_attn_ext_mma_f16_case_qlds_256
#include "qlds-256.cuh"
void ggml_cuda_fattn_qlds_256(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_flash_attn_ext_mma_f16_case_qlds_256<256, 256, 4, 8>(ctx, dst);
}
''')
    commands = json.loads((BUILD / 'compile_commands.json').read_text())
    entry, = [c for c in commands if Path(c['file']).name == 'fattn.cu']
    prefix = entry['command'].rsplit(' -o ', 1)[0]
    cmd = prefix + ' ' + subprocess.list2cmdline(['-I', SOURCE, '-o', OUT/'qlds-256.obj', '-c', OUT/'qlds-256.cu'])
    sdk = WORK/'rocm10-build-tools/Lib/site-packages'
    vs = Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
    vars_text = subprocess.check_output(f'cmd.exe /d /c "call "{vs}" >nul && set"', text=True)
    env = dict(line.split('=', 1) for line in vars_text.splitlines() if '=' in line and not line.startswith('='))
    path = next(v for k, v in env.items() if k.upper() == 'PATH')
    env = {k: v for k, v in env.items() if k.upper() != 'PATH'}
    env['PATH'] = os.pathsep.join(map(str, [sdk/'_rocm_sdk_core/bin', sdk/'_rocm_sdk_libraries/bin', path]))
    with (OUT/'build.log').open('w') as log:
        subprocess.run(cmd, cwd=BUILD, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest = {'compile': cmd, 'source_sha256': digest(SOURCE/'fattn-mma-f16.cuh'),
                'header_sha256': digest(OUT/'qlds-256.cuh'), 'object_sha256': digest(OUT/'qlds-256.obj')}
    (OUT/'build.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
