"""Build the isolated gfx1152 eight-block attention workgroup comparison."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from build_fine_kv import BUILD, WORK


DEST = WORK / 'hip-workgroup'
RUNTIME = DEST / 'bin'
SOURCE = WORK / 'llama.cpp/ggml/src/ggml-cuda'
CONTROL_HASH = '51230c04fe096226e2a8cd5fc3a37cfc4930c13fc23e086c4c2771bf7343b4b5'

DISPATCH = r'''                if (aligned && !small && !large) {
                    const char * mode = std::getenv("GGML_HIP_FA_WORKGROUP");
                    if (mode && (std::strcmp(mode, "128") == 0 || std::strcmp(mode, "256") == 0)) {
                        const char * split = std::getenv("GGML_HIP_FA_KV_SPLIT");
                        GGML_ASSERT(split && std::strcmp(split, "double") == 0);
                        static std::atomic<bool> logged_workgroup{false};
                        if (!logged_workgroup.exchange(true)) {
                            fprintf(stderr, "HIP_FA_WORKGROUP threads=%s columns=32 q=3 kv=%" PRId64 "\n", mode, K->ne[1]);
                        }
                        if (std::strcmp(mode, "128") == 0) {
                            ggml_cuda_fattn_workgroup_128(ctx, dst);
                        } else {
                            ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 4, 8>(ctx, dst);
                        }
                        break;
                    }
                }
'''


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'Unexpected source while locating {old!r}')
    return source.replace(old, new)


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_sources():
    # The old entry point and HIP kernels stay in their original objects. Only the new
    # object compiles the private config, including both host and device accessors.
    original = (SOURCE/'fattn-mma-f16.cuh').read_text()
    header = original[:original.index('#define DECL_FATTN_MMA_F16_CASE(')]
    config = 'GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 256, 2,  64, 128, 128,  64, 1, true);'
    header = replace_once(header, config, config.replace('32, 256, 2,', '32, 128, 2,'))
    (DEST/'workgroup-128.cuh').write_text(header)
    (DEST/'workgroup-128.cu').write_text('''#define flash_attn_ext_f16 flash_attn_ext_f16_workgroup_128
#define ggml_cuda_flash_attn_ext_mma_f16_case ggml_cuda_flash_attn_ext_mma_f16_case_128
#include "workgroup-128.cuh"
void ggml_cuda_fattn_workgroup_128(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_flash_attn_ext_mma_f16_case_128<256, 256, 4, 8>(ctx, dst);
}
''')
    dispatcher = (SOURCE/'fattn.cu').read_text()
    dispatcher = replace_once(dispatcher, '#include <string>', '#include <string>\n'
                              'void ggml_cuda_fattn_workgroup_128(ggml_backend_cuda_context &, ggml_tensor *);')
    anchor = '                if (aligned && small) {'
    dispatcher = replace_once(dispatcher, anchor, DISPATCH + anchor)
    # Occupancy may differ between kernels. The controlled variable is threads per
    # workgroup, so require eight split blocks for both experimental selections.
    dispatcher = replace_once(dispatcher, '        blocks = 2*normal_blocks;',
                              '        const char * wg = std::getenv("GGML_HIP_FA_WORKGROUP");\n'
                              '        blocks = wg && (std::strcmp(wg, "128") == 0 || '
                              'std::strcmp(wg, "256") == 0) ? 8 : 2*normal_blocks;')
    (DEST/'fattn.cu').write_text(dispatcher)


def link_runtime(destination, kernel_object, compile_kernel=False):
    runtime = destination/'bin'
    reference = WORK/'hip-kv-split-bin'
    for dll in (BUILD/'bin/ggml-hip.dll', reference/'ggml-hip.dll'):
        if digest(dll) != CONTROL_HASH:
            raise RuntimeError(f'Expected eight-block baseline at {dll}')
    runtime.mkdir(parents=True, exist_ok=True)
    sdk = WORK/'rocm10-build-tools/Lib/site-packages'
    compiler = sdk/'_rocm_sdk_devel/lib/llvm/bin/clang++.exe'
    vs = Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
    raw = subprocess.check_output(f'cmd.exe /d /c "call "{vs}" >nul && set"', text=True)
    env = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line and not line.startswith('='))
    path = next(v for k, v in env.items() if k.upper() == 'PATH')
    env = {k: v for k, v in env.items() if k.upper() != 'PATH'}
    env['PATH'] = os.pathsep.join(map(str, [sdk/'_rocm_sdk_core/bin', sdk/'_rocm_sdk_libraries/bin', path]))
    commands = json.loads((BUILD/'compile_commands.json').read_text())
    entry, = (c for c in commands if Path(c['file']).name == 'fattn.cu')
    prefix = entry['command'].rsplit(' -o ', 1)[0]
    ninja = (BUILD/'build.ninja').read_text()
    target = re.search(r'^build bin/ggml-hip.dll: \S+ (.+?) \|.*?\n((?:  .*\n)+)', ninja, re.M)
    objects = target[1].split()
    old_obj, = (obj for obj in objects if obj.endswith('/fattn.cu.obj'))
    objects[objects.index(old_obj):objects.index(old_obj)+1] = [
        (destination/'fattn.obj').as_posix(), kernel_object.as_posix()]
    fields = dict(line.strip().split(' = ', 1) for line in target[2].splitlines() if ' = ' in line)
    rsp = destination/'link.rsp'
    rsp.write_text(subprocess.list2cmdline(objects) + ' ' + fields['LINK_LIBRARIES'])
    units = [destination/'fattn.cu']
    if compile_kernel:
        units.insert(0, kernel_object.with_suffix('.cu'))
    compile_cmds = [prefix + ' ' + subprocess.list2cmdline(['-I', SOURCE, '-o', unit.with_suffix('.obj'),
                                                            '-c', unit]) for unit in units]
    link = [compiler, '-nostartfiles', '-nostdlib', *fields['LANGUAGE_COMPILE_FLAGS'].split(),
            *fields['LINK_FLAGS'].split(), '-o', runtime/'ggml-hip.dll', '-Xlinker', '/MANIFEST:EMBED',
            '-Xlinker', '/implib:'+str(destination/'hip.lib'), '-Xlinker', '/version:0.27', '@'+str(rsp)]
    with (destination/'build.log').open('w') as log:
        for command in compile_cmds:
            subprocess.run(command, cwd=BUILD, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(link, cwd=BUILD, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for item in reference.iterdir():
        if item.is_file() and item.name != 'ggml-hip.dll' and not (runtime/item.name).exists():
            shutil.copy2(item, runtime/item.name)
    manifest = {'control_sha256': CONTROL_HASH, 'private_sha256': digest(runtime/'ggml-hip.dll'),
                'extra_object_sha256': digest(kernel_object),
                'compile': compile_cmds, 'link': list(map(str, link)),
                'source_sha256': {p.name: digest(p) for p in [SOURCE/'fattn.cu', SOURCE/'fattn-mma-f16.cuh',
                                                               SOURCE/'fattn-common.cuh']}}
    (destination/'build.json').write_text(json.dumps(manifest, indent=2))
    print(runtime, manifest['private_sha256'])


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    write_sources()
    link_runtime(DEST, DEST/'workgroup-128.obj', compile_kernel=True)


if __name__ == '__main__':
    main()
