"""Build isolated gfx1152 HIP DLLs whose quantized matrix-vector (MMVQ) launch shape is set at compile time.

gfx1152 (RDNA 3.5) uses the RDNA2 MMVQ table, which has no tuned entries: every launch uses one warp and one
row per block. This build adds compile-time overrides on that table only:
  NW1/RPB1  warps / rows per block for 1-token MUL_MAT (dense; also MUL_MAT_ID with one token)
  NW/RPB    warps / rows per block for 2-8 token dense MUL_MAT
  MOE       rows per block in the multi-token MUL_MAT_ID kernel (upstream 2)
Each variant replaces only mmvq.cu's object in the measured eight-block build and relinks ggml-hip.dll.
"""
import argparse
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
SOURCE = WORK / 'llama.cpp/ggml/src/ggml-cuda'
DEST = WORK / 'hip-mmvq'
REFERENCE = WORK / 'hip-kv-split-bin'
CONTROL_HASH = '51230c04fe096226e2a8cd5fc3a37cfc4930c13fc23e086c4c2771bf7343b4b5'
KNOBS = ('NW1', 'RPB1', 'NW', 'RPB', 'MOE')

NWARPS = '''static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_RDNA2) {
        return ncols_dst == 1 ? GGML_HIP_MMVQ_NW1 : GGML_HIP_MMVQ_NW;
    }
'''
ROWS = '''static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_RDNA2) {
        return ncols_dst == 1 ? GGML_HIP_MMVQ_RPB1 : GGML_HIP_MMVQ_RPB;
    }
'''
DEFAULTS = ''.join(f'#ifndef GGML_HIP_MMVQ_{k}\n#define GGML_HIP_MMVQ_{k} {v}\n#endif\n'
                   for k, v in dict(NW1=1, RPB1=1, NW=1, RPB=1, MOE=2).items())


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Unexpected source near {old[:60]!r}')
    return text.replace(old, new)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def patched_source():
    text = (SOURCE/'mmvq.cu').read_text()
    text = replace_once(text, '#include <type_traits>\n', '#include <type_traits>\n\n' + DEFAULTS)
    head = 'static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {\n'
    text = replace_once(text, head, NWARPS)
    head = 'static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {\n'
    text = replace_once(text, head, ROWS)
    text = replace_once(text, '    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning\n',
                        '    constexpr int rows_per_block = GGML_HIP_MMVQ_MOE; // upstream: 2\n')
    return text


ASYNC_DECL = '''    const bool profile_graph = ggml_cuda_op_profile_enabled() && profile_kv >= 100000 && profile_q <= 3;
    // GGML_HIP_OP_PROFILE=2: pooled start/stop events per op group, no host synchronization until the graph ends,
    // so the stream stays full and the event pairs measure GPU execution rather than launch gaps.
    static const bool profile_async = [] { const char * v = std::getenv("GGML_HIP_OP_PROFILE"); return v && std::strcmp(v, "2") == 0; }();
    static std::vector<hipEvent_t> profile_pool;
    std::vector<std::pair<int, int>> profile_groups;
'''
ASYNC_END = '''    const auto profile_end = [&](int first, int last) {
        if (!profile_graph) {
            return;
        }
        if (profile_async) {
            CUDA_CHECK(hipEventRecord(profile_pool[2*profile_groups.size() + 1], cuda_ctx->stream()));
            profile_groups.emplace_back(first, last);
            return;
        }
        CUDA_CHECK(hipEventRecord(profile_stop, cuda_ctx->stream()));
        CUDA_CHECK(hipEventSynchronize(profile_stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(hipEventElapsedTime(&elapsed_ms, profile_start, profile_stop));
        profile_print(first, last, elapsed_ms, ggml_time_us() - profile_wall_start);
    };
'''
ASYNC_START = '''                    if (profile_async) {
                        const size_t k = profile_groups.size();
                        while (profile_pool.size() < 2*k + 2) {
                            hipEvent_t event;
                            CUDA_CHECK(hipEventCreateWithFlags(&event, hipEventDefault));
                            profile_pool.push_back(event);
                        }
                        CUDA_CHECK(hipEventRecord(profile_pool[2*k], cuda_ctx->stream()));
                    } else {
                        CUDA_CHECK(hipEventRecord(profile_start, cuda_ctx->stream()));
                    }
'''
ASYNC_FLUSH = '''    if (profile_graph) {
        if (profile_async && !profile_groups.empty()) {
            CUDA_CHECK(hipEventSynchronize(profile_pool[2*profile_groups.size() - 1]));
            for (size_t k = 0; k < profile_groups.size(); ++k) {
                float elapsed_ms = 0.0f;
                CUDA_CHECK(hipEventElapsedTime(&elapsed_ms, profile_pool[2*k], profile_pool[2*k + 1]));
                profile_print(profile_groups[k].first, profile_groups[k].second, elapsed_ms, 0);
            }
        }
        CUDA_CHECK(hipEventDestroy(profile_start));
'''


def async_profiler_source():
    """ggml-cuda.cu with the private op profiler gaining an unsynchronized mode (GGML_HIP_OP_PROFILE=2)."""
    text = (SOURCE/'ggml-cuda.cu').read_text()
    text = replace_once(text, '    const bool profile_graph = ggml_cuda_op_profile_enabled() && profile_kv >= 100000 && profile_q <= 3;\n',
                        ASYNC_DECL)
    start = text.index('    const auto profile_end = [&](int first, int last) {\n')
    fmt = text.index('        const ggml_tensor * node = cgraph->nodes[first];\n', start)
    close = text.index('    };\n', fmt) + len('    };\n')
    body = text[fmt:close - len('    };\n')]
    printer = ('    const auto profile_print = [&](int first, int last, float elapsed_ms, int64_t wall_us) {\n' + body
               .replace('ggml_time_us() - profile_wall_start', 'wall_us') + '    };\n')
    text = text[:start] + printer + ASYNC_END + text[close:]
    text = replace_once(text, '                    CUDA_CHECK(hipEventRecord(profile_start, cuda_ctx->stream()));\n', ASYNC_START)
    text = replace_once(text, '    if (profile_graph) {\n        CUDA_CHECK(hipEventDestroy(profile_start));\n', ASYNC_FLUSH)
    return text


def toolchain_env():
    sdk = WORK/'rocm10-build-tools/Lib/site-packages'
    vs = Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
    raw = subprocess.check_output(f'cmd.exe /d /c "call "{vs}" >nul && set"', text=True)
    env = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line and not line.startswith('='))
    path = next(v for k, v in env.items() if k.upper() == 'PATH')
    env = {k: v for k, v in env.items() if k.upper() != 'PATH'}
    env['PATH'] = os.pathsep.join(map(str, [sdk/'_rocm_sdk_core/bin', sdk/'_rocm_sdk_libraries/bin', path]))
    return env, sdk/'_rocm_sdk_devel/lib/llvm/bin/clang++.exe'


def build(name, knobs, probe=None, async_profiler=False):
    for dll in (BUILD/'bin/ggml-hip.dll', REFERENCE/'ggml-hip.dll'):
        if digest(dll) != CONTROL_HASH:
            raise RuntimeError(f'Expected the eight-block baseline at {dll}')
    out = DEST/name
    runtime = out/'bin'
    runtime.mkdir(parents=True, exist_ok=True)
    source = DEST/'mmvq.cu'
    source.write_text(patched_source())
    env, compiler = toolchain_env()
    commands = json.loads((BUILD/'compile_commands.json').read_text())
    entry, = (c for c in commands if re.search(r'[\\/]mmvq\.cu$', c['file']))
    prefix = entry['command'].rsplit(' -o ', 1)[0].rsplit(' -c ', 1)[0]
    defines = [f'-DGGML_HIP_MMVQ_{k}={v}' for k, v in knobs.items()]
    obj = out/'mmvq.obj'
    compile_cmds = [prefix + ' ' + subprocess.list2cmdline([*defines, '-I', SOURCE, '-o', obj, '-c', source])]
    replaced = {'/mmvq.cu.obj': obj}
    if async_profiler:
        cuda_src = out/'ggml-cuda.cu'
        cuda_src.write_text(async_profiler_source())
        cuda_entry, = (c for c in commands if re.search(r'[\\/]ggml-cuda\.cu$', c['file']))
        cuda_prefix = cuda_entry['command'].rsplit(' -o ', 1)[0]
        cuda_obj = out/'ggml-cuda.obj'
        compile_cmds.append(cuda_prefix + ' ' + subprocess.list2cmdline(['-I', SOURCE, '-o', cuda_obj, '-c', cuda_src]))
        replaced['/ggml-cuda.cu.obj'] = cuda_obj
    ninja = (BUILD/'build.ninja').read_text()
    target = re.search(r'^build bin/ggml-hip.dll: \S+ (.+?) \|.*?\n((?:  .*\n)+)', ninja, re.M)
    objects = target[1].split()
    for suffix, new in replaced.items():
        old, = (o for o in objects if o.endswith(suffix))
        objects[objects.index(old)] = new.as_posix()
    fields = dict(line.strip().split(' = ', 1) for line in target[2].splitlines() if ' = ' in line)
    rsp = out/'link.rsp'
    rsp.write_text(subprocess.list2cmdline(objects) + ' ' + fields['LINK_LIBRARIES'])
    link = [compiler, '-nostartfiles', '-nostdlib', *fields['LANGUAGE_COMPILE_FLAGS'].split(),
            *fields['LINK_FLAGS'].split(), '-o', runtime/'ggml-hip.dll', '-Xlinker', '/MANIFEST:EMBED',
            '-Xlinker', '/implib:'+str(out/'hip.lib'), '-Xlinker', '/version:0.27', '@'+str(rsp)]
    with (out/'build.log').open('w') as log:
        for compile_cmd in compile_cmds:
            subprocess.run(compile_cmd, cwd=BUILD, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(link, cwd=BUILD, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for item in REFERENCE.iterdir():
        if item.is_file() and item.name != 'ggml-hip.dll' and not (runtime/item.name).exists():
            shutil.copy2(item, runtime/item.name)
    if probe:
        shutil.copy2(probe, runtime/probe.name)
    manifest = dict(name=name, knobs=knobs, control_sha256=CONTROL_HASH, private_sha256=digest(runtime/'ggml-hip.dll'),
                    source_sha256={'mmvq.cu': digest(SOURCE/'mmvq.cu'), 'patched': digest(source)},
                    async_profiler=async_profiler, compile=compile_cmds, link=list(map(str, link)))
    (out/'build.json').write_text(json.dumps(manifest, indent=2))
    print(name, knobs, manifest['private_sha256'], flush=True)
    return runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('variants', nargs='+', help='name:KNOB=v,KNOB=v (empty knob list = upstream values)')
    p.add_argument('--probe', type=Path, help='probe executable to copy into each runtime')
    p.add_argument('--async-profiler', action='store_true', help='also build ggml-cuda.cu with GGML_HIP_OP_PROFILE=2')
    args = p.parse_args()
    for spec in args.variants:
        name, _, rest = spec.partition(':')
        knobs = dict(kv.split('=') for kv in rest.split(',') if kv)
        if set(knobs) - set(KNOBS):
            raise ValueError(f'unknown knob in {spec}')
        build(name, {k: int(v) for k, v in knobs.items()}, args.probe, args.async_profiler)


if __name__ == '__main__':
    main()
