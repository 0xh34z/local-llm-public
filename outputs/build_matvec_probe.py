"""Compile test-backend-ops.cpp (with the LLAMA_HIP_MATVEC_PROBE case set) against the diagnostic build's ggml libs."""
import subprocess

from build_mmvq_variants import BUILD, DEST, WORK, toolchain_env

SOURCE = WORK/'llama.cpp'


def main():
    env, compiler = toolchain_env()
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST/'test-matvec-probe.exe'
    cmd = [compiler, '-std=c++17', '-O2', '-DNDEBUG', '-DGGML_SHARED', '-DGGML_BACKEND_DL',
           '-I', SOURCE/'ggml/include', SOURCE/'tests/test-backend-ops.cpp',
           BUILD/'ggml/src/ggml.lib', BUILD/'ggml/src/ggml-base.lib', '-o', out]
    with (DEST/'probe-build.log').open('w') as log:
        subprocess.run(list(map(str, cmd)), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(out)


if __name__ == '__main__':
    main()
