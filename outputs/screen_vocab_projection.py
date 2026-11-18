"""Exact, seeded Q8_0 projection check and wall-clock screen on the private HIP build."""
import argparse
import array
import ctypes as c
import json
import os
from pathlib import Path
import random
import socket
import statistics
import struct
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parents[1] / 'work'
BACKEND = WORK / 'hip-vocab-probe/bin'
WIDTH, ROWS, PERIOD = 2048, 248320, 256


class Init(c.Structure):
    _fields_ = [('size', c.c_size_t), ('buffer', c.c_void_p), ('no_alloc', c.c_bool)]


def sample(mode):
    os.environ['GGML_HIP_VOCAB_ROWS'] = mode
    sdk = WORK / 'rocm10-runtime/Lib/site-packages'
    paths = [BACKEND, sdk/'_rocm_sdk_core/bin', sdk/'_rocm_sdk_libraries/bin']
    os.environ['PATH'] = os.pathsep.join(map(str, paths)) + os.pathsep + os.environ['PATH']
    dirs = [os.add_dll_directory(str(p)) for p in paths]
    libs = [c.CDLL(str(BACKEND / name)) for name in ('ggml-base.dll', 'ggml.dll')]
    ptr, size, integer = c.c_void_p, c.c_size_t, c.c_int

    def api(name, result, *types):
        fn = next(getattr(lib, name) for lib in libs if hasattr(lib, name))
        fn.restype, fn.argtypes = result, types
        return fn

    plugin = api('ggml_backend_load', ptr, c.c_char_p)(os.fsencode(BACKEND / 'ggml-hip.dll'))
    if not plugin:
        raise RuntimeError('Could not load private HIP DLL')
    device = api('ggml_backend_init_by_name', ptr, c.c_char_p, c.c_char_p)(b'ROCm0', None)
    if not device:
        raise RuntimeError('Could not open ROCm0')
    ctx = api('ggml_init', ptr, Init)(Init(1024*1024, None, True))
    tensor = api('ggml_new_tensor_2d', ptr, ptr, integer, c.c_int64, c.c_int64)
    weights, x = tensor(ctx, 8, WIDTH, ROWS), tensor(ctx, 0, WIDTH, 1)
    y = api('ggml_mul_mat', ptr, ptr, ptr, ptr)(ctx, weights, x)
    graph = api('ggml_new_graph_custom', ptr, ptr, size, c.c_bool)(ctx, 32, False)
    api('ggml_build_forward_expand', None, ptr, ptr)(graph, y)
    buf = api('ggml_backend_alloc_ctx_tensors', ptr, ptr, ptr)(ctx, device)
    if not buf:
        raise MemoryError('GPU allocation failed')
    upload = api('ggml_backend_tensor_set', None, ptr, c.c_char_p, size, size)
    rng = random.Random(742)
    inputs = [127 if i % 32 == 0 else rng.randint(-127, 127) for i in range(WIDTH)]
    xdata = array.array('f', inputs).tobytes()
    upload(x, xdata, 0, len(xdata))
    chunk = bytearray()
    reference = []
    for row in range(PERIOD):
        value = 0
        for start in range(0, WIDTH, 32):
            scale = rng.choice([0.25, 0.5, 1.0])
            quants = [rng.randint(-127, 127) for _ in range(32)]
            chunk.extend(struct.pack('<e32b', scale, *quants))
            value += scale * sum(a*b for a,b in zip(quants, inputs[start:start+32]))
        reference.append(value)
    chunk = bytes(chunk)
    assert ROWS % PERIOD == 0
    for i in range(ROWS // PERIOD):
        upload(weights, chunk, i * len(chunk), len(chunk))
    compute = api('ggml_backend_graph_compute', integer, ptr, ptr)
    samples = []
    for block in range(8):
        start = time.perf_counter()
        for _ in range(24):
            if compute(device, graph) != 0:
                raise RuntimeError('graph compute failed')
        elapsed = 1000*(time.perf_counter() - start)/24
        if block:
            samples.append(elapsed)
    actual = (c.c_float * ROWS)()
    api('ggml_backend_tensor_get', None, ptr, ptr, size, size)(y, actual, 0, c.sizeof(actual))
    errors = sum(value != reference[i % PERIOD] for i,value in enumerate(actual))
    print(json.dumps({'mode': mode, 'ms': samples, 'median_ms': statistics.median(samples),
                      'checked_logits': ROWS, 'incorrect': errors}), flush=True)
    api('ggml_backend_buffer_free', None, ptr)(buf)
    api('ggml_free', None, ptr)(ctx)
    api('ggml_backend_free', None, ptr)(device)
    if errors:
        raise RuntimeError('Exact synthetic projection mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--child', choices=['auto','4','8'])
    args = parser.parse_args()
    if args.child:
        sample(args.child)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for idx, mode in enumerate(('auto','4','8','auto','8','4','auto')):
        for port in (9931, 19931):
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError(f'Server port {port} is occupied')
        stem = args.out / f'{idx+1}-{mode}'
        with stem.with_suffix('.log').open('w') as err:
            res = subprocess.run([sys.executable, __file__, args.out, '--child', mode],
                                 cwd=BACKEND, stderr=err, capture_output=False,
                                 stdout=subprocess.PIPE, text=True)
        if res.returncode or 'HIP_VOCAB_ROWS mode='+mode not in stem.with_suffix('.log').read_text():
            raise RuntimeError(f'Failed dispatch or oracle, see {stem}.log and {res.stdout}')
        row = json.loads(res.stdout.strip().splitlines()[-1])
        rows.append(row)
        (args.out/'samples.json').write_text(json.dumps(rows, indent=2))
        print(row, flush=True)


if __name__ == '__main__':
    main()
