"""Independent dense attention reference for exported synthetic GGML probe tensors."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    shapes = {}
    equal = {}
    scale = None
    for line in (args.dump_dir / 'metadata.txt').read_text().splitlines():
        fields = line.split()
        if fields[0] == 'input_equal':
            equal[fields[1]] = bool(int(fields[2]))
        elif fields[0] == 'scale':
            scale = float(fields[1])
        else:
            shapes[fields[0]] = tuple(map(int, fields[1:]))
    if not equal or not all(equal.values()) or scale is None:
        raise RuntimeError('Inputs differ between backends or metadata is incomplete')
    arrays = {name: np.memmap(args.dump_dir / (name + '.bin'), dtype='<f4', mode='r',
                              shape=shape, order='F') for name, shape in shapes.items()}
    q, k, v, mask = (arrays[name] for name in ('q','k','v','mask'))
    if q.shape != (256,3,16,1) or k.shape != v.shape or k.shape[2:] != (2,1):
        raise RuntimeError('Unexpected tensor shape')
    expected_shape = (v.shape[0], q.shape[2], q.shape[1], q.shape[3])
    if arrays['gpu'].shape != expected_shape or arrays['cpu'].shape != expected_shape:
        raise RuntimeError('Unexpected attention output layout')
    reference = np.empty(expected_shape, dtype=np.float64)
    mask64 = np.asarray(mask[:, :, 0, 0], dtype=np.float64).T
    for kv_head in range(2):
        k64 = np.asarray(k[:, :, kv_head, 0], dtype=np.float64)
        v64 = np.asarray(v[:, :, kv_head, 0], dtype=np.float64)
        for head in range(kv_head * 8, (kv_head + 1) * 8):
            q64 = np.asarray(q[:, :, head, 0], dtype=np.float64)
            logits = (q64.T @ k64) * scale + mask64
            logits -= np.max(logits, axis=1, keepdims=True)
            probabilities = np.exp(logits)
            totals = np.sum(probabilities, axis=1, keepdims=True, dtype=np.float64)
            if not np.all(totals > 0):
                raise RuntimeError('Fully masked query row')
            probabilities /= totals
            reference[:, head, :, 0] = v64 @ probabilities.T
    energy = np.sum(reference * reference, dtype=np.float64)
    if not np.all(np.isfinite(reference)) or not np.isfinite(energy) or energy <= 0:
        raise RuntimeError('Invalid float64 reference result')
    results = {}
    for backend in ('gpu','cpu'):
        values = np.asarray(arrays[backend], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError('Non-finite backend output: ' + backend)
        diff = values - reference
        results[backend] = {'nmse_vs_float64': float(np.sum(diff * diff, dtype=np.float64) / energy),
                            'max_absolute_error': float(np.max(np.abs(diff))),
                            'relative_rms_error': float(np.sqrt(np.sum(diff * diff, dtype=np.float64) / energy))}
    gpu = np.asarray(arrays['gpu'], dtype=np.float64)
    cpu = np.asarray(arrays['cpu'], dtype=np.float64)
    result = {'numpy_version': np.__version__, 'inputs_equal': equal, 'shapes': shapes, 'scale': scale,
              'reference_rms': float(np.sqrt(energy / reference.size)), 'backends': results,
              'gpu_cpu_nmse_gpu_denominator': float(np.sum((gpu-cpu)**2) / np.sum(gpu**2)),
              'scope': 'One synthetic regular-logit Q3 case, F16 K/V storage, F32 Q storage, F64 dot/softmax/value reduction; no sinks, ALiBi or softcap'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
