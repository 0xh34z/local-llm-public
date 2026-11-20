"""Bytes each decode step must read, from the GGUF tensor table, and the bandwidth floor they imply.

This is a calculation, not a measurement. It counts each weight byte once per step
(perfect reuse across the tokens of a batch) and ignores activations, logits
transfer, sampling and launch overhead.

The model path defaults to $LLM_MODEL_PATH or ~/Models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf.
The script auto-finds gguf-py under work/llama.cpp*/gguf-py; use --gguf-py if your
checkout is elsewhere. Runtime dependencies are numpy and PyYAML, because gguf-py
imports both even when only the tensor table is read.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path(os.environ.get(
    'LLM_MODEL_PATH',
    Path.home() / 'Models' / 'Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf',
)).expanduser()
DEFAULT_STREAM_GBPS = 81.98  # standalone GPU read median, outputs/stream-bandwidth-gfx1152/REPORT.md
DEFAULT_STREAM_SOURCE = 'outputs/stream-bandwidth-gfx1152/REPORT.md (81.98 GB/s median)'
DEFAULT_DEPTHS = (0, 8192, 32768, 65536, 103194, 131072, 160511)


def find_gguf_py(explicit):
    """Return a directory containing gguf/__init__.py."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    work = ROOT / 'work'
    candidates.extend([
        work / 'llama.cpp' / 'gguf-py',
        work / 'llama.cpp-b11888' / 'gguf-py',
    ])
    if work.is_dir():
        candidates.extend(sorted(work.glob('llama.cpp*/gguf-py')))

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.name == 'gguf' and (candidate / '__init__.py').is_file():
            return candidate.parent
        if (candidate / 'gguf' / '__init__.py').is_file():
            return candidate
    raise SystemExit(
        'Could not find gguf-py. Pass --gguf-py <path> or clone llama.cpp to '
        'work/llama.cpp-b11888 (see BUILD.md).'
    )


def field_value(field):
    """Read a scalar GGUF field across gguf-py versions."""
    if hasattr(field, 'contents'):
        value = field.contents()
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        return value
    return field.parts[field.data[0]].tolist()[0]


def load(model, gguf_py):
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise SystemExit('numpy is required. Install it with: python -m pip install numpy PyYAML')

    # gguf-py imports yaml for metadata helpers that this script never calls.
    sys.modules.setdefault('yaml', types.ModuleType('yaml'))
    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader

    reader = GGUFReader(str(model))
    arch_field = reader.fields.get('general.architecture')
    if arch_field is None:
        raise SystemExit(f'{model} has no general.architecture field')
    arch = str(field_value(arch_field))

    fields = {}
    prefix = arch + '.'
    for name, field in reader.fields.items():
        if not name.startswith(prefix):
            continue
        if len(field.data) == 0:
            continue
        fields[name[len(prefix):]] = field_value(field)

    token_field = reader.fields.get('tokenizer.ggml.tokens')
    if token_field is not None:
        n_vocab = len(token_field.data)
    else:
        n_vocab = int(fields.get('vocab_size', 0))
    if n_vocab <= 0:
        raise SystemExit('Could not determine the vocabulary size from GGUF metadata')

    tensors = {t.name: int(t.n_bytes) for t in reader.tensors}
    return tensors, fields, n_vocab, arch


def as_int(value, name):
    if isinstance(value, (list, tuple)):
        if not value:
            raise SystemExit(f'GGUF field {name} is empty')
        value = value[0]
    return int(value)


def budget(tensors, hp, n_vocab, model, arch, stream_gbps, stream_source, depths):
    n_layer = as_int(hp['block_count'], 'block_count')
    n_nextn = as_int(hp.get('nextn_predict_layers', 0), 'nextn_predict_layers')
    n_expert = as_int(hp.get('expert_count', 1), 'expert_count')
    n_used = as_int(hp.get('expert_used_count', n_expert), 'expert_used_count')
    if n_expert < 1 or n_used < 1 or n_used > n_expert:
        raise SystemExit(f'Invalid expert counts: expert_count={n_expert}, expert_used_count={n_used}')
    if n_nextn > n_layer:
        raise SystemExit(f'nextn_predict_layers={n_nextn} exceeds block_count={n_layer}')

    embd_bytes = tensors.get('token_embd.weight')
    if embd_bytes is None:
        raise SystemExit('GGUF has no token_embd.weight tensor')
    if embd_bytes % n_vocab:
        raise SystemExit(
            f'token_embd.weight ({embd_bytes} bytes) is not divisible by vocab size {n_vocab}; '
            'cannot derive the embedding row size'
        )
    embd_row = embd_bytes // n_vocab

    head = tensors.get('output.weight')
    head_tied = head is None
    if head_tied:
        head = embd_bytes

    main_layers = range(n_layer - n_nextn)
    mtp_layer = n_layer - n_nextn

    def layer(i):
        prefix = f'blk.{i}.'
        own = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
        routed = sum(v for k, v in own.items() if k.endswith('_exps.weight'))
        return {
            'nonrouted': sum(v for k, v in own.items() if not k.endswith('_exps.weight')),
            'expert': routed / n_expert,
            'attention': any(k.startswith('attn_') for k in own),
        }

    layers = [layer(i) for i in main_layers]
    mtp = layer(mtp_layer) if n_nextn else {'nonrouted': 0, 'expert': 0, 'attention': False}
    attn_layers = sum(1 for l in layers if l['attention'])
    deltanet_layers = len(layers) - attn_layers

    head_count = as_int(hp['attention.head_count'], 'attention.head_count')
    head_count_kv = as_int(hp['attention.head_count_kv'], 'attention.head_count_kv')
    key_length = as_int(
        hp.get('attention.key_length', as_int(hp['embedding_length'], 'embedding_length') // head_count),
        'attention.key_length',
    )
    kv_per_token_layer = 2 * head_count_kv * key_length * 2  # K+V, F16

    # Gated DeltaNet recurrent state: F32 [S_k, S_v, heads], read and written once per step (in place).
    recurrent = 0
    ssm_fields = ('ssm.state_size', 'ssm.time_step_rank', 'ssm.conv_kernel', 'ssm.inner_size', 'ssm.group_count')
    if all(name in hp for name in ssm_fields):
        state_size = as_int(hp['ssm.state_size'], 'ssm.state_size')
        heads = as_int(hp['ssm.time_step_rank'], 'ssm.time_step_rank')
        inner_size = as_int(hp['ssm.inner_size'], 'ssm.inner_size')
        group_count = as_int(hp['ssm.group_count'], 'ssm.group_count')
        conv_kernel = as_int(hp['ssm.conv_kernel'], 'ssm.conv_kernel')
        state = state_size ** 2 * heads * 4
        conv = (conv_kernel - 1) * (inner_size + 2 * group_count * state_size) * 4
        recurrent = 2 * (state + conv) * deltanet_layers

    def unique_experts(k, mode):
        if mode == 'min':
            return n_used
        if mode == 'max':
            return min(n_used * k, n_expert)
        return n_expert * (1 - (1 - n_used / n_expert) ** k)  # independent uniform routing

    def main_step(k, depth, mode):
        weights = sum(l['nonrouted'] + unique_experts(k, mode) * l['expert'] for l in layers) + head
        return {
            'weights': weights,
            'kv': attn_layers * kv_per_token_layer * depth,
            'recurrent': recurrent,
            'embd': k * embd_row,
        }

    def draft_step(depth):
        if n_nextn == 0:
            return {'weights': 0, 'kv': 0, 'embd': 0}
        return {
            'weights': mtp['nonrouted'] + n_used * mtp['expert'] + head,
            'kv': kv_per_token_layer * depth,
            'embd': embd_row,
        }

    rows = []
    for depth in depths:
        row = {'depth': depth}
        for k in (1, 2, 3):
            for mode in ('min', 'independent', 'max'):
                parts = main_step(k, depth, mode)
                row[f'main_k{k}_{mode}_bytes'] = sum(parts.values())
        row['main_kv_bytes'] = main_step(1, depth, 'min')['kv']
        row['draft_bytes'] = sum(draft_step(depth).values())
        for mode in ('min', 'independent', 'max'):
            cycle = row[f'main_k3_{mode}_bytes'] + 2 * row['draft_bytes']
            row[f'cycle_mtp2_{mode}_bytes'] = cycle
            row[f'cycle_mtp2_{mode}_floor_ms'] = cycle / stream_gbps / 1e6
        row['main_k1_floor_ms'] = row['main_k1_min_bytes'] / stream_gbps / 1e6
        row['main_k3_independent_floor_ms'] = row['main_k3_independent_bytes'] / stream_gbps / 1e6
        row['draft_floor_ms'] = row['draft_bytes'] / stream_gbps / 1e6
        rows.append(row)

    return {
        'model': str(model),
        'architecture': arch,
        'vocab_size': n_vocab,
        'bandwidth_gbps': stream_gbps,
        'bandwidth_source': stream_source,
        'head_tied_to_embedding': head_tied,
        'block_count': n_layer,
        'nextn_predict_layers': n_nextn,
        'expert_count': n_expert,
        'expert_used_count': n_used,
        'main_layers': len(layers),
        'attention_layers': attn_layers,
        'deltanet_layers': deltanet_layers,
        'nonrouted_main_bytes': sum(l['nonrouted'] for l in layers),
        'head_bytes': head,
        'routed_bytes_per_token_main': n_used * sum(l['expert'] for l in layers),
        'mtp_nonrouted_bytes': mtp['nonrouted'],
        'mtp_routed_bytes': n_used * mtp['expert'],
        'kv_bytes_per_token_main': attn_layers * kv_per_token_layer,
        'kv_bytes_per_token_mtp': kv_per_token_layer if n_nextn else 0,
        'recurrent_state_rw_bytes': recurrent,
        'embedding_row_bytes': embd_row,
        'rows': rows,
        'assumptions': [
            'each weight byte read once per step',
            'routed experts: min/independent/max unique per layer',
            'MTP cycle = one 3-token main step + two 1-token MTP passes (catch-up batching ignored)',
            'KV bytes assume uniform head_count_kv and key_length across attention layers',
            'activations, logits copies, sampling, synchronization and launch costs excluded',
        ],
    }


def parse_depths(value):
    depths = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            depth = int(part)
        except ValueError:
            raise SystemExit(f'Invalid depth: {part!r}')
        if depth < 0:
            raise SystemExit(f'Depth must be non-negative: {depth}')
        depths.append(depth)
    if not depths:
        raise SystemExit('At least one depth is required')
    return depths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path, help='output JSON file')
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='GGUF model file (default: %(default)s)')
    p.add_argument('--gguf-py', type=Path, help='directory containing gguf/__init__.py')
    p.add_argument('--stream-gbps', type=float, default=DEFAULT_STREAM_GBPS,
                   help='measured streaming read bandwidth in GB/s (default: %(default)s)')
    p.add_argument('--stream-source', default=DEFAULT_STREAM_SOURCE,
                   help='where the bandwidth figure came from')
    p.add_argument('--depths', default=','.join(str(d) for d in DEFAULT_DEPTHS),
                   help='comma-separated populated-context depths (default: %(default)s)')
    args = p.parse_args()

    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f'Model not found: {model}. Set LLM_MODEL_PATH or pass --model.')
    if args.stream_gbps <= 0:
        raise SystemExit('--stream-gbps must be positive')
    depths = parse_depths(args.depths)
    gguf_py = find_gguf_py(args.gguf_py)

    tensors, hp, n_vocab, arch = load(model, gguf_py)
    result = budget(tensors, hp, n_vocab, model, arch, args.stream_gbps, args.stream_source, depths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

    gb = 1e9
    print(f"architecture {arch}, vocab {n_vocab}, gguf-py {gguf_py}")
    print(f"main non-routed {result['nonrouted_main_bytes']/gb:.3f} GB (head {result['head_bytes']/gb:.3f}), "
          f"routed/token {result['routed_bytes_per_token_main']/gb:.3f} GB, "
          f"MTP layer {result['mtp_nonrouted_bytes']/gb:.3f}+{result['mtp_routed_bytes']/gb:.3f} GB, "
          f"KV/token {result['kv_bytes_per_token_main']} B, recurrent r+w {result['recurrent_state_rw_bytes']/gb:.3f} GB")
    for row in result['rows']:
        print(f"d={row['depth']:6d}  k1 {row['main_k1_floor_ms']:6.1f} ms  "
              f"k3(indep) {row['main_k3_independent_floor_ms']:6.1f} ms  "
              f"draft {row['draft_floor_ms']:5.1f} ms  "
              f"cycle {row['cycle_mtp2_min_floor_ms']:6.1f}/"
              f"{row['cycle_mtp2_independent_floor_ms']:6.1f}/"
              f"{row['cycle_mtp2_max_floor_ms']:6.1f} ms")


if __name__ == '__main__':
    main()
