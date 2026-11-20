"""Bytes each decode step must read, from the GGUF tensor table, and the bandwidth floor they imply.

This is a calculation, not a measurement. It counts each weight byte once per step (perfect reuse
across the tokens of a batch) and ignores activations, logits transfer, sampling and launch overhead.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import types

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get('LLM_MODEL_PATH', Path.home() / 'Models' / 'Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf'))
STREAM_GBPS = 81.98  # fastest standalone GPU read median, outputs/stream-bandwidth-gfx1152
DEPTHS = (0, 8192, 32768, 65536, 103194, 131072, 160511)


def load(model):
    sys.modules.setdefault('yaml', types.ModuleType('yaml'))  # gguf-py imports yaml only for metadata helpers
    sys.path.insert(0, str(ROOT/'work/llama.cpp/gguf-py'))
    from gguf import GGUFReader
    reader = GGUFReader(str(model))
    fields = {}
    for f in reader.fields.values():
        if f.name.startswith('qwen35moe.') and len(f.data) == 1:
            fields[f.name.split('.', 1)[1]] = f.parts[f.data[0]].tolist()[0]
    return {t.name: int(t.n_bytes) for t in reader.tensors}, fields


def budget(tensors, hp):
    n_layer = hp['block_count']
    n_nextn = hp['nextn_predict_layers']
    main_layers = range(n_layer - n_nextn)
    mtp_layer = n_layer - n_nextn
    n_expert, n_used = hp['expert_count'], hp['expert_used_count']
    head = tensors['output.weight']
    embd_row = tensors['token_embd.weight'] // 248320

    def layer(i):
        prefix = f'blk.{i}.'
        own = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
        routed = sum(v for k, v in own.items() if k.endswith('_exps.weight'))
        return dict(nonrouted=sum(v for k, v in own.items() if not k.endswith('_exps.weight')),
                    expert=routed / n_expert, attention='attn_q.weight' in own)

    layers = [layer(i) for i in main_layers]
    mtp = layer(mtp_layer)
    attn_layers = sum(l['attention'] for l in layers)
    kv_per_token_layer = 2 * hp['attention.head_count_kv'] * hp['attention.key_length'] * 2  # K+V, F16
    # Gated DeltaNet recurrent state: F32 [S_k, S_v, heads], read and written once per step (in place).
    heads = hp['ssm.time_step_rank']
    state = hp['ssm.state_size'] ** 2 * heads * 4
    conv = (hp['ssm.conv_kernel'] - 1) * (hp['ssm.inner_size'] + 2 * hp['ssm.group_count'] * hp['ssm.state_size']) * 4
    recurrent = 2 * (state + conv) * (len(layers) - attn_layers)

    def unique_experts(k, mode):
        if mode == 'min':
            return n_used
        if mode == 'max':
            return min(n_used * k, n_expert)
        return n_expert * (1 - (1 - n_used / n_expert) ** k)  # independent uniform routing

    def main_step(k, depth, mode):
        weights = sum(l['nonrouted'] + unique_experts(k, mode) * l['expert'] for l in layers) + head
        return dict(weights=weights, kv=attn_layers * kv_per_token_layer * depth, recurrent=recurrent,
                    embd=k * embd_row)

    def draft_step(depth):
        return dict(weights=mtp['nonrouted'] + n_used * mtp['expert'] + head, kv=kv_per_token_layer * depth,
                    embd=embd_row)

    rows = []
    for depth in DEPTHS:
        row = dict(depth=depth)
        for k in (1, 2, 3):
            for mode in ('min', 'independent', 'max'):
                parts = main_step(k, depth, mode)
                row[f'main_k{k}_{mode}_bytes'] = sum(parts.values())
        row['main_kv_bytes'] = main_step(1, depth, 'min')['kv']
        row['draft_bytes'] = sum(draft_step(depth).values())
        for mode in ('min', 'independent', 'max'):
            cycle = row[f'main_k3_{mode}_bytes'] + 2 * row['draft_bytes']
            row[f'cycle_mtp2_{mode}_bytes'] = cycle
            row[f'cycle_mtp2_{mode}_floor_ms'] = cycle / STREAM_GBPS / 1e6
        row['main_k1_floor_ms'] = row['main_k1_min_bytes'] / STREAM_GBPS / 1e6
        row['main_k3_independent_floor_ms'] = row['main_k3_independent_bytes'] / STREAM_GBPS / 1e6
        row['draft_floor_ms'] = row['draft_bytes'] / STREAM_GBPS / 1e6
        rows.append(row)
    return dict(
        model=str(MODEL), bandwidth_gbps=STREAM_GBPS,
        main_layers=len(layers), attention_layers=attn_layers, deltanet_layers=len(layers) - attn_layers,
        nonrouted_main_bytes=sum(l['nonrouted'] for l in layers), head_bytes=head,
        routed_bytes_per_token_main=n_used * sum(l['expert'] for l in layers),
        mtp_nonrouted_bytes=mtp['nonrouted'], mtp_routed_bytes=n_used * mtp['expert'],
        kv_bytes_per_token_main=attn_layers * kv_per_token_layer, kv_bytes_per_token_mtp=kv_per_token_layer,
        recurrent_state_rw_bytes=recurrent, rows=rows,
        assumptions=['each weight byte read once per step', 'routed experts: min/independent/max unique per layer',
                     'MTP cycle = one 3-token main step + two 1-token MTP passes (catch-up batching ignored)',
                     'activations, logits copies, sampling, synchronization and launch costs excluded'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    args = p.parse_args()
    tensors, hp = load(MODEL)
    result = budget(tensors, hp)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    gb = 1e9
    print(f"main non-routed {result['nonrouted_main_bytes']/gb:.3f} GB (head {result['head_bytes']/gb:.3f}), "
          f"routed/token {result['routed_bytes_per_token_main']/gb:.3f} GB, "
          f"MTP layer {result['mtp_nonrouted_bytes']/gb:.3f}+{result['mtp_routed_bytes']/gb:.3f} GB, "
          f"KV/token {result['kv_bytes_per_token_main']} B, recurrent r+w {result['recurrent_state_rw_bytes']/gb:.3f} GB")
    for r in result['rows']:
        print(f"d={r['depth']:6d}  k1 {r['main_k1_floor_ms']:6.1f} ms  k3(indep) {r['main_k3_independent_floor_ms']:6.1f} ms"
              f"  draft {r['draft_floor_ms']:5.1f} ms  cycle {r['cycle_mtp2_min_floor_ms']:6.1f}"
              f"/{r['cycle_mtp2_independent_floor_ms']:6.1f}/{r['cycle_mtp2_max_floor_ms']:6.1f} ms")


if __name__ == '__main__':
    main()
