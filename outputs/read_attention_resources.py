"""Read exact long-context attention code-object descriptors from Windows HIP objects."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

from build_attention_workgroup import BUILD, WORK


FIELDS = ('max_flat_workgroup_size', 'wavefront_size', 'sgpr_count', 'vgpr_count',
          'sgpr_spill_count', 'vgpr_spill_count', 'private_segment_fixed_size',
          'group_segment_fixed_size', 'workgroup_processor_mode')


def gfx1152_elf(obj):
    bundle = obj.read_bytes()
    signature = b'__CLANG_OFFLOAD_BUNDLE__'
    start = bundle.index(signature)
    cursor = start + len(signature)
    count, = struct.unpack_from('<Q', bundle, cursor)
    cursor += 8
    for _ in range(count):
        offset, size, name_len = struct.unpack_from('<QQQ', bundle, cursor)
        cursor += 24
        target = bundle[cursor:cursor + name_len]
        cursor += name_len
        if target == b'hipv4-amdgcn-amd-amdhsa--gfx1152':
            image = bundle[start + offset:start + offset + size]
            if len(image) != size or not image.startswith(b'\x7fELF'):
                raise ValueError(f'Invalid embedded code object in {obj}')
            return image
    raise ValueError(f'No gfx1152 bundle entry in {obj}')


def read_kernel(obj, function, folder, dlls):
    image = gfx1152_elf(obj)
    path = folder/(function + '.elf')
    path.write_bytes(image)
    llvm = WORK/'rocm10-build-tools/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/llvm-readobj.exe'
    note = subprocess.check_output([llvm, '--notes', path], text=True)
    (folder/(function + '.notes')).write_text(note)
    # Template arguments are DKQ, DV, query columns, head columns, softcap, V-as-K-view, sparse.
    ident = f'{function}ILi256ELi256ELi4ELi8ELb0ELb0ELb0EE'
    matches = [k for k in note.split('  - .args:') if re.search(r'^    \.name:\s+\S*' + ident, k, re.M)]
    if len(matches) != 1:
        raise ValueError(f'Expected one non-softcapped, dense target in {obj}, found {len(matches)}')
    kernel = matches[0]
    row = {'object': str(obj.relative_to(WORK.parent)),
           'image_sha256': hashlib.sha256(image).hexdigest(),
           'mangled_name': re.search(r'^    \.name:\s+(\S+)', kernel, re.M)[1],
           'verified_image_offsets': {}}
    for dll in dlls:
        offset = dll.read_bytes().find(image)
        if offset < 0:
            raise ValueError(f'Code object from {obj} is absent in {dll}')
        row['verified_image_offsets'][str(dll.relative_to(WORK.parent))] = offset
    for key in FIELDS:
        row[key] = int(re.search(r'^    \.' + key + r':\s+(\d+)$', kernel, re.M)[1])
    disassembler = llvm.with_name('llvm-objdump.exe')
    assembly = subprocess.check_output([disassembler, '--disassemble-symbols=' + row['mangled_name'], path], text=True)
    (folder/(function + '.asm')).write_text(assembly)
    row['static_instruction_sites'] = {
        prefix: len(re.findall(r'^\s*' + prefix + r'\w*\s', assembly, re.M))
        for prefix in ('scratch_load', 'scratch_store', 'ds_load')
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--qlds-dll', type=Path)
    args = parser.parse_args()
    folder = WORK/'hip-wg-metadata'
    folder.mkdir(exist_ok=True)
    obj, = BUILD.rglob('fattn-mma-f16-instance-ncols1_4-ncols2_8.cu.obj')
    new_dll = WORK/'hip-workgroup/bin/ggml-hip.dll'
    rows = {'control': read_kernel(obj, 'flash_attn_ext_f16', folder, [WORK/'hip-kv-split-bin/ggml-hip.dll', new_dll]),
            'threads_128': read_kernel(WORK/'hip-workgroup/workgroup-128.obj',
                                       'flash_attn_ext_f16_workgroup_128', folder, [new_dll]),
            'q_lds_256': read_kernel(folder/'qlds/qlds-256.obj', 'flash_attn_ext_f16_qlds_256', folder,
                                     [args.qlds_dll.resolve()] if args.qlds_dll else [])}
    with args.out.open('x', encoding='utf-8') as output:
        json.dump(rows, output, indent=2)


if __name__ == '__main__':
    main()
