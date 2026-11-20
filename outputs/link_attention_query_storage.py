"""Stage registers and LDS storage for Q in the scoped 256-thread eight-block kernel."""
import json

from build_attention_qlds import OUT
from build_attention_workgroup import WORK, SOURCE, digest, link_runtime, replace_once


DEST = WORK/'hip-query-lds'
DISPATCH = r'''                if (aligned && !small && !large) {
                    const char * mode = std::getenv("GGML_HIP_FA_QUERY_STORAGE");
                    if (mode && (std::strcmp(mode, "registers") == 0 || std::strcmp(mode, "lds") == 0)) {
                        const char * split = std::getenv("GGML_HIP_FA_KV_SPLIT");
                        GGML_ASSERT(split && std::strcmp(split, "double") == 0);
                        static std::atomic<bool> logged_storage{false};
                        if (!logged_storage.exchange(true)) {
                            fprintf(stderr, "HIP_FA_QUERY_STORAGE mode=%s threads=256 columns=32 q=3 kv=%" PRId64 "\n",
                                    mode, K->ne[1]);
                        }
                        if (std::strcmp(mode, "lds") == 0) {
                            ggml_cuda_fattn_qlds_256(ctx, dst);
                        } else {
                            ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 4, 8>(ctx, dst);
                        }
                        break;
                    }
                }
'''


def main():
    metadata = json.loads((OUT/'build.json').read_text())
    kernel = OUT/'qlds-256.obj'
    if digest(kernel) != metadata['object_sha256'] or digest(SOURCE/'fattn-mma-f16.cuh') != metadata['source_sha256']:
        raise RuntimeError('Compile the matching query-storage object with build_attention_qlds.py')
    DEST.mkdir(parents=True, exist_ok=True)
    dispatcher = (SOURCE/'fattn.cu').read_text()
    dispatcher = replace_once(dispatcher, '#include <string>', '#include <string>\n'
                              'void ggml_cuda_fattn_qlds_256(ggml_backend_cuda_context &, ggml_tensor *);')
    anchor = '                if (aligned && small) {'
    dispatcher = replace_once(dispatcher, anchor, DISPATCH + anchor)
    dispatcher = replace_once(dispatcher, '        blocks = 2*normal_blocks;',
                              '        const char * storage = std::getenv("GGML_HIP_FA_QUERY_STORAGE");\n'
                              '        blocks = storage && (std::strcmp(storage, "registers") == 0 || '
                              'std::strcmp(storage, "lds") == 0) ? 8 : 2*normal_blocks;')
    (DEST/'fattn.cu').write_text(dispatcher)
    link_runtime(DEST, kernel)


if __name__ == '__main__':
    main()
