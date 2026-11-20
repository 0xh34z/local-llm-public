# Building the 128-thread attention experiment

> **Note:** This is the public copy of this report. Small summary files (summaries, samples, manifests) are included, with local paths replaced by placeholders such as `<repo>` and `<models>`. The full raw data (request/response dumps, server logs, generated text, build logs and test fixtures) is not public, so some links in this report do not resolve.

The private build uses a dedicated AMD HIP kernel and a guarded dispatcher in one DLL. It was assembled from the previous eight-block build of llama.cpp b11888, commit `48c9f08b7b25766324282b7605177e2f2be755bb`. The 256-thread control dispatches the existing kernel from its original object. The 128-thread setting uses a new specialization compiled for gfx1152. [Builder](../build_attention_workgroup.py) · [Compiler and linker commands, source hashes and DLL hashes](build.json)

The starting RDNA configuration for this shape specifies 256 threads, a target occupancy of two and 64 KV rows per softmax rescaling. The private source changes the thread count to 128. Both host and device code use that configuration, including launch bounds, the workgroup dimensions passed to HIP, warp indexing and the host calculation of dynamic shared memory. Other configuration fields keep their original values. The kernel has a distinct device symbol, `flash_attn_ext_f16_workgroup_128`.

The dispatch applies to aligned tensors on gfx1152 with F32 query dimensions [256, 3, 16, 1], F16 K/V with two heads and D256, a mask, zero bias and softcap, no attention sinks, and a padded KV length of at least 100000. The tile is four queries by eight GQA lanes. Both settings pin eight stream-K blocks for this case, removing the dependence of the block count on the compiler's occupancy result. The rest of the model uses existing dispatch.

## Running the isolated build

From the repository root, use the existing compiler, Visual Studio environment and built objects in `work/`.

```powershell
& .\work\rocm10-runtime\Scripts\python.exe outputs/build_attention_workgroup.py
```

The output lives in `work/hip-workgroup/bin`. Select `GGML_HIP_FA_WORKGROUP=128` or `256` with `GGML_HIP_FA_KV_SPLIT=double` and automatic tile/family selection. The first matching dispatch prints `HIP_FA_WORKGROUP`; the split marker reports actual blocks, warps and dynamic shared bytes. On this wave32 GPU, four and eight warps correspond to 128 and 256 threads respectively.

The test DLL was 72,366,080 bytes, SHA256 `fe56b1843c1fad6993455bfc46ea7b7c2f58581a49a7f2a9bae1b406e5e8886f`. The production eight-block DLL still hashed to `51230c04fe096226e2a8cd5fc3a37cfc4930c13fc23e086c4c2771bf7343b4b5` after building. AMD clang emitted warnings including unsupported import/export attributes during device compilation. The compiler and linker exited successfully. The complete local build log is in [build.log](../../work/hip-workgroup/build.log).

## Separate query-storage experiment

The compiled 256-thread control reported 169 vector-register spills and 416 private scratch bytes per lane. A second specialization selects the existing `Q_in_reg=false` code path and retains 256 threads. This path stores the query in GPU local data share (LDS) between operations. Its gfx1152 code object reported 13 vector-register spills and 56 private bytes per lane. These are compiler resource figures. The additional shared storage and accesses can affect performance in either direction. [Kernel metadata](kernel-resources.json)

The two commands below compile the separate object and link the new dispatcher into `work/hip-query-lds/bin`. Use `GGML_HIP_FA_QUERY_STORAGE=registers` or `lds` with `GGML_HIP_FA_KV_SPLIT=double` and automatic tile/family selection. Both modes use eight stream-K blocks, the original 32-column geometry, and 256 threads. The shape guard matches the workgroup experiment above. Logs identify the query-storage selection and actual workgroup geometry. [Object compiler](../build_attention_qlds.py) · [Dispatcher and linker](../link_attention_query_storage.py)

```powershell
& .\work\rocm10-runtime\Scripts\python.exe outputs/build_attention_qlds.py
& .\work\rocm10-runtime\Scripts\python.exe outputs/link_attention_query_storage.py
```
