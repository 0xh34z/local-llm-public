# GPU kernel selection, register use and shared memory

> **Note:** This is the public copy of this report. Small summary files (summaries, samples, manifests) are included, with local paths replaced by placeholders such as `<repo>` and `<models>`. The full raw data (request/response dumps, server logs, generated text, build logs and test fixtures) is not public, so some links in this report do not resolve.

## Compiled kernel resources

The embedded AMDGPU code object contains a descriptor for each gfx1152 kernel. The table uses the dense, masked D256, 4-query-column × 8-head-column specialization with softcap disabled and separate K/V tensors. Every row uses 32-lane waves. The regular D256 entry has the same device image in the working eight-block DLL and the private workgroup DLL. The extraction script verified the bytes in both files. [Exact symbols, image hashes and compiler fields](kernel-resources.json) · [Reader](../read_attention_resources.py)

| Configuration | Threads | Waves per workgroup | SGPRs | VGPRs | Compiler VGPR spill count | Private bytes per lane | Dynamic shared bytes per workgroup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q held in registers | 256 | 8 | 107 | 256 | 169 | 416 | 34,816 |
| Q held in registers | 128 | 4 | 103 | 256 | 478 | 1,312 | 34,368 |
| Q held in shared memory | 256 | 8 | 98 | 256 | 13 | 56 | 51,264 |

The thread column is the compiled maximum workgroup size. The register, spill and private-memory columns are compiler descriptors. Waves per workgroup are calculated as threads divided by 32. Each scalar spill count and each static group-segment size was zero. The last column comes from host launch calculations. The 128- and 256-thread workgroup fixture logs confirm their corresponding dynamic allocations. Spill counts and private-segment sizes describe the compiled kernels. The actual traffic depends on the executed paths and cache behavior. Device occupancy and elapsed-time predictions cannot be read directly from these counts. [128-thread launch](../workgroup-screen/contiguous100k-128-test.log) · [256-thread launch](../workgroup-screen/contiguous100k-256-test.log)

## Host and device agreement

The Radeon specialization normally uses 256 threads, 32 columns, 64 KV rows per iteration, 128 half2 elements across both K and V, and 64 half2 elements per combine chunk. The private 128-thread translation unit changes the thread count in the shared host/device configuration row and emits a separate kernel symbol. This propagates to the device launch bound, device loops, host launch dimensions, occupancy query and host dynamic shared-memory calculation. The scoped dispatcher keeps the usual kernel for other tensor shapes. [Private compiler and dispatch setup](../build_attention_workgroup.py)

With 16 columns per wave and 32 columns per tile, the number of waves computing parallel partials for each column drops from four to two at 128 threads. Each lane consequently retains more partial values across some loop iterations. The compiler spill count rises in this build. The differential interpretation depends on the full-kernel execution, including memory access and reductions.

The automatic stream-K grid uses the resident-block estimate. The earlier eight-block setting doubled that automatic number. The private comparison explicitly selects eight blocks within the narrow experiment. The old control and new 128-thread paths both logged four normal blocks, two output tiles and eight actual blocks in the numerical fixture. This distinguishes the number of scheduled blocks across the grid from threads per block. [Original control launch](../fine-split-screen/strided160k-8-perf-2.log) · [128-thread launch](../workgroup-screen/contiguous100k-128-test.log)

## A second hypothesis from the compiler

Keeping 256 threads and switching `Q_in_reg` to false reduces the live query-fragment array from 16 entries to one for this D256 shape. The inner loop now loads the query fragment it needs from workgroup shared memory. This trades register spills against extra shared-memory loads and additional reserved shared storage. The per-block shared-memory request increases by 16,448 bytes after accounting for scratch areas that overlap. This candidate addresses the register-pressure evidence while retaining the original tile dimensions, KV precision and thread count. The numerical and throughput measurements determine whether the trade helps. [Single-flag compilation script](../build_attention_qlds.py)

## Reproduction

From the repository root, build the private workgroup runtime, compile the 256-thread shared-query object, and read the compiler descriptors into a fresh JSON path.

```powershell
& .\work\rocm10-runtime\Scripts\python.exe outputs/build_attention_workgroup.py
& .\work\rocm10-runtime\Scripts\python.exe outputs/build_attention_qlds.py
& .\work\rocm10-runtime\Scripts\python.exe outputs/read_attention_resources.py outputs/resources-repeat.json
```

The reader extracts the tagged device ELF from the Windows HIP object and invokes the local SDK's `llvm-readobj --notes`. The `--qlds-dll` option also checks that the shared-query device image occurs byte-for-byte in the specified private DLL. The code-object fields describe this build of this particular specialization.
