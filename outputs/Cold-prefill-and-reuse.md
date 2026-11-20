# Cold prefill and persistent prefix reuse on the Radeon 860M

> **Note:** This is the public copy of this report. Small summary files (summaries, samples, manifests) are included, with local paths replaced by placeholders such as `<repo>` and `<models>`. The full raw data (request/response dumps, server logs, generated text, build logs and test fixtures) is not public, so some links in this report do not resolve.

The useful new capability is saving a populated server slot to a local file and reloading it after the server restarts. This was tested successfully with over 100K cached tokens, including this model's hybrid and MTP state. It is a conversation-cache operation; it is not filesystem repair or recovery.

Keep ROCm, F16 K/V, MTP 2, context 163840, one slot, batch 2048 and microbatch 512 as the baseline. Microbatch 1024 is a reasonable cold-prefill candidate, but it did not beat 512 in the matched long-context suffix test. Microbatch 2048 was slower in that test. No global optimum or sustained 15 tokens/s floor has been established.

## Full cold run

User hardware: Ryzen AI 7 350 / Radeon 860M, 64 GB DDR5-5600, Windows. llama.cpp b11888 / 48c9f08b7, ROCm0, isolated ROCm 10.0 runtime. Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf. Both runs use F16 K/V, MTP 2, allocated context 163840, one slot, GPU layers 99, Flash Attention, batch 2048, eight threads and auto load mode. The draft context follows the target microbatch default.

| Measurement | Microbatch 512, earlier baseline | Microbatch 1024, new |
|---|---:|---:|
| Cold input tokens, none cached | 102400 | 102400 |
| Prompt evaluation seconds | 807.994 | 748.912 |
| Prompt tokens/s | 126.734 | 136.732 |
| Cold client TTFT seconds | 808.047 | 748.980 |
| Initial generated tokens | 512 | 512 |
| Initial generation tokens/s | 15.744 | 16.315 |
| Cold request client wall seconds | 840.517 | 780.317 |
| Follow-up reused tokens | 102911 | 102911 |
| Follow-up processed tokens | 28 | 28 |
| Follow-up generated tokens | 256 | 256 |
| Follow-up generation tokens/s | 14.975 | 15.497 |
| Follow-up client TTFT seconds | 0.689 | 0.646 |
| Follow-up client wall seconds | 17.731 | 17.116 |

The cold request JSON hashes are identical: SHA256 `572a0e822a86213d60e076b486c7f72a22d8040a322d69605b487130c63c31cd`. The complete output token sequences matched the baseline on both the initial 512-token output and the 256-token follow-up. Neither request truncated.

The new prefill was 7.3% shorter, but these were single runs at different times without power, temperature or background-load controls. The new diagnostic run used log verbosity 4; the earlier baseline used 3. Do not attribute the entire improvement to microbatch size. The shorter 64-token cache probes below also produced generation averages below 15 tokens/s, so the new 15.50 follow-up does not establish a minimum rate.

## Matched prefill at a populated 100K prefix

The exact same saved slot was restored before each test. Each request reused 103194 tokens and processed 8193 tokens: an identical 8192-token new suffix plus one previous output token that needed evaluation. Total input was 111387, with 163840 allocated context. Only one output token was requested; this is a prompt-processing test, not a generation-speed benchmark.

| Microbatch | Prompt evaluation seconds | Prompt tokens/s | Client TTFT seconds |
|---|---:|---:|---:|
| 512 | 99.393 | 82.431 | 99.452 |
| 1024 | 99.391 | 82.432 | 99.457 |
| 2048 | 114.215 | 71.733 | 114.292 |

512 and 1024 are effectively tied in these samples. 2048 took about 15% longer. Requests match byte-for-byte across the three runs; launch commands are recorded with each result. Order was 1024, 2048, 512, without repeats or randomized ordering. The 1024 server had already performed the cache-validation requests; the other servers were freshly loaded before restoring the snapshot. These are practical elapsed measurements, not isolated kernel timings, and do not substitute for full cold runs at every microbatch size.

## Cache save and reload after a real process restart

The completed synthetic conversation was saved to `work\slot-cache\synthetic-helper-check.bin`. The server was stopped and a new process loaded with matching settings. The reusable PowerShell helper and the Python measurement probe both exercised the actual save/restore endpoints.

| Measurement | Observed |
|---|---:|
| Cached tokens saved and restored | 103194 |
| Snapshot bytes | 2659443228 (2.477 GiB) |
| Save, helper / second measured write | 2.611 s / 7.224 s |
| Reload, helper / measured probe | 2.309 s / 1.803 s |
| Continuation reused tokens | 103194 |
| Continuation newly processed tokens | 26 |
| Live-cache continuation TTFT | 0.602 s |
| Continuation TTFT after reload | 0.963 s |
| Output tokens compared | 64 |
| Live versus reloaded output token IDs | identical |

Reload time excludes starting the executable and loading model weights. The continuation TTFT excludes the preceding reload operation. Measurements followed recent writes and used a warm OS filesystem cache; no reboot or cold-storage-read test was performed. The snapshot survives a process restart, but these numbers do not establish reboot-to-first-token latency.

A separate controlled edit changed one token 128 tokens before the end of the saved history. The server restored a checkpoint at 102935 cached tokens, processed 260 tokens and reached its first output in 3.041 seconds. It did not reprocess the full prefix. This demonstrates checkpoint replay for this particular edit, not all possible history changes or Pi tool-call formats.

## Using this with Pi

The launcher now enables a local slot-save directory by default, at `work\slot-cache`. This enables the endpoint; it does not automatically save or restore a conversation. Passing `-SlotSavePath ''` disables this launcher feature. Defaults remain F16 K/V, MTP 2, batch 2048, microbatch 512 and context 163840. `-MicroBatchSize 1024` remains available for an explicit experiment.

Start the server in one terminal:

```powershell
$llmTools = '.\outputs'
& "$llmTools\Start-ROCmServer.ps1"
```

After Pi has finished a request, pause it. In a second terminal, save the current slot before shutting the server down:

```powershell
$llmTools = '.\outputs'
& "$llmTools\Manage-LLMSlot.ps1" -Action Save -Name my-project
& "$llmTools\Stop-ROCmServer.ps1"
```

On the next use, start the server with the same configuration. Once it reports ready, run this in the second terminal before sending another Pi request:

```powershell
& "$llmTools\Manage-LLMSlot.ps1" -Action Restore -Name my-project
```

Then resume the same Pi conversation. Loading the server cache does not reopen a Pi session or restore project files. Pi still sends its history; the server can reuse the matching token prefix. New history, changes near its beginning, different tool definitions, or changed formatting can invalidate part or all of that prefix. A new prefix still needs prefill.

The helper requires one idle slot, verifies the configured local save directory, and checks the model path, file size/modification time, build, context, K/V types, MTP settings and batch sizes against its saved metadata. Keep Pi paused during save/load. Saving again with the same name replaces that named snapshot. Use distinct names for distinct conversations. The synthetic snapshot retained here is benchmark evidence, not a Pi session to load for normal work.

The server already enables ordinary prompt caching by default. `--cache-ram` controls an in-memory prompt cache; it is not disk persistence. No changes were made to its limit, checkpoint count or checkpoint spacing. `--cache-reuse` performs KV shifting when supported; it is not a general repair for a changed prefix in a hybrid model. Pi's earlier prompt-processing totals alone cannot distinguish new tool output from replayed history, so no Pi-specific cache bug has been established or fixed here.

## Evidence and limitations

- [New complete cold run and cached continuation](rocm-100k-ub1024-results/manifest.json); requests, results and server log are in the same directory.
- [512 suffix result](deep-prefill-ub512-results/deep-prefill-continuation-result.json), [1024 suffix result](deep-prefill-ub1024-results/deep-prefill-continuation-result.json), [2048 suffix result](deep-prefill-ub2048-results/deep-prefill-continuation-result.json).
- [Restart comparison](slot-cache-reuse-results/comparison.json), [restored continuation](slot-cache-reuse-results/restore-continuation-result.json), [checkpoint-replay result](slot-cache-reuse-results/rewind-continuation-result.json).
- [Save/restore helper](Manage-LLMSlot.ps1), [launcher](Start-ROCmServer.ps1), [reproducible probe](probe_slot_cache.py).

The synthetic throughput workload and sampling settings match the earlier benchmark: repetitive-structure Python with unique names, temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1, presence penalty 0, completed empty thinking blocks, and forced output lengths. No coding/retrieval/reasoning/tool-call quality benchmark was added. Output-token equality is a limited state-restoration check, not a general quality guarantee. Pi-specific replay was not exercised with a live Pi session during these tests. No populated 160K benchmark was performed.

The inference server was shut down after the experiments. Model weights, Pi conversations and system files were not modified by these tests. The launcher and diagnostic helpers are local workspace artifacts.
