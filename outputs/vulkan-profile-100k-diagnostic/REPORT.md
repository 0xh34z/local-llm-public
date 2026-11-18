# Vulkan profiling: the 100K bottleneck is long-context attention

## Configuration and collection

User's Radeon 860M, Windows Vulkan, llama.cpp b11888/48c9f08b7, Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf, MTP maximum one draft token, F16 K/V, context allocation 163840, one slot, GPU layers 99, Flash Attention enabled, batch 2048, microbatch 512, eight CPU threads, auto load mode.

The installed ggml-vulkan.dll was checked for the built-in GGML_VK_PERF_LOGGER, GGML_VK_PERF_LOGGER_CONCURRENT and GGML_VK_PERF_LOGGER_FREQUENCY settings. The profiling launcher enables them for its process, using concurrent grouping and a logging frequency of 32 graphs, and restores its inherited environment on exit.

A 1024-input-token, 128-output-token diagnostic was followed by a 102400-input-token, 128-output-token diagnostic and a cached 128-output-token follow-up. The long cold prompt uses the same token IDs as the earlier F16 benchmark. Output lengths are forced and prompting is non-thinking. This is operation profiling, not another normal throughput comparison.

## Generation-only recorded GPU time

The logger emits timing windows for separate backend contexts without explicit main/draft labels. Windows containing Gated DeltaNet are classified as main-model-like, consistent with the 40-layer target; windows without it that contain the full vocabulary projection are classified as draft-like, consistent with the single MTP layer. These family labels are inferences from the operation patterns. Windows containing prompt batches or query lengths greater than two were excluded from this generation summary.

| Recorded operation groups | Short sample (~1K populated) | Long sample (~100K populated) |
|---|---:|---:|
| Flash Attention groups | 1.61% | 52.79% |
| Other matrix multiplication groups | 41.83% | 19.59% |
| Expert matrix multiplication groups | 32.38% | 15.54% |
| Recurrent groups (Gated DeltaNet / SSM convolution) | 10.88% | 5.03% |
| Remaining groups | 13.30% | 7.05% |

The short result covers one generation-only main-like timing window. The long result combines two main-like windows weighted by recorded time. Their individual attention fractions were 52.63% and 52.96%, respectively. The long attention tensor shape contains two query tokens, 16 query heads, two KV heads, head dimension 256 and padded KV length 102656. That padded length is a logged kernel dimension, not a claim that the input contained more than 102400 tokens.

In the two long windows, the unfused Flash Attention group averaged about 6.11 ms per recorded call. Each main-like window had 320 attention calls across 32 graph executions, consistent with ten full-attention layers. Attention accounted for roughly 61 ms of recorded GPU time per such graph execution. An execution verifies multiple query positions; this is not 61 ms per accepted output token.

The short profile shows GATED_DELTA_NET/SILU groups on the GPU. The long generation-only windows show the same operation family on the GPU. Thus a hypothesis that the relevant Gated DeltaNet decode operation is entirely missing from Vulkan and falling back to CPU is contradicted by these samples. This does not establish that every operation is offloaded or exclude other CPU costs.

Draft-like long-context windows devote about 50-51% of recorded time to attention and about 35-36% to a vocabulary projection with shape m=248320, n=1, k=2048. In short draft-like windows, that projection accounted for about 70-72%. Those percentages belong to separate backend-context windows and must not be added directly to main-model percentages or interpreted as fractions of whole-request wall time.

## What this supports

There is now a measured, specific optimization target: the long-context Flash Attention path for this model's shapes on the 860M. Comparing another backend's attention implementation, or profiling and tuning that Vulkan kernel, is more justified than assuming a generic compiler rebuild will make the whole workload faster. Matrix/expert operations remain a secondary substantial target.

This profile does not distinguish bandwidth limitations from arithmetic throughput, occupancy, dispatch behavior or synchronization. Hardware counters or a backend/kernel comparison are needed before selecting an implementation change. A different engine cannot be promised to reach 15 tokens/s from this evidence.

The previous uninstrumented best result was about 12.38 tokens/s initially and 12.05 on a follow-up, so the user's threshold requires approximately 21-25% improvement. An illustrative Amdahl model using a 53% attention fraction would require about 1.5-1.6x faster attention to produce that overall improvement if all other time were fixed and the recorded GPU fraction equalled the relevant end-to-end fraction. Those assumptions are not established; this is a research target, not a speed prediction.

## Limits and artifacts

Instrumentation changes execution and synchronization. Do not rank this profile's tokens/s against normal benchmarks. Fused/concurrent operation groups cannot be broken into individual member costs from this log. Group categorization gives a group's whole measured duration to a contained operation family. GPU timing windows do not constitute a full CPU+GPU wall-time trace. There are few generation samples, and background workload, power and thermal state were not controlled. No quality score was measured.

The diagnostic requests completed without truncation. The initial long input processed 102400 new tokens; its follow-up reused 102527 tokens and processed 28 new tokens. Raw HTTP responses and timing events are saved here. vulkan-profile-snapshot.log contains the full logger capture; timing-summary.json contains parsed windows. manifest.json records the byte offset where the long diagnostic began. The short-only snapshot and summary are in the sibling vulkan-profile-diagnostic directory.

The profiling server is still running. Stop it with Ctrl+C before ordinary Pi use; the launcher restores its process environment when it exits. Its profiling configuration is diagnostic and should not be treated as the normal production server configuration.
