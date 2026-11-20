# Attention workgroup experiment on the Radeon 860M

## Question

Does using 128 GPU threads per workgroup in the long-context, three-query attention kernel improve the speed of replies at a populated 160K context? The existing kernel uses 256 threads. Both variants should use 32 columns, eight stream-K blocks, F16 keys and values, and the same output reduction. Other attention shapes should use their usual kernels.

The previous four control launches produced the same 128-token repetitive answer at both lengths. Dividing generation time by the number of tokens after the first gives 48.07 ms/token at 103K and 56.45 at 160K. The slowest sampled 160K code continuation took 70.79 ms/token. A rate of 15 tokens/s corresponds to 66.67 ms/token. The 4.12 ms difference is the size of improvement needed on that sample. [Earlier generation records](../fine-split-generation/samples.json)

## Controls and decision

1. Build the experimental HIP DLL under `work/`. Verify which specialization the host launches, the compiled workgroup size and the eight-block split. A successful compile alone does not confirm runtime selection.
2. Exercise the private control and candidate on Q3 contiguous and strided F16 K/V fixtures at both long-context lengths. Use the existing independent float64 reference and its pre-existing normalized mean squared error threshold of 0.0005. Record the upstream CPU comparison separately. [Reference implementation](../check_attention_float64.py)
3. Compare repeated isolated attention timings with the control interleaved. Use a warmed GPU. Preserve dispatch logs, samples and any failed attempts.
4. When warranted, restore identical synthetic prefixes and run ordinary inference with operation profiling disabled. Compare sampled code and deterministic generation at actual 103K and 160K input. Check output token IDs and speculative draft work. Report rates at both lengths, including the slow 160K samples. Keep the usual production setting unless repeatable end-to-end evidence supports a change.

The usual server uses one slot, an allocation of 163840, MTP 2, F16 K/V, logical batch 2048, microbatch 512 and eight CPU threads. The trial targets GPU threads per workgroup. Windows AC mode was Balanced at the start. The live production `ggml-hip.dll` SHA-256 was `51230c04fe096226e2a8cd5fc3a37cfc4930c13fc23e086c4c2771bf7343b4b5`. The source checkout is llama.cpp b11888 at `48c9f08b7b25766324282b7605177e2f2be755bb` with the local long-context patches. [Machine and driver](../laptop-hardware-round/system.json)
