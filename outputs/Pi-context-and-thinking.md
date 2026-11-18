# Pi context and thinking recommendation, 2026-10-29

Keep model `contextWindow` and server allocation at 163840. For the local Qwen model, set per-model compaction `reserveTokens` to 61440: Pi's inspected 1.4.0 trigger is usage > contextWindow - reserveTokens, or approximately 102400 tokens. This is an early-compaction trigger, not a hard input cap; a large tool result can overshoot it. Keep `maxTokens` at 16384 for combined reasoning/output. The 61440 compaction reserve does not authorize a 61440-token response.

At initial inspection the installed settings set default thinking to off, while the user's model entry omitted `reasoning` (defaults false). Missing reasoning metadata does not tell llama.cpp to disable its model's thinking. Thus Pi's UI can show no thinking controls while server/template defaults still produce reasoning. More reasoning consumes tokens, context and task time; hiding a display block does not improve decoding speed.

`pi-models-recommended.json` is a complete provider example. `pi-settings-recommended-merge.json` is a fragment to merge into the existing settings, preserving preferences. Neither was copied into the live profile. The per-model compaction override key is exactly `llama_cpp/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf`.

The model uses `reasoning: true`, `thinkingFormat: qwen-chat-template` and `thinkingTokenBudgetField: thinking_budget_tokens`. Pi's installed provider emits explicit thinking on/off and preserves prior reasoning. llama.cpp b11888 `server-common.cpp` accepts `chat_template_kwargs.enable_thinking` and the `thinking_budget_tokens` alias for its sampler reasoning budget. The override keeps system messages as system and uses max_tokens.

Offline verification with the installed provider and no network calls is in `pi-thinking-payload-check.json`, reproduced by `node outputs/check_pi_thinking_payload.mjs`. Off emits false with no thinking cap; low emits true and 2048; medium emits true and 8192. These are token-budget controls, not evidence of calibrated native Qwen low/medium/high reasoning strength. No live inference or tool-call quality check was performed with this config.

Recommended starting default: `defaultThinkingLevel: low`, included in the settings merge fragment. Low caps reasoning at 2048 tokens; the model can finish sooner. Use off for simple lookups/mechanical edits and medium (8192 thinking tokens) for difficult debugging or architectural reasoning. The shared response maximum remains 16384. This is a starting recommendation, not an accuracy-tested optimum.

Reload model configuration through `/model`, select the local model, then choose `/thinking off`, `/thinking low` or `/thinking medium`. Off prioritizes latency; enabling reasoning may help difficult tasks but increases total task time. Updating the example files does not change the live profile; existing sessions can retain their previous thinking selection.

Sources: installed Pi 1.4.0 model-config, settings-manager, compaction and openai-completions/simple-options sources; local llama.cpp b11888 server-common.cpp; https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models-configuration.md . The earlier benchmark results support approximately 100K as the more comfortable tested working region; they do not establish a universal 15 TPS floor.
