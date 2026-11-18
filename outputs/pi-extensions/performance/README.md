# Pi performance footer

A small local extension for installed Pi 1.4.0. It adds a status entry to the existing footer using `setStatus`, rather than replacing Pi's footer or forking the application.

Example format (illustrative fixture, not a new hardware measurement):

```text
Perf | TTFT 0.75s | 17.6 tok/s srv | 80 out | 5.0s done
```

Start Pi with the extension, while running your usual local inference server:

```powershell
pi --extension "C:\Users\kasper\Documents\GitHub\local-llm\outputs\pi-extensions\performance\index.js"
```

Installed for automatic loading on 2026-10-29 at `C:\Users\kasper\.pi\agent\extensions\local-llm-performance\` (`index.js` and `performance.mjs`). Pi's installed loader discovered exactly one copy with no errors, and the installed files match these repository sources. Restart Pi or use `/reload` in an existing session. No settings entry or explicit `--extension` argument is needed for this installation; avoid adding a second copy through those mechanisms. Live terminal rendering remains unverified.

For a different profile, either copy those two files into its `agent/extensions/local-llm-performance/` folder or append the repository entry-point path to the existing `extensions` array in its settings. `settings-merge.json` supplies the latter fragment; use only one loading method.

Commands: `/perf` shows the last measurement, `/perf off` disables the display and extra timing fields on future requests, `/perf on` enables them, and `/perf reset` clears the last measurement.

## Definitions

- **TTFT** is client-observed time from Pi's `before_provider_request` payload hook to the first non-empty thinking, text or tool name/arguments delta. Empty assistant-role, progress, and tool-ID-only metadata are ignored. Thinking counts even if its display is hidden. Tool/parser buffering can make this later than the engine's first internally decoded token. A `~` means the request hook was unavailable and assistant-message start supplied an approximate start time.
- **tok/s srv** is llama.cpp's reported `timings.predicted_per_second`, excluding prefill. The generated-token count includes thinking and tool syntax as reported by the server; reasoning tokens are not added twice. The counter does not divide characters by four or count SSE chunks as tokens. If valid server timing metadata is absent, TPS stays `--`.
- **out** is server-reported generated tokens, with final provider output usage as a fallback for the count. There is no fabricated live token estimate.
- **elapsed** is the client request interval. It includes prefill/waiting but excludes time between model requests, including tool execution. It is not total agent-task time.

This targets provider ID `llama_cpp`, API `openai-completions`, with a loopback base URL. Other providers are left unchanged. It requests `timings_per_token: true` and `return_progress: true` for this local provider. The tested b11888 source attaches timings to OpenAI-compatible stream chunks. No sampling, model, messages, tools, context or thinking settings are changed.

The display updates on stream events, throttled to at most four updates per second, plus lifecycle/final updates. While awaiting initial output it shows `TTFT waiting`; it does not invent an elapsed countdown. No background timers, telemetry, network polling, or transcript logging are used by the extension. Extra stream metadata/rendering has not been benchmarked for overhead; this is not the synchronized GPU profiler.

Each request replaces the previous measurement. Model/session changes clear it. Failed or aborted assistant messages are marked accordingly; final raw usage/timing frames also finish requests whose assistant lifecycle is not exposed, such as some internal calls. Different Pi providers or future API revisions may require revalidation.

## Verification

```powershell
node --test .\outputs\pi-extensions\performance\performance.test.mjs
node .\outputs\pi-extensions\performance\integration-check.mjs
```

Five tests cover role/progress filtering, thinking/tool output, invalid or missing timing data, output-usage accounting, request resets/tool-time exclusion, unchanged payload contents and other-provider filtering, and requests without assistant lifecycle events. The integration check loads the extension through the installed Pi loader and runs a mocked OpenAI SSE stream through the installed provider, validating the final displayed status. See `integration-result.json`.

The initial implementation was verified offline against Pi 1.4.0 and local llama.cpp b11888 serialization source. The later installation check used Pi's actual global extension discovery and loader. No live inference or visual terminal session was run, and no server was started/stopped.

Primary API references: [Pi extension documentation](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/extensions.md), [official status-line example](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/examples/extensions/status-line.ts). The implementation uses the locally installed declarations and provider hooks, and the local pinned server-task.cpp timing serialization.
