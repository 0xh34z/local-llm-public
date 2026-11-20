# Automatic file lookup in Pi

> **Note:** This is the public copy of this report. Small summary files (summaries, samples, manifests) are included, with local paths replaced by placeholders such as `<repo>` and `<models>`. The full raw data (request/response dumps, server logs, generated text, build logs and test fixtures) is not public, so some links in this report do not resolve.

Pi answered an ordinary question about a function in the middle of a 603-line Python file. With a 40-line preview and a search suggestion attached to the tool result, it found the function at line 301 and correctly answered that the delay reaches its cap of 37 on attempt 4. The user supplied the filename and question without a line number or slash command. [Tool calls and answer](tool-result.json)

| Behavior | Model requests | Initial input tokens | New input tokens across follow-ups | Output tokens across all requests | Total time in model requests |
|---|---:|---:|---:|---:|---:|
| [Project instructions alone](../pi-automatic-read-verified/requests.jsonl) | 2 | 1,322 | 7,153 | 174 | 30.14 s |
| [40-line preview](../pi-paged-lookup-windows/requests.jsonl) | 4 | 1,346 | 7,280 | 337 | 36.36 s |
| [40-line preview with search suggestion](requests.jsonl) | 5 | 1,344 | 1,084 | 333 | 19.34 s |

All three runs answered correctly. The first read the entire file. The second followed the preview with two larger reads and ultimately sent the entire file in pieces. In the third, a first search returned no matches, a second found the function, and Pi requested 30 surrounding lines. Compared with instructions alone, the third run evaluated 6,069 fewer new input tokens after tool calls and spent 10.80 fewer seconds across model requests. [Whole-file read](../pi-automatic-read-verified/tool-result.json) · [Sequential reads](../pi-paged-lookup-windows/tool-result.json) · [Search and excerpt](tool-result.json)

## Using it

Open Pi in the trusted repository root with project extensions enabled and the local `llama_cpp` provider selected. Use `/reload` for an existing session, then ask normally. The extension supplies a 40-line default for a direct read with no explicit limit. Pi can request larger excerpts as needed. When a partial text result includes a continuation offset, the extension points Pi to an available file search tool. [Project instructions](../pi-extensions/read-range/APPEND_SYSTEM.md) · [Extension](../pi-extensions/read-range/local-read.js)

## Measurement scope

Each row is one fresh local server and Pi session on the Radeon 860M. Pi 1.5.2 used the read and grep tools, thinking off and a deterministic synthetic file. The generator used F16 K/V, MTP 2, batch 2048, microbatch 512, eight threads, one slot and the eight-block launcher configuration. The allocation was 163,840 tokens. The largest actual prompt in each run reached 8,499, 8,697 and 2,496 tokens respectively. This comparison covers one lookup at short context. The earlier 103K result tested an explicitly requested range. [Exact launch](launch.json) · [Provider guidance audit](guidance.json) · [Earlier 103K measurements](../pi-template-103k-windows/requests.jsonl)

Time is the sum of intervals from each provider request to its assistant message end. Startup and time between requests are excluded. Different tool choices and response lengths affect the result. The observed savings apply to this task. [Meter](../pi-prefix-meter.mjs) · [Runner](../run_pi_prefix.py)

### Windows default tools

With Pi's default read, bash, edit and write tools, the extension also worked using shell search. Pi previewed the file, searched twice, read the function around line 301 and answered correctly. The follow-ups evaluated 840 new tokens across the lookup, and all model requests together took 20.97 seconds. The first shell search found no match and returned exit code 1. The initial benchmark verifier flagged that intermediate result after the answer completed. Inspection of the saved events confirmed that Pi recovered, saw the implementation and received only 57 of the 600 unrelated constants. The runner now accepts a recovered search miss and checks the successful excerpts. This was one run. [Tool calls and answer](../pi-default-tools-lookup/tool-result.json) · [Timings](../pi-default-tools-lookup/requests.jsonl)
