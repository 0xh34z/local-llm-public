# Agent setup guide

Goal: get a local llama.cpp server running on Windows with an AMD GPU (ROCm 10.0 wheels, tested on gfx1152 / Radeon 860M). Run everything from the repository root in PowerShell. Python 3.11+ must be on `PATH`.

## Steps

1. **Model.** `.\llm.ps1 -DownloadModel` downloads the recommended ~21 GB model into `models\` (resumable; re-run if interrupted). Any other `.gguf` placed in `models\` also works. If `models\` holds several, set `$ModelName` at the top of `llm.ps1`.
2. **Backend.** `.\llm.ps1 -Update` does all of this:
   - creates `work\rocm10-runtime` (isolated ROCm runtime, large pip download) if missing;
   - downloads the newest official llama.cpp Windows ROCm release into `work\backends\`;
   - smoke-tests it with the model (starts a server on port 9939, waits up to 5 min for `/health`);
   - only on success writes the folder name to `work\backends\current.txt`.
   - Other GPU: pass the device package via `python outputs\update_backend.py --gfx gfxNNNN --model models\<file>.gguf`.
   - Pin or roll back: `.\llm.ps1 -Update -Tag b11888`.
3. **Run.** `.\llm.ps1` starts the server on port 9931.

## Success check

`Invoke-RestMethod http://127.0.0.1:9931/health` returns status ok. The first model load takes a minute or more.

## Notes

- MTP speculative decoding is enabled automatically only when the model file name contains `MTP`. Override with `outputs\Start-ROCmServer.ps1 -Speculative off|mtp`.
- If the smoke test fails, read `work\update-smoke-test.log`. The previous backend stays active.
- `work\`, `models\*` and `*.gguf` are gitignored. Do not commit them.
- The `outputs\` folder is benchmark evidence from one machine. The patched-DLL build in `BUILD.md` step 3 and everything under `Start-KVSplitServer.ps1` are optional and need Visual Studio Build Tools, CMake, Ninja and a HIP SDK. Skip them unless asked to reproduce the reports.
- Do not run `llama-server` with a context larger than the machine's memory allows; the default 163840 tokens with F16 KV needs about 24 GB of unified memory at 103K populated context. Lower it with `outputs\Start-ROCmServer.ps1 -Context 32768`.
