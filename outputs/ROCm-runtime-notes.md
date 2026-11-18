# ROCm 10 setup repair for Radeon 860M

The downloaded b11888 Windows ROCm package includes ggml-hip.dll and the core HIP runtime, but its extracted directory did not contain hipblas.dll. dumpbin /DEPENDENTS confirmed ggml-hip.dll imports hipblas.dll. It was also absent from PATH. Only CPU and RPC backends loaded, producing "Available devices: (none)".

The matching AMD runtime was installed into this workspace's isolated Python virtual environment at work/rocm10-runtime. Installed packages: rocm, rocm-sdk-core, rocm-sdk-libraries, rocm-sdk-device-gfx1152, all version 10.0.0. Packages came from https://stable.repo.amd.com/rocm/whl-next/, the same index and core version used by b11888's Windows ROCm build action. The libraries and gfx1152 extras were selected; development tools were not installed.

No display driver or system PATH was changed. The launcher temporarily prepends the isolated core/bin and libraries/bin directories to its process PATH. Pip also uses its ordinary download cache. The existing ROCm binary directory and its bundled runtime files were not modified.

With that temporary search path, device enumeration succeeded:

```text
ROCm0: AMD Radeon (TM) 860M Graphics
gfx1152, VMM: no, Wave Size: 32
44975 MiB total, 44823 MiB free at enumeration
```

The driver reports a different memory budget from Vulkan. This is a backend/runtime-reported budget, not additional physical RAM.

A 1024-input-token / 128-output-token smoke request executed with --device ROCm0, MTP 1, F16 K/V and completed without truncation or a kernel error. Short-context throughput is not evidence of 100K throughput. Matched 100K measurements are saved in the corresponding rocm-100k result directories.

## Launcher

Start-ROCmServer.ps1 uses the installed local libraries and explicitly selects ROCm0 to prevent silent CPU fallback. Its current defaults are F16 K/V, MTP 2, context 163840, one slot, batch 2048, microbatch 512 and eight threads. Optional -DraftTokens 1 or -DraftTokens 3 changes the MTP count. -ListDevices performs device enumeration only. Run one server at a time. Local slot saving is enabled; the save/reload workflow is documented in Cold-prefill-and-reuse.md.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\kasper\Documents\GitHub\local-llm\outputs\Start-ROCmServer.ps1"
```

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\kasper\Documents\GitHub\local-llm\outputs\Start-ROCmServer.ps1" -DraftTokens 2
```

If the agent has already started a background server, do not start another on port 9931. Stop-ROCmServer.ps1 stops an idle matching ROCm server on that port. It checks the executable and launch arguments and refuses to interrupt a busy request.

The launcher restores its inherited process PATH when it exits. The virtual environment must remain in this workspace while this launcher is in use. The repository now contains copies of the ROCm and Vulkan backends under work/backends; launchers use those copies rather than the Downloads folders. No source compilation is required for this repaired prebuilt backend.

References:

- https://github.com/ggml-org/llama.cpp/blob/48c9f08b7/.github/actions/windows-setup-rocm/action.yml
- https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/llamacpp.html
