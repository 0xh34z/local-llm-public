"""Build or measure a gfx1152 device-memory streaming read using the isolated Windows SDK."""
import argparse
from pathlib import Path
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "measure"))
    parser.add_argument("result", nargs="?", type=Path, help="new JSONL file for a measurement")
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    work = tools.parent / "work"
    sdk = work / "rocm10-build-tools/Lib/site-packages"
    core = sdk / "_rocm_sdk_core"
    devel = sdk / "_rocm_sdk_devel"
    build = work / "hip-stream-bandwidth"
    build.mkdir(exist_ok=True)
    exe = build / "stream_bandwidth.exe"
    env = dict(os.environ)
    if args.action == "build":
        vs = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
        settings = subprocess.check_output(f'cmd.exe /d /c "call "{vs}" >nul && set"', text=True)
        env = dict(line.split("=", 1) for line in settings.splitlines() if "=" in line and not line.startswith("="))
    original_path = next(v for k, v in env.items() if k.upper() == "PATH")
    env = {k: v for k, v in env.items() if k.upper() != "PATH"}
    env["PATH"] = os.pathsep.join((str(core / "bin"), str(sdk / "_rocm_sdk_libraries/bin"), original_path))
    if args.action == "build":
        flags = [devel / "lib/llvm/bin/clang++.exe", f"--hip-path={devel}",
                 f"--rocm-device-lib-path={devel / 'lib/llvm/amdgcn/bitcode'}", "--offload-arch=gfx1152",
                 "-O3", "-std=c++17", tools / "stream_bandwidth.hip", core / "lib/amdhip64.lib", "-o", exe]
        subprocess.run(flags, env=env, cwd=build, check=True)
        print(exe)
    else:
        if not args.result:
            parser.error("measure requires a new result path")
        args.result.parent.mkdir(parents=True, exist_ok=True)
        with args.result.open("xb") as output:
            subprocess.run([exe], env=env, cwd=build, stdout=output, check=True)
        print(args.result)


if __name__ == "__main__":
    main()
