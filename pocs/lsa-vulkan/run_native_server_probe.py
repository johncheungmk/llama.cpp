"""One bounded localhost-only smoke test of the experimental native server."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

from capture_flash_next_cpu import enforce_limits


def request(port, route, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=data,
                                 headers={"Content-Type": "application/json"})
    # Explicitly bypass machine/user HTTP proxies for this localhost probe.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-layers", type=int, choices=(4,), default=4)
    parser.add_argument("--port", type=int, choices=(18080,), default=18080)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    stdout = args.output.with_suffix(".server.stdout.log")
    stderr = args.output.with_suffix(".server.stderr.log")
    if stdout.exists() or stderr.exists():
        parser.error("server log exists")
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    limits = enforce_limits(40)
    env = dict(os.environ, LLAMA_QSA_PCA_MAP=str(args.map.resolve(strict=True)), LLAMA_QSA_PCA_EXPERIMENTAL="1")
    command = [str(args.runtime / "llama-server.exe"), "-m", str(args.model),
               "--host", "127.0.0.1", "--port", str(args.port), "--alias", "qwen-native-pca-test",
               "--device", "Vulkan0", "-ngl", "4", "-c", "512", "-np", "1", "-b", "256", "-ub", "256",
               "--threads", "12", "--threads-batch", "12", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on",
               "-lm", "mmap", "--lazy-mode", "on", "--no-repack", "--fit", "off", "--ctx-checkpoints", "0",
               "--cache-ram", "0", "--spec-type", "none", "--no-warmup", "--jinja"]
    result = {"schema": 1, "kind": "native_pca_localhost_server_smoke", "gpu_layers": 4,
              "context_tokens": 512, "slots": 1, "attention_kv_types": "Q8_0/Q8_0",
              "map_sha256": hashlib.sha256(args.map.read_bytes()).hexdigest(),
              "runtime_llama_sha256": hashlib.sha256((args.runtime / "llama.dll").read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "production_ready": False}
    process = None
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            process = subprocess.Popen(command, env=env, cwd=args.runtime, stdout=out, stderr=err,
                                       creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            kernel = limits[0]
            kernel.OpenProcess.restype, kernel.OpenProcess.argtypes = C.c_void_p, [C.c_uint32, C.c_int, C.c_uint32]
            kernel.CloseHandle.restype, kernel.CloseHandle.argtypes = C.c_int, [C.c_void_p]
            handle = kernel.OpenProcess(0x100 | 0x200 | 0x400, 0, process.pid)
            if not handle:
                raise C.WinError(C.get_last_error())
            try:
                if not kernel.SetProcessWorkingSetSizeEx(handle, 16 * 1024**2, 8 * 1024**3, 6):
                    raise C.WinError(C.get_last_error())
            finally:
                kernel.CloseHandle(handle)
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"isolated server exited during startup: {process.returncode}; see private logs")
                try:
                    if request(args.port, "/health").get("status") == "ok":
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("isolated server startup exceeded 600 seconds")
                time.sleep(2)
            if "EXPERIMENTAL QSA PCA rank" not in stderr.read_text(encoding="utf-8", errors="replace"):
                raise RuntimeError("server did not confirm the experimental map")
            start = time.perf_counter()
            response = request(args.port, "/v1/chat/completions", {
                "model": "qwen-native-pca-test", "messages": [{"role": "user", "content": "Return the number after two: 1, 2,"}],
                "max_tokens": 8, "temperature": 0, "seed": 42,
                "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
            result.update(elapsed_seconds=time.perf_counter() - start, usage=response.get("usage"),
                          timings=response.get("timings"), choices=response.get("choices"), smoke_pass=True)
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            result["isolated_server_stopped"] = process.poll() is not None
    with socket.socket() as check:
        result["localhost_port_closed"] = check.connect_ex(("127.0.0.1", args.port)) != 0
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
