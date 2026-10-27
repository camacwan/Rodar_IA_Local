"""Detecção de hardware e benchmark de memória/CPU.

Só depende da biblioteca padrão. psutil e numpy são usados se estiverem
instalados (deixam a medição mais precisa), mas não são obrigatórios.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
import time

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

GB = 1024 ** 3

# Banda de memória aproximada (GB/s) — é o que mais pesa na velocidade de LLMs.
NVIDIA_BW = {
    "5090": 1790, "5080": 960, "5070 ti": 896, "5070": 672, "5060 ti": 448, "5060": 448,
    "4090": 1008, "4080": 717, "4070 ti": 504, "4070": 504, "4060 ti": 288, "4060": 272,
    "3090": 936, "3080": 760, "3070": 448, "3060 ti": 448, "3060": 360, "3050": 224,
    "2080": 448, "2070": 448, "2060": 336, "1660": 192, "1650": 128, "1080": 320, "1070": 256,
    "a100": 1555, "h100": 2000, "a6000": 768, "a4000": 448, "t4": 320, "l4": 300, "v100": 900,
}
APPLE_BW = {
    "m1 ultra": 800, "m1 max": 400, "m1 pro": 200, "m1": 68,
    "m2 ultra": 800, "m2 max": 400, "m2 pro": 200, "m2": 100,
    "m3 ultra": 800, "m3 max": 400, "m3 pro": 150, "m3": 100,
    "m4 max": 546, "m4 pro": 273, "m4": 120,
    "m5 max": 600, "m5 pro": 300, "m5": 153,
}


def _run(cmd: list[str], timeout: float = 5) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def _ram() -> tuple[float, float]:
    """(total_gb, disponível_gb)."""
    if psutil:
        vm = psutil.virtual_memory()
        return vm.total / GB, vm.available / GB
    if os.path.exists("/proc/meminfo"):
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.strip().split()[0]) * 1024
        return info["MemTotal"] / GB, info.get("MemAvailable", info["MemFree"]) / GB
    if platform.system() == "Darwin":
        total = int(_run(["sysctl", "-n", "hw.memsize"]) or 0)
        return total / GB, total * 0.6 / GB
    if platform.system() == "Windows":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullTotalPhys / GB, ms.ullAvailPhys / GB
    return 8.0, 4.0


def _cpu_name() -> str:
    name = platform.processor()
    if platform.system() == "Linux" and os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or name
    return name or platform.machine()


def _lookup(table: dict[str, float], name: str) -> float | None:
    n = name.lower()
    for key in sorted(table, key=len, reverse=True):
        if key in n:
            return table[key]
    return None


def _gpus(cpu_name: str, ram_total: float) -> list[dict]:
    gpus: list[dict] = []
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits"])
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                vram = float(parts[1]) / 1024
                gpus.append({
                    "vendor": "NVIDIA", "name": parts[0], "vram_gb": round(vram, 1),
                    "vram_free_gb": round(float(parts[2]) / 1024, 1),
                    "bandwidth_gbs": _lookup(NVIDIA_BW, parts[0]) or max(150, vram * 40),
                    "backend": "CUDA", "unified": False,
                })
    if shutil.which("rocm-smi") and not gpus:
        out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"])
        if out:
            gpus.append({"vendor": "AMD", "name": "GPU AMD (ROCm)", "vram_gb": None,
                         "vram_free_gb": None, "bandwidth_gbs": 400, "backend": "ROCm",
                         "unified": False})
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        bw = _lookup(APPLE_BW, cpu_name) or 100
        usable = ram_total * (0.75 if ram_total > 36 else 0.67)  # limite padrão do Metal
        gpus.append({"vendor": "Apple", "name": cpu_name, "vram_gb": round(usable, 1),
                     "vram_free_gb": round(usable, 1), "bandwidth_gbs": bw,
                     "backend": "Metal", "unified": True})
    return gpus


def detect() -> dict:
    total, avail = _ram()
    cpu = _cpu_name()
    cores_phys = (psutil.cpu_count(logical=False) if psutil else None) or os.cpu_count() or 1
    return {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu": cpu,
        "cores": cores_phys,
        "threads": os.cpu_count() or cores_phys,
        "ram_total_gb": round(total, 1),
        "ram_available_gb": round(avail, 1),
        "gpus": _gpus(cpu, total),
        "disk_free_gb": round(shutil.disk_usage(os.path.expanduser("~")).free / GB, 1),
    }


# ---------------------------------------------------------------- benchmark

def _bandwidth_numpy(seconds: float, threads: int) -> float:
    size = 256 * 1024 * 1024 // 8  # 256 MB por thread em float64
    size //= max(1, threads // 2)
    bufs = [(np.ones(size), np.empty(size)) for _ in range(threads)]
    moved = [0] * threads
    stop = time.perf_counter() + seconds

    def work(i: int) -> None:
        a, b = bufs[i]
        while time.perf_counter() < stop:
            np.copyto(b, a)
            moved[i] += a.nbytes * 2  # leitura + escrita

    t0 = time.perf_counter()
    ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return sum(moved) / (time.perf_counter() - t0) / 1e9


def _bandwidth_stdlib(seconds: float) -> float:
    a = bytearray(128 * 1024 * 1024)
    b = bytearray(len(a))
    mv_a, mv_b = memoryview(a), memoryview(b)
    moved = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        mv_b[:] = mv_a
        moved += len(a) * 2
    # uma thread só não satura o barramento; fator empírico conservador
    return moved / (time.perf_counter() - t0) / 1e9 * 1.6


def _gflops(seconds: float) -> float | None:
    if np is None:
        return None
    n = 1024
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    ops = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a @ b
        ops += 2 * n ** 3
    return ops / (time.perf_counter() - t0) / 1e9


def benchmark(seconds: float = 2.0) -> dict:
    hw = detect()
    threads = min(hw["cores"], 8)
    method = "numpy" if np is not None else "stdlib"
    if np is not None:
        bw = _bandwidth_numpy(seconds, threads)
    else:
        bw = _bandwidth_stdlib(seconds)
    return {
        "cpu_bandwidth_gbs": round(bw, 1),
        "cpu_gflops": round(g, 1) if (g := _gflops(seconds / 2)) else None,
        "threads_used": threads,
        "method": method,
        "measured_at": time.strftime("%Y-%m-%d %H:%M"),
    }
