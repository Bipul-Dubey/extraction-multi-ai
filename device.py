"""
device.py — detect the machine and return a settings profile for the FIR preprocessing notebook.

Profiles:
    mac_apple_silicon : MacBook (Apple Silicon). CPU + OpenCV OpenCL (T-API) if it passes a self-test.
    nvidia_cuda       : NVIDIA GB10 AI server. OpenCV CUDA (only if OpenCV was built with CUDA) + optional CuPy.
    cpu_generic       : anything else, CPU only.

Overrides (highest priority last):
    environment  FIR_DEVICE=cpu|mac|cuda   force a profile
                 FIR_WORKERS=<n>           force number of parallel page workers
    argument     get_device_profile(overrides={...})  any profile key

Usage:
    from device import get_device_profile, apply_profile, print_profile
    PROFILE = get_device_profile()
    apply_profile(PROFILE)
    print_profile(PROFILE)
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess

import cv2
import numpy as np

# Rough peak memory for processing one ~1700x2800 colour page through the whole pipeline
# (several uint8/float32 copies + intermediates). Re-measure once all steps exist.
MEM_PER_WORKER_GB = 1.5
# Use at most this fraction of currently available memory for page workers.
MEMORY_BUDGET_FRACTION = 0.6
# OpenCV operations whose CPU/GPU backend is chosen per profile.
OPS = ("denoise", "filter", "warp", "clahe")
# Max mean absolute pixel difference allowed between a GPU path and the CPU reference.
GPU_TOLERANCE = 1.0

_PROFILE_ALIASES = {
    "cpu": "cpu_generic",
    "mac": "mac_apple_silicon",
    "cuda": "nvidia_cuda",
}


# ---------------------------------------------------------------------------
# Low-level detection helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    """Run a command and return stdout, or None if it fails or is missing."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _sysctl_int(name: str) -> int | None:
    out = _run(["sysctl", "-n", name])
    try:
        return int(out) if out else None
    except ValueError:
        return None


def _cpu_info() -> dict:
    system = platform.system()
    total = os.cpu_count() or 1
    perf = None
    efficiency = None
    name = platform.processor() or platform.machine()
    if system == "Darwin":
        perf = _sysctl_int("hw.perflevel0.physicalcpu")
        efficiency = _sysctl_int("hw.perflevel1.physicalcpu")
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or name
    elif system == "Linux":
        try:
            total = len(os.sched_getaffinity(0))
        except AttributeError:
            pass
        model = _run(["lscpu"])
        if model:
            for line in model.splitlines():
                if line.startswith("Model name:"):
                    name = line.split(":", 1)[1].strip()
                    break
    return {
        "name": name,
        "total_cores": total,
        "performance_cores": perf,
        "efficiency_cores": efficiency,
    }


def _memory_info() -> dict:
    """Total and currently available memory in GB (unified memory on both target machines)."""
    system = platform.system()
    total_gb = available_gb = None
    if system == "Darwin":
        total = _sysctl_int("hw.memsize")
        total_gb = total / 1024**3 if total else None
        vm = _run(["vm_stat"])
        if vm:
            page_size = 16384
            pages = {}
            for line in vm.splitlines():
                if "page size of" in line:
                    try:
                        page_size = int(line.split("page size of")[1].split()[0])
                    except (IndexError, ValueError):
                        pass
                elif ":" in line:
                    key, val = line.split(":", 1)
                    try:
                        pages[key.strip()] = int(val.strip().rstrip("."))
                    except ValueError:
                        pass
            free_pages = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
            available_gb = free_pages * page_size / 1024**3
    elif system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                info = {line.split(":")[0]: int(line.split()[1]) for line in f if ":" in line}
            total_gb = info.get("MemTotal", 0) / 1024**2
            available_gb = info.get("MemAvailable", 0) / 1024**2
        except (OSError, ValueError, IndexError):
            pass
    if available_gb is None and total_gb is not None:
        available_gb = total_gb * 0.5
    return {
        "total_gb": round(total_gb, 1) if total_gb else None,
        "available_gb": round(available_gb, 1) if available_gb else None,
    }


def _nvidia_info() -> list[dict]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits"])
    gpus = []
    if out:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "name": parts[0],
                    # GB10 reports unified memory as "[N/A]"
                    "memory_mb": int(parts[1]) if parts[1].isdigit() else None,
                    "compute_cap": parts[2],
                    "driver": parts[3],
                })
    return gpus


def _opencv_cuda_info() -> dict:
    """Which CUDA functions this OpenCV build actually provides."""
    count = 0
    if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
        except cv2.error:
            count = 0
    ops = {
        "denoise": count > 0 and hasattr(cv2.cuda, "fastNlMeansDenoising"),
        "filter": count > 0 and hasattr(cv2.cuda, "createGaussianFilter") and hasattr(cv2.cuda, "createMorphologyFilter"),
        "warp": count > 0 and hasattr(cv2.cuda, "warpAffine"),
        "clahe": count > 0 and hasattr(cv2.cuda, "createCLAHE"),
    }
    return {"device_count": count, "ops": ops}


def _test_image() -> np.ndarray:
    """Deterministic synthetic 'page': light paper, dark strokes, noise."""
    rng = np.random.default_rng(0)
    img = np.full((512, 512), 235, np.uint8)
    for y in range(40, 480, 24):
        cv2.line(img, (30, y), (480, y + 3), 40, 2)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _opencl_selftest() -> dict:
    """Run a few T-API ops on OpenCL and compare to CPU. Returns per-op pass/fail."""
    result = {"available": False, "device": None, "ops": {op: False for op in OPS}}
    if not cv2.ocl.haveOpenCL():
        return result
    prev = cv2.ocl.useOpenCL()
    try:
        cv2.ocl.setUseOpenCL(True)
        if not cv2.ocl.useOpenCL():
            return result
        result["available"] = True
        try:
            result["device"] = cv2.ocl.Device.getDefault().name()
        except Exception:  # noqa: BLE001 — device query is informational only
            pass

        img = _test_image()
        u = cv2.UMat(img)
        M = cv2.getRotationMatrix2D((256, 256), 3.0, 1.0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        tests = {
            "filter": lambda x: cv2.morphologyEx(cv2.GaussianBlur(x, (0, 0), 2.0), cv2.MORPH_CLOSE, kernel),
            "warp": lambda x: cv2.warpAffine(x, M, (512, 512), flags=cv2.INTER_CUBIC, borderValue=255),
            "clahe": lambda x: clahe.apply(x),
            "denoise": lambda x: cv2.fastNlMeansDenoising(x, None, 7, 7, 21),
        }
        for op, fn in tests.items():
            try:
                cpu_out = fn(img) if op != "clahe" else cv2.createCLAHE(2.0, (8, 8)).apply(img)
                gpu_out = fn(u)
                gpu_out = gpu_out.get() if isinstance(gpu_out, cv2.UMat) else gpu_out
                diff = float(np.mean(np.abs(cpu_out.astype(np.int16) - gpu_out.astype(np.int16))))
                result["ops"][op] = diff <= GPU_TOLERANCE
            except cv2.error:
                result["ops"][op] = False
    finally:
        cv2.ocl.setUseOpenCL(prev)
    return result


def _cupy_info() -> dict:
    try:
        import cupy as cp  # type: ignore

        a = cp.arange(1024, dtype=cp.float32)
        ok = float(cp.sum(a).get()) == float(np.arange(1024, dtype=np.float32).sum())
        return {"available": ok, "version": cp.__version__}
    except Exception:  # noqa: BLE001 — any import/runtime failure means "not usable"
        return {"available": False, "version": None}


# ---------------------------------------------------------------------------
# Profile selection
# ---------------------------------------------------------------------------

def _detect_profile_name(system: str, machine: str, nvidia: list, cv_cuda: dict) -> str:
    if nvidia or cv_cuda["device_count"] > 0:
        return "nvidia_cuda"
    if system == "Darwin" and machine == "arm64":
        return "mac_apple_silicon"
    return "cpu_generic"


def _worker_count(profile_name: str, cpu: dict, mem: dict) -> tuple[int, str]:
    total = cpu["total_cores"]
    if profile_name == "mac_apple_silicon":
        perf = cpu["performance_cores"] or max(1, total // 2)
        by_cpu = max(2, perf - 1)
        rule = f"performance cores ({perf}) - 1, min 2"
    elif profile_name == "nvidia_cuda":
        by_cpu = max(1, min(16, total - 4))
        rule = f"total cores ({total}) - 4, max 16"
    else:
        by_cpu = max(1, total - 1)
        rule = f"total cores ({total}) - 1"

    available = mem["available_gb"] or 4.0
    by_mem = max(1, math.floor(available * MEMORY_BUDGET_FRACTION / MEM_PER_WORKER_GB))
    if by_mem < by_cpu:
        return by_mem, f"{rule}; limited by memory ({available:.1f} GB available x {MEMORY_BUDGET_FRACTION} / {MEM_PER_WORKER_GB} GB)"
    return by_cpu, rule


def get_device_profile(overrides: dict | None = None) -> dict:
    """Detect the machine and return the settings profile (a plain, JSON-serialisable dict)."""
    system, machine = platform.system(), platform.machine()
    cpu = _cpu_info()
    mem = _memory_info()
    nvidia = _nvidia_info()
    cv_cuda = _opencv_cuda_info()

    detected = _detect_profile_name(system, machine, nvidia, cv_cuda)
    forced = os.environ.get("FIR_DEVICE", "").strip().lower()
    name = _PROFILE_ALIASES.get(forced, detected) if forced else detected

    notes: list[str] = []
    if forced:
        notes.append(f"profile forced by FIR_DEVICE={forced} (detected: {detected})")

    backends = {op: "cpu" for op in OPS}
    opencl = {"available": False, "device": None, "ops": {op: False for op in OPS}}
    cupy = {"available": False, "version": None}

    if name == "mac_apple_silicon":
        opencl = _opencl_selftest()
        for op in OPS:
            if opencl["ops"][op]:
                backends[op] = "opencl"
        failed = [op for op in OPS if not opencl["ops"][op]]
        if not opencl["available"]:
            notes.append("OpenCL not usable: all operations on CPU")
        elif failed:
            notes.append(f"OpenCL self-test failed for {failed}: those run on CPU")
    elif name == "nvidia_cuda":
        for op in OPS:
            if cv_cuda["ops"][op]:
                backends[op] = "cuda"
        if cv_cuda["device_count"] == 0:
            notes.append("NVIDIA GPU found but this OpenCV build has no CUDA: OpenCV ops run on CPU "
                         "(use NVIDIA's OpenCV container or build OpenCV with CUDA)")
        cupy = _cupy_info()
        if not cupy["available"]:
            notes.append("CuPy not installed/usable: array maths on NumPy (CPU)")

    workers, worker_rule = _worker_count(name, cpu, mem)
    if os.environ.get("FIR_WORKERS", "").strip().isdigit():
        workers = max(1, int(os.environ["FIR_WORKERS"]))
        worker_rule = f"forced by FIR_WORKERS={workers}"

    total = cpu["total_cores"]
    profile = {
        "name": name,
        "detected": detected,
        "system": f"{system} {platform.release()} ({machine})",
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cpu": cpu,
        "memory": mem,
        "nvidia_gpus": nvidia,
        "opencv_cuda": cv_cuda,
        "opencl": opencl,
        "cupy": cupy,
        # --- settings used by the notebook ---
        "backends": backends,
        "array_backend": "cupy" if cupy["available"] else "numpy",
        "num_workers": workers,
        "num_workers_rule": worker_rule,
        # OpenCV's own threads inside each worker, so workers x threads ~= cores
        "cv_threads_per_worker": max(1, total // max(1, workers)),
        "mem_per_worker_gb": MEM_PER_WORKER_GB,
        "max_render_scale": 1.5 if name == "nvidia_cuda" else 1.0,
        "save_intermediates": name == "nvidia_cuda",
        "preview_max_px": 1200 if name == "cpu_generic" else 1400,
        "gpu_tolerance": GPU_TOLERANCE,
        "notes": notes,
    }

    for key, value in (overrides or {}).items():
        if key not in profile:
            raise KeyError(f"Unknown device profile key: {key!r}")
        profile[key] = value
        profile["notes"].append(f"override: {key}={value!r}")
    return profile


def apply_profile(profile: dict, in_worker: bool = False) -> None:
    """Apply process-wide OpenCV settings. Call once in the notebook and once in each page worker."""
    cv2.setNumThreads(profile["cv_threads_per_worker"] if in_worker else 0)
    cv2.ocl.setUseOpenCL("opencl" in profile["backends"].values())
    if profile["opencv_cuda"]["device_count"] > 0:
        cv2.cuda.setDevice(0)


def print_profile(profile: dict) -> None:
    cpu, mem = profile["cpu"], profile["memory"]
    cores = f"{cpu['total_cores']} cores"
    if cpu["performance_cores"]:
        cores += f" ({cpu['performance_cores']}P + {cpu['efficiency_cores']}E)"
    lines = [
        f"Device profile : {profile['name']}",
        f"System         : {profile['system']} | Python {profile['python']} | OpenCV {profile['opencv']} | NumPy {profile['numpy']}",
        f"CPU            : {cpu['name']} | {cores}",
        f"Memory         : {mem['total_gb']} GB total | {mem['available_gb']} GB available now",
    ]
    for gpu in profile["nvidia_gpus"]:
        lines.append(f"NVIDIA GPU     : {gpu['name']} | compute {gpu['compute_cap']} | driver {gpu['driver']}")
    if profile["opencl"]["available"]:
        lines.append(f"OpenCL         : {profile['opencl']['device']}")
    lines += [
        "Backends       : " + " | ".join(f"{op}: {b}" for op, b in profile["backends"].items())
        + f" | arrays: {profile['array_backend']}",
        f"Workers        : {profile['num_workers']} parallel pages ({profile['num_workers_rule']}), "
        f"{profile['cv_threads_per_worker']} OpenCV threads each",
        f"Render         : max upscale {profile['max_render_scale']}x | save intermediates: {profile['save_intermediates']}",
    ]
    lines += [f"Note           : {n}" for n in profile["notes"]]
    print("\n".join(lines))


if __name__ == "__main__":
    print_profile(get_device_profile())
