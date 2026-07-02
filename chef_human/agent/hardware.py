from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareInfo:
    ram_gb: float | None
    vram_gb: float | None

    def capacity_gb(self) -> float | None:
        """Best-effort estimate of how large a model this machine can
        comfortably run: the larger of GPU VRAM (a model fully offloaded to
        GPU only needs to fit there) and system RAM (the CPU/Ollama
        fallback path). None if neither could be detected."""
        candidates = [v for v in (self.ram_gb, self.vram_gb) if v is not None]
        return max(candidates) if candidates else None


def detect_ram_gb() -> float | None:
    """Total physical RAM in GiB, or None if it couldn't be determined."""
    try:
        import os

        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3)
    except (ValueError, OSError, AttributeError):
        pass

    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return stat.ullTotalPhys / (1024**3)
    except Exception:
        return None


def detect_vram_gb() -> float | None:
    """Total VRAM in GiB of the largest detected NVIDIA GPU, or None if
    nvidia-smi isn't available (e.g. no NVIDIA GPU, or AMD/Apple Silicon --
    not currently detected)."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        output = subprocess.run(
            [nvidia_smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None

    values: list[float] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    if not values:
        return None
    return max(values) / 1024  # MiB -> GiB


def detect_hardware() -> HardwareInfo:
    return HardwareInfo(ram_gb=detect_ram_gb(), vram_gb=detect_vram_gb())
