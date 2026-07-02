from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from chef_human.agent.hardware import (
    HardwareInfo,
    detect_hardware,
    detect_ram_gb,
    detect_vram_gb,
)


class TestHardwareInfoCapacity:
    def test_uses_larger_of_ram_and_vram(self):
        info = HardwareInfo(ram_gb=16.0, vram_gb=24.0)
        assert info.capacity_gb() == 24.0

    def test_ram_only(self):
        info = HardwareInfo(ram_gb=16.0, vram_gb=None)
        assert info.capacity_gb() == 16.0

    def test_vram_only(self):
        info = HardwareInfo(ram_gb=None, vram_gb=24.0)
        assert info.capacity_gb() == 24.0

    def test_neither_detected(self):
        info = HardwareInfo(ram_gb=None, vram_gb=None)
        assert info.capacity_gb() is None

    def test_ram_larger_than_vram(self):
        info = HardwareInfo(ram_gb=64.0, vram_gb=8.0)
        assert info.capacity_gb() == 64.0


class TestDetectRamGb:
    def test_uses_sysconf_when_available(self):
        with patch("os.sysconf", side_effect=lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 8_000_000}[name]):
            result = detect_ram_gb()
        assert result == (4096 * 8_000_000) / (1024**3)

    def test_returns_none_when_sysconf_and_ctypes_both_fail(self):
        # On non-Windows platforms `ctypes.windll` doesn't exist, so the
        # Windows fallback naturally raises and is swallowed -- no need to
        # mock it separately.
        with patch("os.sysconf", side_effect=ValueError("not supported")):
            result = detect_ram_gb()
        assert result is None


class TestDetectVramGb:
    def test_returns_none_when_nvidia_smi_missing(self):
        with patch("shutil.which", return_value=None):
            assert detect_vram_gb() is None

    def test_parses_nvidia_smi_output(self):
        fake_result = MagicMock()
        fake_result.stdout = "24576\n"
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=fake_result):
                result = detect_vram_gb()
        assert result == 24576 / 1024

    def test_uses_largest_gpu_when_multiple(self):
        fake_result = MagicMock()
        fake_result.stdout = "8192\n24576\n"
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=fake_result):
                result = detect_vram_gb()
        assert result == 24576 / 1024

    def test_returns_none_on_subprocess_error(self):
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", side_effect=subprocess.SubprocessError):
                assert detect_vram_gb() is None

    def test_returns_none_on_empty_output(self):
        fake_result = MagicMock()
        fake_result.stdout = ""
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=fake_result):
                assert detect_vram_gb() is None

    def test_ignores_unparseable_lines(self):
        fake_result = MagicMock()
        fake_result.stdout = "not-a-number\n16384\n"
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=fake_result):
                result = detect_vram_gb()
        assert result == 16384 / 1024


class TestDetectHardware:
    def test_combines_ram_and_vram(self):
        with patch("chef_human.agent.hardware.detect_ram_gb", return_value=16.0):
            with patch("chef_human.agent.hardware.detect_vram_gb", return_value=8.0):
                info = detect_hardware()
        assert info.ram_gb == 16.0
        assert info.vram_gb == 8.0
