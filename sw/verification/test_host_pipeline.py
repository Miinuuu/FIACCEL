"""Exercise bounded host scheduling without XRT or an FPGA."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class HostPipelineTests(unittest.TestCase):
    def run_cpp(self, source, needs_config_path=False):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "pipeline_test"
            subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT / "sw/host"), str(ROOT / "sw/host/frame_io.cpp"),
                str(ROOT / "sw/verification" / source), "-o", str(binary)], check=True)
            arguments = [str(Path(directory) / "config.hex")] if needs_config_path else []
            subprocess.run([str(binary), *arguments], check=True, timeout=30)

    def test_ordered_overlap_and_eof(self):
        self.run_cpp("host_pipeline_test.cpp")

    def test_rgb_and_coefficient_validation(self):
        self.run_cpp("frame_io_test.cpp", needs_config_path=True)


if __name__ == "__main__":
    unittest.main()
