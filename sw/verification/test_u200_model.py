"""Deployment integrity and exact-stream checks without an FPGA."""
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from sw.deploy.u200_model import FRAME_BYTES, compare_rgb, pack, self_test, sha256, verify_bundle
from sw.verification.export_weights import export, NAMES, shapes


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fiaccel_deployment_test_")
        cls.root = Path(cls.temporary.name)
        cls.model = cls.root / "weights"
        document = dict(depths=[8, 16, 32, 64, 32, 16, 8], output_channels=3, layers={
            name: dict(shape=list(shape), values=[0] * math.prod(shape), shift=8)
            for name, shape in zip(NAMES, shapes([8, 16, 32, 64, 32, 16, 8]))})
        export(document, cls.model)
        (cls.model / "quantized_model.json").write_text(json.dumps(document))
        cls.fixture = cls.root / "fixture"
        cls.fixture.mkdir()
        # Sparse files suffice for bundle integrity; no inference is claimed.
        for name, size in (("input.rgb", 2 * FRAME_BYTES), ("expected.rgb", 4 * FRAME_BYTES)):
            with (cls.fixture / name).open("wb") as stream:
                stream.truncate(size)
        data = dict(format="fiaccel-rgb24-fixture-v1", width=1920, height=1080,
            input_frames=2, output_frames=4, interpolated_frames=1,
            config_sha256=sha256(cls.model / "config.hex"),
            model_sha256=sha256(cls.model / "quantized_model.json"),
            sha256={name: sha256(cls.fixture / name) for name in ("input.rgb", "expected.rgb")})
        (cls.fixture / "manifest.json").write_text(json.dumps(data))
        # CPU fixture emits a known stream; it does not emulate the CNN or XRT.
        cls.host = cls.root / "host"
        cls.host.write_text("#!/usr/bin/env python3\nimport os,sys\n"
            "from pathlib import Path\n"
            "sys.stdin.buffer.read()\n"
            "data=bytearray((Path(__file__).parent/'fixture/expected.rgb').read_bytes())\n"
            "if os.environ.get('FIACCEL_TEST_CORRUPT'): data[42]=1\n"
            "sys.stdout.buffer.write(data)\n"
            "print('VIDEO_SUMMARY input_frames=2 output_frames=4 interpolated_frames=1 width=1920 height=1080 max_kernel_cycles=100 over_budget_frames=0',file=sys.stderr)\n")
        cls.host.chmod(0o755)
        cls.bundle = cls.root / "bundle"
        pack(cls.model, cls.host, cls.fixture, cls.bundle)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_bundle_is_portable_and_detects_tampering(self):
        result = subprocess.run(["python3", str(self.bundle / "u200_model.py"), "check", str(self.bundle)],
            cwd="/tmp", capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["result"], "PASS")
        target = self.bundle / "weights/config.hex"
        original = target.read_bytes()
        try:
            target.write_bytes(b"000000000000\n" + original)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_bundle(self.bundle)
        finally:
            target.write_bytes(original)

    def test_rejects_mismatched_weights_and_fixture(self):
        target = self.model / "config.hex"
        original = target.read_bytes()
        try:
            target.write_bytes(original.replace(b"800000000008", b"800000000007", 1))
            with self.assertRaisesRegex(ValueError, "does not match"):
                pack(self.model, self.host, self.fixture, self.root / "bad")
        finally:
            target.write_bytes(original)
        path = self.fixture / "manifest.json"
        original = path.read_text()
        try:
            manifest = json.loads(original)
            manifest["config_sha256"] = "0" * 64
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "different model"):
                pack(self.model, self.host, self.fixture, self.root / "bad")
        finally:
            path.write_text(original)

    def test_stream_comparison_rejects_pixel_changes_and_truncation(self):
        actual, expected = self.root / "actual", self.root / "expected"
        expected.write_bytes(bytes([0, 10, 20, 30, 40]))
        actual.write_bytes(bytes([0, 10, 25, 30, 40]))
        result = compare_rgb(actual, expected)
        self.assertFalse(result["exact"])
        self.assertEqual((result["differing_rgb_bytes"], result["max_rgb_error"], result["first_difference_byte"]), (1, 5, 2))
        actual.write_bytes(expected.read_bytes()[:-1])
        self.assertFalse(compare_rgb(actual, expected)["exact"])

    def test_completion_requires_exact_output_and_writes_failure_evidence(self):
        xclbin = self.root / "unused.xclbin"
        xclbin.write_bytes(b"CPU fixture, not hardware")
        with patch("sw.deploy.u200_model.require_idle_device"):
            passed = self_test(self.bundle, xclbin, self.root / "pass")
            self.assertEqual(passed["result"], "PASS")
            self.assertTrue((self.root / "pass/midpoint.ppm").is_file())
            with patch.dict("os.environ", {"FIACCEL_TEST_CORRUPT": "1"}):
                failed = self_test(self.bundle, xclbin, self.root / "fail")
            self.assertEqual(failed["result"], "FAIL")
            self.assertEqual(failed["comparison"]["differing_rgb_bytes"], 1)
            self.assertTrue((self.root / "fail/validation.json").is_file())


if __name__ == "__main__":
    unittest.main()
