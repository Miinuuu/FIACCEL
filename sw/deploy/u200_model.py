#!/usr/bin/env python3
"""Package, check and run a trained FIACCEL model without a training environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "fiaccel-u200-model-v1"
FRAME_BYTES = 1920 * 1080 * 3


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def fixture_metadata(directory):
    data = json.loads((directory / "manifest.json").read_text())
    count = data["input_frames"]
    if (data["format"] != "fiaccel-rgb24-fixture-v1" or
            (data["width"], data["height"]) != (1920, 1080) or
            type(count) is not int or count < 2 or data["output_frames"] != 2 * count or
            data["interpolated_frames"] != count - 1):
        raise ValueError("Fixture must describe at least two Full-HD RGB24 input frames")
    for name, size in (("input.rgb", count * FRAME_BYTES), ("expected.rgb", 2 * count * FRAME_BYTES)):
        path = directory / name
        if path.stat().st_size != size or sha256(path) != data["sha256"][name]:
            raise ValueError(f"Invalid fixture bytes: {name}")
    return data


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "bundle.json").read_text())
    if manifest["format"] != FORMAT:
        raise ValueError("Unsupported U200 model bundle")
    required = {"host", "u200_model.py", "weights/config.hex", "weights/manifest.json",
                "weights/quantized_model.json", "fixture/manifest.json", "fixture/input.rgb",
                "fixture/expected.rgb"}
    if not required.issubset(manifest["sha256"]):
        raise ValueError("Incomplete model bundle")
    for name, digest in manifest["sha256"].items():
        relative = Path(name)
        path = directory / relative
        if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(directory):
            raise ValueError("Invalid bundle member path")
        if sha256(path) != digest:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    fixture = fixture_metadata(directory / "fixture")
    if (fixture["config_sha256"] != sha256(directory / "weights/config.hex") or
            fixture["model_sha256"] != sha256(directory / "weights/quantized_model.json")):
        raise ValueError("Reference fixture belongs to a different model")
    if not os.access(directory / "host", os.X_OK):
        raise ValueError("The packaged host must be executable")
    return manifest


def pack(model, host, fixture, output, ffmpeg=None, ffprobe=None):
    model, host, fixture, output = map(lambda p: Path(p).resolve(), (model, host, fixture, output))
    if output.exists():
        raise ValueError("Choose a new bundle directory")
    sys.path.insert(0, str(ROOT))
    from sw.verification.export_weights import export
    document = json.loads((model / "quantized_model.json").read_text())
    if document.get("depths") != [8, 16, 32, 64, 32, 16, 8]:
        raise ValueError("The U200 kernel requires the full production channel counts")
    fixture_info = fixture_metadata(fixture)
    with tempfile.TemporaryDirectory(prefix="fiaccel_model_export_") as temporary:
        count = export(document, Path(temporary))
        if (Path(temporary) / "config.hex").read_bytes() != (model / "config.hex").read_bytes():
            raise ValueError("config.hex does not match the exported integer model")
    if (fixture_info["config_sha256"] != sha256(model / "config.hex") or
            fixture_info["model_sha256"] != sha256(model / "quantized_model.json")):
        raise ValueError("Reference fixture belongs to a different model")
    if not host.is_file() or not os.access(host, os.X_OK):
        raise ValueError("Provide a compiled, executable XRT host")
    if bool(ffmpeg) != bool(ffprobe):
        raise ValueError("Provide both FFmpeg and FFprobe")
    for binary in (ffmpeg, ffprobe):
        if binary and (not Path(binary).is_file() or not os.access(binary, os.X_OK)):
            raise ValueError(f"Missing executable video tool: {binary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fiaccel_bundle_", dir=output.parent) as temporary:
        stage = Path(temporary) / "model"
        stage.mkdir()
        def copy(source, name):
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        copy(host, "host")
        copy(Path(__file__), "u200_model.py")
        for name in ("config.hex", "manifest.json", "quantized_model.json"):
            copy(model / name, "weights/" + name)
        for name in ("manifest.json", "input.rgb", "expected.rgb"):
            copy(fixture / name, "fixture/" + name)
        if (fixture / "source.json").is_file():
            copy(fixture / "source.json", "fixture/source.json")
        for name in ("initialization.json", "validation.json", "verification.json"):
            if (model / name).is_file():
                copy(model / name, "weights/" + name)
        if ffmpeg:
            copy(ffmpeg, "bin/ffmpeg")
            copy(ffprobe, "bin/ffprobe")
        for name in ("sw/__init__.py", "sw/video/__init__.py", "sw/verification/__init__.py",
                     "sw/video/interpolate_video.py", "sw/verification/export_weights.py"):
            copy(ROOT / name, name)
        (stage / "README.txt").write_text(
            "FIACCEL U200 trained-model bundle\n"
            "Requires Python >= 3.9, XRT and a completed, timing-validated FIACCEL hardware xclbin.\n"
            "No PyTorch, Brevitas or CUDA is needed on the board server.\n\n"
            "python3 u200_model.py check .\n"
            "python3 u200_model.py self-test . --xclbin /path/FIACCEL.xclbin\n"
            "python3 u200_model.py video . input.mp4 output.mp4 --xclbin /path/FIACCEL.xclbin\n\n"
            "Video uses bundled bin/ffmpeg and bin/ffprobe when present, otherwise system tools.\n"
            "FFmpeg must include libx264. Full-HD RGB input is required.\n"
            "The bundle includes weights, fixed shifts and an exact RGB24 test stream, not an xclbin.\n"
            "Use sw/verification/run_u200.py --build BUILD --model-bundle BUNDLE from the build\n"
            "machine to check routed timing/resources and device availability before deployment.\n")
        manifest = dict(format=FORMAT, board="Alveo U200", width=1920, height=1080,
            configuration_records=count, weight_count=count - 12, shifts=12,
            model_manifest=json.loads((model / "manifest.json").read_text()),
            fixture=fixture_info, hardware_included=False,
            sha256={str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()})
        (stage / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_bundle(stage)
        stage.rename(output)
    return manifest


def require_idle_device():
    if os.environ.get("XCL_EMULATION_MODE") == "hw_emu":
        return
    nodes = []
    for entry in Path("/sys/class/drm").glob("renderD*"):
        vendor = entry / "device/vendor"
        if vendor.is_file() and vendor.read_text().strip() == "0x10ee":
            nodes.append("/dev/dri/" + entry.name)
    if not nodes:
        raise RuntimeError("No Xilinx render device found")
    check = subprocess.run(["fuser", *nodes], capture_output=True, text=True)
    if check.returncode != 1 or check.stdout.strip() or check.stderr.strip():
        raise RuntimeError("U200 is occupied or its availability cannot be checked: " + check.stdout + check.stderr)


def compare_rgb(actual, expected):
    differences, maximum, first, offset = 0, 0, None, 0
    with Path(actual).open("rb") as a, Path(expected).open("rb") as b:
        while True:
            left, right = a.read(1024 * 1024), b.read(1024 * 1024)
            if not left and not right:
                break
            if left != right:
                for index, (x, y) in enumerate(zip(left, right)):
                    if x != y:
                        differences += 1
                        maximum = max(maximum, abs(x - y))
                        if first is None:
                            first = offset + index
                if len(left) != len(right) and first is None:
                    first = offset + min(len(left), len(right))
            offset += len(right)
    sizes = (Path(actual).stat().st_size, Path(expected).stat().st_size)
    return dict(exact=sizes[0] == sizes[1] and differences == 0,
        actual_bytes=sizes[0], expected_bytes=sizes[1], differing_rgb_bytes=differences,
        max_rgb_error=maximum, first_difference_byte=first)


def self_test(bundle, xclbin, output, timeout=120, strict=False):
    bundle, xclbin, output = map(lambda p: Path(p).resolve(), (bundle, xclbin, output))
    manifest = verify_bundle(bundle)
    if not xclbin.is_file():
        raise ValueError("Missing FIACCEL hardware xclbin")
    if output.exists():
        raise ValueError("Choose a new result directory")
    require_idle_device()
    output.mkdir(parents=True)
    command = [str(bundle / "host"), str(xclbin), "--video-stream", "--weights", str(bundle / "weights/config.hex")]
    if strict:
        command.append("--enforce-cycle-budget")
    environment = dict(os.environ, FIACCEL_TIMEOUT_SECONDS=str(timeout))
    environment.pop("FIACCEL_CONFIG_SMOKE", None)
    report = dict(result="FAIL", backend="XRT hw_emu" if environment.get("XCL_EMULATION_MODE") == "hw_emu" else "U200 XRT",
        weights_sha256=sha256(bundle / "weights/config.hex"), xclbin_sha256=sha256(xclbin),
        command=command, scope="Exact trained-model RGB stream versus integer reference; no PSNR claim")
    try:
        with (bundle / "fixture/input.rgb").open("rb") as source, \
                (output / "actual.rgb").open("wb") as actual, (output / "host.log").open("wb") as log:
            result = subprocess.run(command, stdin=source, stdout=actual, stderr=log, env=environment,
                timeout=timeout * (manifest["fixture"]["input_frames"] + 2) + 60)
        report["host_exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError("Trained-model host failed; see host.log")
        summaries = re.findall(r"VIDEO_SUMMARY (.*)", (output / "host.log").read_text())
        if len(summaries) != 1:
            raise RuntimeError("Missing unique video completion summary")
        counts = {key: int(value) for key, value in (word.split("=", 1) for word in summaries[0].split())}
        report["stream"] = counts
        for name in ("width", "height", "input_frames", "output_frames", "interpolated_frames"):
            if counts[name] != manifest["fixture"][name]:
                raise RuntimeError("Unexpected trained-model stream dimensions or frame counts")
        report["comparison"] = compare_rgb(output / "actual.rgb", bundle / "fixture/expected.rgb")
        if not report["comparison"]["exact"]:
            raise RuntimeError("Board RGB output differs from the integer reference")
        with (output / "actual.rgb").open("rb") as actual, (output / "midpoint.ppm").open("wb") as image:
            actual.seek(FRAME_BYTES)
            image.write(b"P6\n1920 1080\n255\n")
            image.write(actual.read(FRAME_BYTES))
        report["result"] = "PASS"
        (output / "actual.rgb").unlink()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        report["error"] = str(error)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    package = commands.add_parser("pack")
    for name in ("model", "host", "fixture", "output"):
        package.add_argument("--" + name, type=Path, required=True)
    package.add_argument("--ffmpeg", type=Path)
    package.add_argument("--ffprobe", type=Path)
    check = commands.add_parser("check")
    check.add_argument("bundle", type=Path)
    test = commands.add_parser("self-test")
    test.add_argument("bundle", type=Path)
    test.add_argument("--xclbin", type=Path, required=True)
    test.add_argument("--output", type=Path)
    test.add_argument("--timeout", type=int, default=120)
    test.add_argument("--enforce-cycle-budget", action="store_true")
    video = commands.add_parser("video")
    video.add_argument("bundle", type=Path)
    video.add_argument("input", type=Path)
    video.add_argument("output", type=Path)
    video.add_argument("--xclbin", type=Path, required=True)
    video.add_argument("--ffmpeg")
    video.add_argument("--ffprobe")
    video.add_argument("--threads", type=int, default=2)
    video.add_argument("--crf", type=int, choices=range(52), default=18)
    for name in ("overwrite", "audio", "enforce-cycle-budget"):
        video.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "pack":
            manifest = pack(args.model, args.host, args.fixture, args.output, args.ffmpeg, args.ffprobe)
            print(json.dumps(dict(result="PASS", bundle=str(args.output), configuration_records=manifest["configuration_records"])))
        elif args.action == "check":
            manifest = verify_bundle(args.bundle)
            print(json.dumps(dict(result="PASS", files=len(manifest["sha256"]), configuration_records=manifest["configuration_records"])))
        elif args.action == "self-test":
            if args.timeout < 1:
                raise ValueError("Timeout must be positive")
            report = self_test(args.bundle, args.xclbin, args.output or args.bundle / "result", args.timeout, args.enforce_cycle_budget)
            print(json.dumps(report, indent=2))
            return 0 if report["result"] == "PASS" else 1
        else:
            bundle = args.bundle.resolve()
            verify_bundle(bundle)
            if not args.xclbin.is_file() or args.threads < 1:
                raise ValueError("Provide a hardware xclbin and at least one encoding thread")
            require_idle_device()
            ffmpeg = args.ffmpeg or (str(bundle / "bin/ffmpeg") if (bundle / "bin/ffmpeg").is_file() else "ffmpeg")
            ffprobe = args.ffprobe or (str(bundle / "bin/ffprobe") if (bundle / "bin/ffprobe").is_file() else "ffprobe")
            options = ["--ffmpeg", ffmpeg, "--ffprobe", ffprobe,
                       "--threads", str(args.threads), "--crf", str(args.crf)]
            for name in ("overwrite", "audio", "enforce_cycle_budget"):
                if getattr(args, name):
                    options.append("--" + name.replace("_", "-"))
            return subprocess.run([sys.executable, str(bundle / "sw/video/interpolate_video.py"),
                str(args.input), str(args.output), "--xclbin", str(args.xclbin), "--host", str(bundle / "host"),
                "--weights", str(bundle / "weights/config.hex"), *options]).returncode
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"FAIL: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
