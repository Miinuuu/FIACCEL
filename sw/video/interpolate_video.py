#!/usr/bin/env python3
"""Decode Full-HD video, interpolate on U200, and encode at twice the FPS."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sw.verification.export_weights import NAMES, export, shapes

DEPTHS = [8, 16, 32, 64, 32, 16, 8]
RANDOM_SHIFTS = [8, 9, 10, 11, 11, 11, 11, 10, 10, 9, 8, 3]
SUMMARY = re.compile(r"VIDEO_SUMMARY (.*)")


def random_weights(destination, seed):
    """Generate complete signed INT8 coefficients in the production layout."""
    rng = random.Random(seed)
    document = {"depths": DEPTHS, "output_channels": 3, "layers": {}}
    for name, shape, shift in zip(NAMES, shapes(DEPTHS), RANDOM_SHIFTS):
        document["layers"][name] = dict(shape=shape, shift=shift,
            values=[rng.randint(-128, 127) for _ in range(math.prod(shape))])
    export(document, destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(training="untrained random INT8; I/O validation only", seed=seed)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return destination / "config.hex"


def probe(path, ffprobe, count=False):
    command = [str(ffprobe), "-v", "error", "-select_streams", "v:0"]
    if count:
        command += ["-count_frames"]
    result = subprocess.run(command + ["-show_streams", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError("Input contains no video stream")
    stream = streams[0]
    if (stream["width"], stream["height"]) != (1920, 1080):
        raise ValueError("The kernel requires 1920x1080 input; resize explicitly before running")
    try:
        fps = Fraction(stream.get("avg_frame_rate", "0/1"))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Cannot determine the input frame rate") from error
    if fps <= 0:
        raise ValueError("Cannot determine the input frame rate")
    return stream, fps


def pipeline(commands, log_paths):
    """Connect bounded OS pipes; stderr goes to files to avoid deadlocks."""
    processes, logs = [], []
    try:
        for index, (command, log_path) in enumerate(zip(commands, log_paths)):
            log = log_path.open("wb")
            logs.append(log)
            source = processes[-1].stdout if processes else subprocess.DEVNULL
            process = subprocess.Popen(command, stdin=source,
                stdout=subprocess.PIPE if index + 1 < len(commands) else subprocess.DEVNULL,
                stderr=log)
            if processes:
                processes[-1].stdout.close()
            processes.append(process)
        while True:
            codes = [process.poll() for process in processes]
            failed = [i for i, code in enumerate(codes) if code not in (None, 0)]
            if failed:
                index = failed[0]
                raise RuntimeError(f"Video stage {index + 1} failed ({codes[index]}); see {log_paths[index]}")
            if all(code is not None for code in codes):
                break
            time.sleep(0.05)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()
        for log in logs:
            log.close()


def run_video(source, destination, xclbin, host, weights, ffmpeg, ffprobe,
              *, overwrite=False, audio=False, strict=False, crf=18, threads=2):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Input and output paths must differ")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {destination}; use --overwrite")
    if destination.suffix.lower() not in (".mp4", ".mov", ".mkv"):
        raise ValueError("Choose an .mp4, .mov or .mkv output")
    stream, fps = probe(source, ffprobe)
    output_fps = fps * 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    evidence = destination.with_name(destination.name + ".fiaccel")
    evidence.mkdir(exist_ok=True)
    logs = [evidence / name for name in ("decode.log", "host.log", "encode.log")]
    with tempfile.TemporaryDirectory(prefix=".fiaccel_", dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        decoder = [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-threads", str(threads), "-i", str(source), "-map", "0:v:0", "-an",
            "-vf", f"fps={fps.numerator}/{fps.denominator}", "-pix_fmt", "rgb24",
            "-threads", str(threads), "-f", "rawvideo", "pipe:1"]
        accelerator = [str(Path(host).resolve()), str(Path(xclbin).resolve()),
                       "--video-stream", "--weights", str(Path(weights).resolve())]
        if strict:
            accelerator.append("--enforce-cycle-budget")
        encoder = [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1920x1080",
            "-framerate", f"{output_fps.numerator}/{output_fps.denominator}", "-i", "pipe:0"]
        if audio:
            encoder += ["-i", str(source), "-map", "0:v:0", "-map", "1:a?", "-c:a", "aac"]
        encoder += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-threads", str(threads), str(temporary)]
        pipeline([decoder, accelerator, encoder], logs)
        matches = SUMMARY.findall(logs[1].read_text())
        if len(matches) != 1:
            raise RuntimeError(f"Missing unique host completion summary: {logs[1]}")
        counts = {key: int(value) for key, value in
                  (word.split("=", 1) for word in matches[0].split())}
        n = counts["input_frames"]
        if (counts["width"], counts["height"]) != (1920, 1080):
            raise RuntimeError("Host frame dimensions mismatch")
        if n < 1 or counts["output_frames"] != 2*n or counts["interpolated_frames"] != n-1:
            raise RuntimeError("Host frame accounting mismatch")
        output_stream, actual_fps = probe(temporary, ffprobe, count=True)
        if actual_fps != output_fps or int(output_stream["nb_read_frames"]) != 2*n:
            raise RuntimeError("Encoded output FPS/frame count mismatch")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output appeared during encoding: {destination}")
        # A completed output becomes visible only after all three stages pass.
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    backend = "XRT hw_emu" if os.environ.get("XCL_EMULATION_MODE") == "hw_emu" else "U200 XRT"
    report = dict(result="PASS", backend=backend, input=str(source), output=str(destination),
        input_fps=str(fps), output_fps=str(output_fps), **counts,
        last_frame="repeated once to preserve duration", audio=audio,
        weights=str(Path(weights).resolve()), weights_sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        xclbin=str(Path(xclbin).resolve()), commands=[decoder, accelerator, encoder])
    manifest = Path(weights).parent / "manifest.json"
    if manifest.is_file():
        information = json.loads(manifest.read_text())
        report.update(weight_manifest=str(manifest.resolve()),
            weight_training=information.get("training", "user-provided coefficients"),
            random_seed=information.get("seed"))
    (evidence / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--xclbin", type=Path, required=True)
    parser.add_argument("--host", type=Path, default=ROOT/"build/u200/host")
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--weights", type=Path)
    weights.add_argument("--random-weights", action="store_true", help="untrained; I/O validation only")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audio", action="store_true", help="also encode the original audio")
    parser.add_argument("--enforce-cycle-budget", action="store_true")
    parser.add_argument("--crf", type=int, choices=range(52), default=18)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        config = args.weights
        if args.random_weights:
            config = random_weights(args.output.resolve().with_name(args.output.name + ".fiaccel")/"random_weights", args.seed)
        report = run_video(args.input, args.output, args.xclbin, args.host, config,
            args.ffmpeg, args.ffprobe, overwrite=args.overwrite, audio=args.audio,
            strict=args.enforce_cycle_budget, crf=args.crf, threads=args.threads)
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
