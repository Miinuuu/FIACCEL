#!/usr/bin/env python3
"""Run the encrypted FIACCEL black-box smoke test with Vivado XSim."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/public_xsim')
    parser.add_argument('--data-width', type=int, choices=[32, 64, 128, 256, 512], default=32)
    parser.add_argument('--waves', action='store_true', help='Record internal signals to waves.wdb')
    args = parser.parse_args()
    fixture = ROOT / 'hardware/fixtures/smoke'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / p for p in (ROOT / 'rtl_vp/files.f').read_text().splitlines() if p.strip()]
    manifest = json.loads((ROOT / 'rtl_vp/manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Encrypted source checksum mismatch: ' + name)
    for source in sources:
        if 'data_block' not in source.read_text() or re.search(r'^\s*module\s', source.read_text(), re.M):
            raise RuntimeError('Expected fully encrypted RTL: ' + str(source))
    tools = {}
    for name in ('xvlog', 'xelab', 'xsim'):
        tools[name] = shutil.which(name)
        if not tools[name]:
            raise RuntimeError('Missing ' + name + '; source Vivado settings64.sh first')
    for source in fixture.iterdir():
        if source.is_file():
            shutil.copy2(source, output / source.name)
    benches = [ROOT / 'hardware/tb/fiaccel_public_tb.sv', ROOT / 'hardware/tb/memory_model.sv']
    simulation = [tools['xsim'], 'fiaccel_public', '-runall']
    if args.waves:
        script = output/'waves.tcl'
        script.write_text('log_wave -r /fiaccel_public_tb\nrun 10ns\n'
            'set sample [get_value -radix bin /fiaccel_public_tb/debug_encoder_x]\n'
            'puts "INTERNAL_SIGNAL_READ encoder2.rx=$sample"\nrun all\nexit\n')
        simulation = [tools['xsim'], 'fiaccel_public', '-wdb', 'waves.wdb', '-tclbatch', str(script)]
    commands = [
        [tools['xvlog'], '--incr', '-d', 'FIACCEL_SIM_DEBUG', *map(str, sources)],
        [tools['xvlog'], '--sv', '-d', 'FIACCEL_SIM_DEBUG', '-i', str(fixture), '-d', 'AXI_DW=' + str(args.data_width), *map(str, benches)],
        [tools['xelab'], 'work.fiaccel_public_tb', '-s', 'fiaccel_public', '-debug', 'all', '--O0'],
        simulation,
    ]
    for index, command in enumerate(commands):
        log = output / ('step_%d.log' % index)
        print('Running ' + Path(command[0]).name + ' (log: ' + str(log) + ')', flush=True)
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Command failed; see ' + str(log))
    text = (output / 'step_3.log').read_text()
    if 'FIACCEL_PUBLIC_PASS' not in text or re.search(r'(?im)^\s*(Fatal:|FATAL:|ERROR:)', text):
        raise RuntimeError('No successful simulation completion; see ' + str(output / 'step_3.log'))
    actual = (output / 'rtl_output.hex').read_text().split()
    expected = (fixture / 'expected.hex').read_text().split()
    if [int(x,16) for x in actual] != [int(x,16) for x in expected]:
        raise RuntimeError('RGB output differs from golden fixture')
    if args.waves and ('INTERNAL_SIGNAL_READ encoder2.rx=0' not in text or
                      not (output/'waves.wdb').is_file()):
        raise RuntimeError('Internal signal visibility / waveform recording check failed')
    report = dict(result='PASS', simulator='Vivado XSim', axi_data_width=args.data_width,
                  protected_module_count=len(sources),
                  encrypted_manifest_sha256=hashlib.sha256((ROOT/'rtl_vp/manifest.json').read_bytes()).hexdigest(),
                  source_fingerprint_sha256=manifest['source_fingerprint_sha256'],
                  checked_rgb_pixels=len(actual), fixture='smoke',
                  internal_signal_read='PASS' if args.waves else 'not requested',
                  waveform='waves.wdb' if args.waves else None,
                  scope='Small deterministic functional test; synthetic weights; no timing or paper PSNR validation')
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
