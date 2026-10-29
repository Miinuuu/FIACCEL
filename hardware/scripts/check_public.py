#!/usr/bin/env python3
"""Check the protected RTL package before sharing or running it."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
APPROVED_IMAGES = {
    'hardware/u200/prebuilt/FIACCEL.xclbin': '27bbf5cc8098762fab3396f54c55d26a38a7b618f70b14f7fc37963051d78bda',
    'hardware/u200/hw_emu/FIACCEL.xclbin': 'bdaaa446d2dee1c3dbfeb0dbfdfdb6f9121d92f62e4421a8e7e588fec3a19b28',
}


def check():
    for name, digest in APPROVED_IMAGES.items():
        image = ROOT/name
        image_manifest = json.loads(image.with_name('manifest.json').read_text())
        if (hashlib.sha256(image.read_bytes()).hexdigest() != digest or
                image_manifest['sha256'] != digest or
                image_manifest['bytes'] != image.stat().st_size or
                image_manifest['file'] != image.name):
            raise ValueError('Prebuilt image checksum/manifest mismatch: '+name)
        for support, checksum in image_manifest.get('support_files', {}).items():
            if support not in ('emconfig.json', 'xrt.ini') or hashlib.sha256(image.with_name(support).read_bytes()).hexdigest() != checksum:
                raise ValueError('Emulation support file checksum mismatch: '+support)
    manifest = json.loads((ROOT/'rtl_vp/manifest.json').read_text())
    listed = (ROOT/'rtl_vp/files.f').read_text().splitlines()
    actual = {str(p.relative_to(ROOT)) for p in (ROOT/'rtl_vp').rglob('*.vp')}
    if len(actual) != 30 or set(listed) != actual or set(manifest['sha256']) != actual:
        raise ValueError('Expected exactly 30 protected modules in list and manifest')
    forbidden = {'.v', '.vhd', '.vhdl', '.lic', '.dcp', '.edf', '.edn', '.bit', '.xo', '.xclbin', '.rpt', '.log', '.jou', '.pb'}
    for folder in ('rtl_vp', 'hardware'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix.lower() in forbidden and str(path.relative_to(ROOT)) not in APPROVED_IMAGES:
                raise ValueError('Private/generated artifact in public package: '+str(path.relative_to(ROOT)))
    for name in listed:
        path = ROOT/name
        text = path.read_text()
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256'][name]:
            raise ValueError('Checksum mismatch: '+name)
        if 'data_block' not in text or re.search(r'^\s*module\s', text, re.M):
            raise ValueError('Expected fully protected implementation: '+name)
        for control in ('xilinx_enable_netlist_export', 'xilinx_enable_bitstream'):
            if f'{control} = "false"' not in text:
                raise ValueError('Missing access restriction: '+name+' / '+control)
        for control in ('xilinx_enable_probing', 'xilinx_schematic_visibility'):
            if f'{control} = "true"' not in text:
                raise ValueError('Missing debug visibility: '+name+' / '+control)
        if 'xilinx_activity==simulation' not in text or 'activity==simulation' not in text:
            raise ValueError('Missing simulation-only decryption restriction: '+name)
    tracked = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'],
                             capture_output=True, check=True).stdout.decode().split('\0')
    for name in filter(None, tracked):
        if not (ROOT/name).is_file():
            continue
        if name not in APPROVED_IMAGES and (name.startswith('rtl/') or Path(name).suffix.lower() in forbidden):
            raise ValueError('Tracked private/generated artifact: '+name)
    print('PASS: 30 encrypted modules; simulation-only rights; approved U200/hw_emu image checksums; no other private hardware artifacts')


if __name__ == '__main__':
    try:
        check()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit('FAIL: '+str(error))
