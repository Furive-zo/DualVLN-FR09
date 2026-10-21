#!/usr/bin/env python3
"""Verify downloaded model weights against SHA256SUMS (models/ and checkpoints/)."""
import hashlib
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]


def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def main():
    entries = [line.split(None, 1) for line in (BUNDLE / 'SHA256SUMS').read_text().splitlines() if line.strip()]
    missing, mismatched = [], []
    for i, (digest, name) in enumerate(entries, 1):
        path = BUNDLE / name.strip()
        if not path.is_file():
            missing.append(name)
            status = 'MISSING'
        elif sha256(path) != digest:
            mismatched.append(name)
            status = 'MISMATCH'
        else:
            status = 'ok'
        print(f'[{i}/{len(entries)}] {status:8} {name}', flush=True)
    if missing or mismatched:
        print(f'FAILED: {len(missing)} missing, {len(mismatched)} mismatched')
        sys.exit(1)
    print(f'all {len(entries)} files verified')


if __name__ == '__main__':
    main()
