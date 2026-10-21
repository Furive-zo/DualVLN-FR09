#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve()
BUNDLE = HERE.parents[1]
PROJECT = HERE.parents[3]


def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def resolve_model(spec):
    portable = BUNDLE / spec['portable_path']
    source = PROJECT / spec['source_path']
    return portable if portable.is_dir() else source


def check_model(name, path):
    required = [
        'config.json', 'generation_config.json', 'model.safetensors.index.json',
        'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt',
    ]
    missing = [f for f in required if not (path / f).is_file()]
    if missing:
        raise RuntimeError(f'{name}: missing files: {missing}')
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    absent = [s for s in shards if not (path / s).is_file()]
    if absent:
        raise RuntimeError(f'{name}: missing model shards: {absent}')
    cfg = json.loads((path / 'config.json').read_text())
    return {
        'path': str(path),
        'architecture': cfg.get('architectures', [None])[0],
        'vocab_size': cfg.get('vocab_size'),
        'image_token_id': cfg.get('image_token_id'),
        'eos_token_id': cfg.get('eos_token_id'),
        'shards': len(shards),
        'weights_gib': round(sum((path / s).stat().st_size for s in shards) / 2**30, 3),
        'config_sha256': sha256(path / 'config.json'),
        'index_sha256': sha256(path / 'model.safetensors.index.json'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write-lock', action='store_true')
    args = ap.parse_args()
    manifest = json.loads((BUNDLE / 'config/release_manifest.json').read_text())
    dav2_name = 'depth_anything_v2_metric_hypersim_vits.pth'
    portable_dav2 = BUNDLE / 'checkpoints' / dav2_name
    project_dav2 = PROJECT / 'checkpoints' / dav2_name
    dav2_path = portable_dav2 if portable_dav2.is_file() else project_dav2
    if not dav2_path.is_file():
        raise RuntimeError(f'missing required DepthAnything checkpoint: {dav2_path}')
    resolved = {k: resolve_model(v) for k, v in manifest['models'].items()}
    report = {k: check_model(k, p) for k, p in resolved.items()}
    s2, s1 = report['system2'], report['system1']
    for key in ('vocab_size', 'image_token_id', 'eos_token_id'):
        if s2[key] != s1[key]:
            raise RuntimeError(f'S2/S1 incompatible {key}: {s2[key]} != {s1[key]}')
    report['depth_anything'] = {'path': str(dav2_path), 'sha256': sha256(dav2_path)}
    report['handoff'] = {
        'compatible_special_token_ids': True,
        'retokenization_required': sha256(resolved['system2'] / 'vocab.json') != sha256(resolved['system1'] / 'vocab.json'),
        'rule': 'decode S2 coordinate text, then tokenize with S1 tokenizer',
    }
    contract = json.loads((BUNDLE / 'config/runtime_contract.json').read_text())
    if contract['trajectory']['semantic_forward_axis'] != 'negative_y':
        raise RuntimeError('unexpected trajectory forward axis')
    if args.write_lock:
        out = BUNDLE / 'config/release_lock.json'
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(f'wrote {out}')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
