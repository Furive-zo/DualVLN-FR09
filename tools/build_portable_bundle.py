#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve()
BUNDLE = HERE.parents[1]
PROJECT = HERE.parents[3]
MODEL_AUX = {
    'config.json', 'generation_config.json', 'model.safetensors.index.json',
    'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
    'vocab.json', 'merges.txt', 'tokenizer.json', 'chat_template.jinja',
    'preprocessor_config.json', 'video_preprocessor_config.json',
}


def copy_model(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / 'model.safetensors.index.json').read_text())
    names = MODEL_AUX | set(index['weight_map'].values())
    copied = []
    for name in sorted(names):
        path = src / name
        if path.is_file():
            shutil.copy2(path, dst / name)
            copied.append(name)
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('destination')
    args = ap.parse_args()
    destination = Path(args.destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f'destination must be absent or empty: {destination}')
    shutil.copytree(BUNDLE, destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('models', '__pycache__'))
    manifest = json.loads((BUNDLE / 'config/release_manifest.json').read_text())
    copied = {}
    for key, spec in manifest['models'].items():
        src = PROJECT / spec['source_path']
        copied[key] = copy_model(src, destination / spec['portable_path'])
    runtime_src = PROJECT / 'InternNav/internnav'
    shutil.copytree(runtime_src, destination / 'runtime/internnav', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT / 'scripts/_fix_nan_params.py', destination / 'runtime/_fix_nan_params.py')
    checkpoint_dir = destination / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT / 'checkpoints/depth_anything_v2_metric_hypersim_vits.pth',
                 checkpoint_dir / 'depth_anything_v2_metric_hypersim_vits.pth')
    (destination / 'config/copied_files.json').write_text(
        json.dumps(copied, indent=2, ensure_ascii=False) + '\n')
    print(destination)


if __name__ == '__main__':
    main()
