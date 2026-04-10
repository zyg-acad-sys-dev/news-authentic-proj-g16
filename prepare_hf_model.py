from __future__ import annotations

import argparse
from pathlib import Path


def safe_model_slug(model_name: str) -> str:
    return model_name.replace('/', '__').replace('-', '_')


def main() -> int:
    parser = argparse.ArgumentParser(description='Download one Hugging Face model snapshot into a local project directory for offline / firewalled use.')
    parser.add_argument('--model-name', default='bert-base-uncased')
    parser.add_argument('--output-dir', default='assets/hf')
    parser.add_argument('--cache-dir', default='')
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError('prepare_hf_model.py requires huggingface_hub.') from exc

    repo_root = Path(__file__).resolve().parent
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    target_dir = output_root / safe_model_slug(args.model_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        'repo_id': args.model_name,
        'local_dir': str(target_dir),
        'repo_type': 'model',
    }
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = repo_root / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs['cache_dir'] = str(cache_dir)

    print(f'[info] Downloading {args.model_name} -> {target_dir}')
    snapshot_download(**kwargs)
    print(f'[ok] local pretrained snapshot is ready at {target_dir}')
    print('[tip] Share this folder with teammates and keep train_bert.py pointed at it, or just keep it under assets/hf/.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
