from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_list(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(',') if x.strip()]


def safe_float_label(value) -> str:
    text = f"{value}"
    return text.replace('.', 'p').replace('-', 'm')


def slug_for(model: str, loss_type: str, run_tag: str) -> str:
    return f"{model}_{loss_type}".replace('cross_entropy', 'ce') + f'_{run_tag}'


def run_recurrent(root: Path, args_list: list[str]) -> None:
    cmd = [sys.executable, str(root / 'train_recurrent.py'), *args_list]
    print('[run]', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=str(root))


def read_epoch_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def ensure_epoch_run(root: Path, common: list[str], model: str, loss_type: str, learning_rate: float, batch_size: int, run_tag: str) -> tuple[str, pd.DataFrame]:
    slug = slug_for(model, loss_type, run_tag)
    out_csv = root / 'outputs' / 'tables' / f'{slug}_epoch_logs.csv'
    run_recurrent(
        root,
        common + [
            '--loss-type', loss_type,
            '--learning-rate', str(learning_rate),
            '--batch-size', str(batch_size),
            '--run-tag', run_tag,
        ],
    )
    return slug, read_epoch_csv(out_csv)


def plot_base_curves(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10), sharex=True)
    axes[0].plot(df['epoch'], df['train_loss'], marker='o', label='train_loss')
    axes[0].set_ylabel('Train loss')
    axes[0].legend()

    if 'train_accuracy' in df.columns:
        axes[1].plot(df['epoch'], df['train_accuracy'], marker='o', label='train_accuracy')
    if 'test_accuracy' in df.columns:
        axes[1].plot(df['epoch'], df['test_accuracy'], marker='s', label='test_accuracy')
    if 'val_accuracy' in df.columns:
        axes[1].plot(df['epoch'], df['val_accuracy'], marker='^', label='val_accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    if 'train_f1_macro' in df.columns:
        axes[2].plot(df['epoch'], df['train_f1_macro'], marker='o', label='train_f1_macro')
    if 'test_f1_macro' in df.columns:
        axes[2].plot(df['epoch'], df['test_f1_macro'], marker='s', label='test_f1_macro')
    if 'val_f1_macro' in df.columns:
        axes[2].plot(df['epoch'], df['val_f1_macro'], marker='^', label='val_f1_macro')
    axes[2].set_ylabel('Macro-F1')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_two_run_comparison(base_df: pd.DataFrame, alt_df: pd.DataFrame, out_path: Path, title: str, alt_label: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10), sharex=True)
    axes[0].plot(base_df['epoch'], base_df['train_loss'], marker='o', label='base_train_loss')
    axes[0].plot(alt_df['epoch'], alt_df['train_loss'], marker='s', label=f'{alt_label}_train_loss')
    axes[0].set_ylabel('Train loss')
    axes[0].legend()

    axes[1].plot(base_df['epoch'], base_df['train_accuracy'], marker='o', label='base_train_acc')
    axes[1].plot(base_df['epoch'], base_df['test_accuracy'], marker='o', linestyle='--', label='base_test_acc')
    axes[1].plot(alt_df['epoch'], alt_df['train_accuracy'], marker='s', label=f'{alt_label}_train_acc')
    axes[1].plot(alt_df['epoch'], alt_df['test_accuracy'], marker='s', linestyle='--', label=f'{alt_label}_test_acc')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    axes[2].plot(base_df['epoch'], base_df['test_f1_macro'], marker='o', label='base_test_f1')
    axes[2].plot(alt_df['epoch'], alt_df['test_f1_macro'], marker='s', label=f'{alt_label}_test_f1')
    axes[2].set_ylabel('Test macro-F1')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_sweep(metric_frames: list[tuple[str, pd.DataFrame]], out_path_loss: Path, out_path_acc: Path, title_prefix: str) -> None:
    fig1, ax1 = plt.subplots(figsize=(8.5, 5.5))
    for label, df in metric_frames:
        ax1.plot(df['epoch'], df['train_loss'], marker='o', label=label)
    ax1.set_title(f'{title_prefix}: training loss by epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train loss')
    ax1.legend()
    fig1.tight_layout()
    out_path_loss.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(out_path_loss, dpi=180)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8.5, 5.5))
    for label, df in metric_frames:
        if 'test_accuracy' in df.columns:
            ax2.plot(df['epoch'], df['test_accuracy'], marker='o', label=f'{label} test_acc')
        elif 'val_accuracy' in df.columns:
            ax2.plot(df['epoch'], df['val_accuracy'], marker='o', label=f'{label} val_acc')
    ax2.set_title(f'{title_prefix}: accuracy by epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    fig2.tight_layout()
    out_path_acc.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_path_acc, dpi=180)
    plt.close(fig2)


def export_first_n_predictions(root: Path, model_slug: str, text_column: str, n: int, out_path: Path) -> None:
    split_df = pd.read_csv(root / 'data' / 'splits' / 'split_table.csv')
    pred_path = root / 'outputs' / 'predictions' / f'{model_slug}_test_predictions.csv'
    pred_df = pd.read_csv(pred_path)
    merge_cols = [c for c in ['sample_id', 'title', text_column] if c in split_df.columns]
    merged = pred_df.merge(split_df[merge_cols], on='sample_id', how='left')
    if text_column in merged.columns:
        merged['text_preview'] = merged[text_column].fillna('').astype(str).str.slice(0, 500)
    keep_cols = [c for c in ['sample_id', 'true_label', 'pred_label', 'pred_score', 'model_name', 'title', text_column, 'text_preview'] if c in merged.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.head(n)[keep_cols].to_csv(out_path, index=False)


def append_summary(summary_rows: list[dict], sweep_type: str, setting, df: pd.DataFrame) -> None:
    last = df.iloc[-1].to_dict()
    summary_rows.append({'sweep_type': sweep_type, 'setting': setting, **last})


def main() -> int:
    parser = argparse.ArgumentParser(description='Run recurrent experiment sweeps and export diagnostic figures/tables.')
    parser.add_argument('--mode', default='full', choices=['full', 'base', 'loss', 'learning_rate', 'batch_size', 'export_first_n'], help='Select a full sweep or a single experiment mode.')
    parser.add_argument('--input', default='data/splits/split_table.csv')
    parser.add_argument('--text-column', default='clean_text')
    parser.add_argument('--model', default='lstm', choices=['rnn', 'gru', 'lstm', 'bigru', 'bilstm'])
    parser.add_argument('--positive-label', default='fake')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--base-loss', default='cross_entropy')
    parser.add_argument('--alt-loss', default='weighted_ce')
    parser.add_argument('--base-learning-rate', type=float, default=1e-3)
    parser.add_argument('--base-batch-size', type=int, default=32)
    parser.add_argument('--learning-rates', default='0.1,0.01,0.001,0.0001')
    parser.add_argument('--batch-sizes', default='8,16,32,64,128')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--first-n', type=int, default=100)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    fig_dir = root / 'outputs' / 'figures' / 'experiment_diagnostics'
    table_dir = root / 'outputs' / 'tables' / 'experiment_diagnostics'
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    common = [
        '--input', args.input,
        '--text-column', args.text_column,
        '--model', args.model,
        '--positive-label', args.positive_label,
        '--epochs', str(args.epochs),
        '--seed', str(args.seed),
    ]

    summary_rows: list[dict] = []
    base_tag = 'exp_base'
    alt_tag = 'exp_alt_loss'
    base_slug, base_df = ensure_epoch_run(root, common, args.model, args.base_loss, args.base_learning_rate, args.base_batch_size, base_tag)

    if args.mode in {'full', 'base'}:
        plot_base_curves(base_df, fig_dir / 'base_training_curves.png', f'{args.model.upper()} base training curves')
        append_summary(summary_rows, 'base', f'loss={args.base_loss},lr={args.base_learning_rate},bs={args.base_batch_size}', base_df)

    if args.mode in {'full', 'loss'}:
        alt_slug, alt_df = ensure_epoch_run(root, common, args.model, args.alt_loss, args.base_learning_rate, args.base_batch_size, alt_tag)
        plot_two_run_comparison(base_df, alt_df, fig_dir / 'loss_function_comparison.png', f'{args.model.upper()} different loss functions', args.alt_loss)
        append_summary(summary_rows, 'loss', args.alt_loss, alt_df)

    if args.mode in {'full', 'learning_rate'}:
        lr_frames = []
        for lr in parse_list(args.learning_rates, float):
            tag = f'lr_{safe_float_label(lr)}'
            _, df = ensure_epoch_run(root, common, args.model, args.base_loss, lr, args.base_batch_size, tag)
            lr_frames.append((f'lr={lr}', df))
            append_summary(summary_rows, 'learning_rate', lr, df)
        plot_sweep(lr_frames, fig_dir / 'learning_rate_sweep_loss.png', fig_dir / 'learning_rate_sweep_accuracy.png', f'{args.model.upper()} learning-rate sweep')

    if args.mode in {'full', 'batch_size'}:
        bs_frames = []
        for bs in parse_list(args.batch_sizes, int):
            tag = f'bs_{bs}'
            _, df = ensure_epoch_run(root, common, args.model, args.base_loss, args.base_learning_rate, bs, tag)
            bs_frames.append((f'batch={bs}', df))
            append_summary(summary_rows, 'batch_size', bs, df)
        plot_sweep(bs_frames, fig_dir / 'batch_size_sweep_loss.png', fig_dir / 'batch_size_sweep_accuracy.png', f'{args.model.upper()} batch-size sweep')

    if args.mode in {'full', 'export_first_n'}:
        export_first_n_predictions(root, base_slug, args.text_column, args.first_n, table_dir / f'first_{args.first_n}_test_predictions_{base_slug}.csv')
        print(f'[ok] wrote first-{args.first_n} prediction export')

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(table_dir / 'sweep_summary.csv', index=False)
    print(f'[ok] mode={args.mode} finished')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
