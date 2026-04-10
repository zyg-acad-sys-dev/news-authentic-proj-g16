from __future__ import annotations

import os
import shutil
from datetime import datetime

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_utils import resolve_positive_label, write_json
from utils.eval_utils import PREDICTION_COLUMNS, classification_report_df, confusion_matrix_df, compute_classification_metrics
from utils.train_utils import TransformerTextDataset, load_sequence_classifier, make_transformer_collate


def safe_model_slug(model_name: str) -> str:
    return model_name.replace('/', '__').replace('-', '_')




def to_repo_path(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else (repo_root / p)


def infer_hf_proxies_from_env() -> dict[str, str] | None:
    http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    proxies = {}
    if http_proxy:
        proxies['http'] = http_proxy
    if https_proxy:
        proxies['https'] = https_proxy
    return proxies or None


def resolve_model_source(repo_root: Path, model_name: str, local_model_dir: str | None = None) -> tuple[str, bool, str]:
    explicit_dir = to_repo_path(repo_root, local_model_dir)
    if explicit_dir and explicit_dir.exists():
        return str(explicit_dir), True, 'explicit_local_dir'

    bundled_dir = repo_root / 'assets' / 'hf' / safe_model_slug(model_name)
    if bundled_dir.exists():
        return str(bundled_dir), True, 'bundled_local_dir'

    return model_name, False, 'hub'



def softmax_np(logits: np.ndarray) -> np.ndarray:
    # Use NumPy here so metric code does not depend on a tensor context.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

def predictions_to_df(sample_ids, split_name: str, y_true, y_pred, y_score, model_name: str, label_names: list[str]) -> pd.DataFrame:
    slug = safe_model_slug(model_name)
    df = pd.DataFrame({
        'sample_id': sample_ids,
        'split': split_name,
        'true_label': y_true,
        'pred_label': y_pred,
        'pred_score': y_score.max(axis=1),
        'model_name': slug,
        'is_correct': [int(a == b) for a, b in zip(y_true, y_pred)],
    })
    for idx, label_name in enumerate(label_names):
        df[f'score_{label_name}'] = y_score[:, idx]
    ordered = PREDICTION_COLUMNS + [c for c in df.columns if c not in PREDICTION_COLUMNS]
    return df[ordered]


def main() -> int:
    parser = argparse.ArgumentParser(description='Fine-tune one BERT-family model with Hugging Face Trainer.')
    parser.add_argument('--input', default='data/splits/split_table.csv')
    parser.add_argument('--text-column', default='clean_text')
    parser.add_argument('--model-name', default='bert-base-uncased', help='English pretrained model for the current dataset, e.g. bert-base-uncased or roberta-base.')
    parser.add_argument('--local-model-dir', default='', help='Optional local pretrained model directory. If omitted, the script will also check assets/hf/<model-slug> before trying the Hugging Face Hub.')
    parser.add_argument('--cache-dir', default='', help='Optional Hugging Face cache directory.')
    parser.add_argument('--local-files-only', action='store_true', help='Only load local files and never attempt to reach the Hugging Face Hub.')
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio relative to the total training steps.')
    parser.add_argument('--warmup-steps', type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--positive-label', default='fake')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cleanup-trainer-tmp', action='store_true', help='Remove the temporary Trainer output directory after predictions and artifacts are saved.')
    args = parser.parse_args()

    try:
        import torch
        from transformers import Trainer, TrainingArguments, set_seed
    except Exception as exc:
        raise RuntimeError('train_bert.py requires transformers and torch.') from exc

    set_seed(args.seed)

    repo_root = Path(__file__).resolve().parent
    df = pd.read_csv(repo_root / args.input)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    label_names = sorted(df['label_name'].astype(str).unique().tolist())
    positive_label = resolve_positive_label(label_names, requested=args.positive_label)
    label_to_id = {label: idx for idx, label in enumerate(label_names)}

    model_source, auto_local_only, source_mode = resolve_model_source(repo_root, args.model_name, args.local_model_dir)
    cache_dir = str(to_repo_path(repo_root, args.cache_dir)) if args.cache_dir else None
    local_files_only = bool(args.local_files_only or auto_local_only or os.environ.get('HF_HUB_OFFLINE') == '1' or os.environ.get('TRANSFORMERS_OFFLINE') == '1')
    proxies = infer_hf_proxies_from_env()

    print(f'[info] BERT source mode: {source_mode}')
    print(f'[info] Loading pretrained source: {model_source}')
    if local_files_only:
        print('[info] local_files_only=True')

    tokenizer, model = load_sequence_classifier(
        model_name_or_path=model_source,
        num_labels=len(label_names),
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        proxies=proxies,
    )
    collate_fn = make_transformer_collate(tokenizer, max_length=args.max_length)
    train_ds = TransformerTextDataset(train_df, text_col=args.text_column, label_to_id=label_to_id)
    val_ds = TransformerTextDataset(val_df, text_col=args.text_column, label_to_id=label_to_id)
    test_ds = TransformerTextDataset(test_df, text_col=args.text_column, label_to_id=label_to_id)

    output_root = repo_root / 'outputs'
    models_dir = output_root / 'models' / 'bert'
    preds_dir = output_root / 'predictions'
    tables_dir = output_root / 'tables'
    reports_dir = output_root / 'reports'
    metadata_dir = reports_dir / 'metadata'
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer_dir = output_root / 'trainer_tmp' / f"{safe_model_slug(args.model_name)}_{run_id}"

    for folder in (models_dir, preds_dir, tables_dir, reports_dir, metadata_dir, trainer_dir):
        folder.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = max(1, int(np.ceil(len(train_ds) / max(1, args.batch_size))))
    total_training_steps = max(1, steps_per_epoch * int(args.epochs))
    explicit_warmup_steps = None
    if args.warmup_steps is not None:
        explicit_warmup_steps = int(args.warmup_steps)
        if explicit_warmup_steps <= 1:
            warmup_ratio = float(args.warmup_steps)
            explicit_warmup_steps = None
        else:
            warmup_ratio = explicit_warmup_steps / total_training_steps
    else:
        warmup_ratio = float(args.warmup_ratio)

    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f'warmup_ratio must be between 0 and 1, got {warmup_ratio}')

    # Convert the user-facing ratio into an integer step count for Trainer.
    warmup_steps = explicit_warmup_steps if explicit_warmup_steps is not None else int(total_training_steps * warmup_ratio)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = softmax_np(np.asarray(logits))
        pred_ids = probs.argmax(axis=1)
        y_true = [label_names[idx] for idx in labels.tolist()]
        y_pred = [label_names[idx] for idx in pred_ids.tolist()]
        metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_score=probs, labels=label_names, positive_label=positive_label)
        return {k: v for k, v in metrics.items() if isinstance(v, (float, int)) and v is not None}

    training_args = TrainingArguments(
        output_dir=str(trainer_dir),
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1_macro',
        greater_is_better=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        logging_strategy='epoch',
        save_total_limit=1,
        seed=args.seed,
        report_to='none',
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    device_name = str(trainer.args.device)
    train_start = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - train_start

    from utils.eval_utils import measure_inference_seconds, summarize_inference_timing

    val_box = {}
    test_box = {}

    def _predict_val():
        val_box['output'] = trainer.predict(val_ds)

    def _predict_test():
        test_box['output'] = trainer.predict(test_ds)

    val_seconds = measure_inference_seconds(_predict_val)
    test_seconds = measure_inference_seconds(_predict_test)
    val_output = val_box['output']
    test_output = test_box['output']
    val_probs = softmax_np(np.asarray(val_output.predictions))
    test_probs = softmax_np(np.asarray(test_output.predictions))
    val_pred_ids = val_probs.argmax(axis=1)
    test_pred_ids = test_probs.argmax(axis=1)
    val_true = [label_names[idx] for idx in val_output.label_ids.tolist()]
    val_pred = [label_names[idx] for idx in val_pred_ids.tolist()]
    test_true = [label_names[idx] for idx in test_output.label_ids.tolist()]
    test_pred = [label_names[idx] for idx in test_pred_ids.tolist()]
    test_metrics = compute_classification_metrics(test_true, test_pred, y_score=test_probs, labels=label_names, positive_label=positive_label)

    save_dir = models_dir / safe_model_slug(args.model_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(save_dir)
    run_metadata = {
        **vars(args),
        'model_name': safe_model_slug(args.model_name),
        'warmup_ratio': warmup_ratio,
        'warmup_steps': warmup_steps,
        'total_training_steps': total_training_steps,
        'requested_model_name': args.model_name,
        'resolved_model_source': model_source,
        'source_mode': source_mode,
        'local_files_only': local_files_only,
        'cache_dir': cache_dir,
        'device': device_name,
        'train_time_seconds': train_seconds,
        'positive_label': positive_label,
        'val_inference': summarize_inference_timing(val_seconds, len(val_ds)),
        'test_inference': summarize_inference_timing(test_seconds, len(test_ds)),
    }
    write_json(run_metadata, save_dir / 'run_metadata.json')
    write_json(run_metadata, metadata_dir / f'{safe_model_slug(args.model_name)}.run_metadata.json')

    predictions_to_df(val_ds.sample_ids, 'val', val_true, val_pred, val_probs, args.model_name, label_names).to_csv(preds_dir / f'{safe_model_slug(args.model_name)}_val_predictions.csv', index=False)
    predictions_to_df(test_ds.sample_ids, 'test', test_true, test_pred, test_probs, args.model_name, label_names).to_csv(preds_dir / f'{safe_model_slug(args.model_name)}_test_predictions.csv', index=False)
    pd.DataFrame(trainer.state.log_history).to_csv(tables_dir / f'{safe_model_slug(args.model_name)}_epoch_logs.csv', index=False)
    classification_report_df(test_true, test_pred, labels=label_names).to_csv(tables_dir / f'{safe_model_slug(args.model_name)}_classification_report.csv', index=False)
    confusion_matrix_df(test_true, test_pred, labels=label_names).to_csv(tables_dir / f'{safe_model_slug(args.model_name)}_confusion_matrix.csv', index=False)
    write_json({'text_column': args.text_column, 'model_name': args.model_name, 'device': device_name, 'epochs': args.epochs, 'positive_label': positive_label, 'train_time_seconds': train_seconds, 'warmup_ratio': warmup_ratio, 'warmup_steps': warmup_steps, 'total_training_steps': total_training_steps, 'best_test_metrics': test_metrics}, reports_dir / 'bert_summary.json')
    if args.cleanup_trainer_tmp:
        shutil.rmtree(trainer_dir, ignore_errors=True)
    print(f'[ok] wrote {preds_dir / f"{safe_model_slug(args.model_name)}_test_predictions.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
