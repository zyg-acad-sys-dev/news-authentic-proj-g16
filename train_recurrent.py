from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.data_utils import resolve_positive_label, write_json
from utils.eval_utils import PREDICTION_COLUMNS, classification_report_df, confusion_matrix_df
from utils.train_utils import (
    RecurrentTextClassifier,
    TextSequenceDataset,
    build_sequence_loss,
    build_vocab,
    compute_balanced_class_weights,
    fit_recurrent_model,
    make_pad_collate,
    predict_recurrent_model,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_torch_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def parse_model_alias(name: str) -> tuple[str, bool]:
    key = name.lower().strip()
    mapping = {
        'rnn': ('rnn', False),
        'gru': ('gru', False),
        'lstm': ('lstm', False),
        'bigru': ('gru', True),
        'bilstm': ('lstm', True),
    }
    if key not in mapping:
        raise ValueError(f'Unsupported recurrent model: {name}')
    return mapping[key]




def infer_best_epoch(history: list[dict], best_val_f1_macro: float | None) -> int | None:
    if best_val_f1_macro is None:
        return None
    for row in history:
        if row.get('val_f1_macro') == best_val_f1_macro:
            return int(row.get('epoch'))
    return None

def predictions_to_df(sample_ids, split_name: str, y_true, y_pred, y_score, model_name: str, label_names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({
        'sample_id': sample_ids,
        'split': split_name,
        'true_label': y_true,
        'pred_label': y_pred,
        'pred_score': y_score.max(axis=1),
        'model_name': model_name,
        'is_correct': [int(a == b) for a, b in zip(y_true, y_pred)],
    })
    for idx, label_name in enumerate(label_names):
        df[f'score_{label_name}'] = y_score[:, idx]
    ordered = PREDICTION_COLUMNS + [c for c in df.columns if c not in PREDICTION_COLUMNS]
    return df[ordered]


def main() -> int:
    parser = argparse.ArgumentParser(description='Train a recurrent text classifier.')
    parser.add_argument('--input', default='data/splits/split_table.csv')
    parser.add_argument('--text-column', default='clean_text')
    parser.add_argument('--model', default='lstm', choices=['rnn', 'gru', 'lstm', 'bigru', 'bilstm'])
    parser.add_argument('--loss-type', default='cross_entropy', choices=['cross_entropy', 'weighted_ce', 'focal_loss'])
    parser.add_argument('--max-vocab-size', type=int, default=30000)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--positive-label', default='fake')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-tag', default='', help='Optional suffix added to output filenames so sweep runs do not overwrite each other.')
    args = parser.parse_args()

    set_global_seed(args.seed)
    repo_root = Path(__file__).resolve().parent
    df = pd.read_csv(repo_root / args.input)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    label_names = sorted(df['label_name'].astype(str).unique().tolist())
    positive_label = resolve_positive_label(label_names, requested=args.positive_label)
    label_to_id = {label: idx for idx, label in enumerate(label_names)}

    vocab = build_vocab(train_df[args.text_column].fillna('').astype(str).tolist(), max_vocab_size=args.max_vocab_size)
    train_ds = TextSequenceDataset(train_df, text_col=args.text_column, label_to_id=label_to_id, vocab=vocab, max_length=args.max_length)
    val_ds = TextSequenceDataset(val_df, text_col=args.text_column, label_to_id=label_to_id, vocab=vocab, max_length=args.max_length)
    test_ds = TextSequenceDataset(test_df, text_col=args.text_column, label_to_id=label_to_id, vocab=vocab, max_length=args.max_length)
    collate_fn = make_pad_collate(vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    rnn_type, bidirectional = parse_model_alias(args.model)
    model = RecurrentTextClassifier(
        vocab_size=len(vocab.itos),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(label_names),
        rnn_type=rnn_type,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=bidirectional,
        pad_id=vocab.pad_id,
    )

    class_weights = None
    if args.loss_type in {'weighted_ce', 'focal_loss'}:
        class_weights = compute_balanced_class_weights(train_ds.labels, len(label_names))
    criterion = build_sequence_loss(args.loss_type, class_weights=class_weights, focal_gamma=2.0)
    device = pick_torch_device()
    fit_result = fit_recurrent_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        label_names=label_names,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        clip_norm=1.0,
        criterion=criterion,
        positive_label=positive_label,
        train_eval_loader=train_eval_loader,
        test_eval_loader=test_loader,
    )
    trained_model = fit_result['model']
    best_val_f1_macro = fit_result.get('best_val_f1_macro')
    best_epoch = infer_best_epoch(fit_result.get('history', []), best_val_f1_macro)
    val_eval = predict_recurrent_model(trained_model, val_loader, device=device, label_names=label_names, positive_label=positive_label)
    test_eval = predict_recurrent_model(trained_model, test_loader, device=device, label_names=label_names, positive_label=positive_label)

    model_slug = f'{args.model}_{args.loss_type}'.replace('cross_entropy', 'ce')
    if args.run_tag.strip():
        safe_tag = args.run_tag.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
        model_slug = f'{model_slug}_{safe_tag}'
    models_dir = repo_root / 'outputs' / 'models' / 'recurrent'
    preds_dir = repo_root / 'outputs' / 'predictions'
    tables_dir = repo_root / 'outputs' / 'tables'
    reports_dir = repo_root / 'outputs' / 'reports'
    metadata_dir = reports_dir / 'metadata'
    for folder in (models_dir, preds_dir, tables_dir, reports_dir, metadata_dir):
        folder.mkdir(parents=True, exist_ok=True)

    torch.save({'model_state_dict': trained_model.state_dict(), 'model_name': model_slug, 'label_names': label_names, 'vocab': vocab.stoi, 'config': vars(args)}, models_dir / f'{model_slug}.pt')
    predictions_to_df(val_eval['sample_ids'], 'val', val_eval['y_true'], val_eval['y_pred'], val_eval['y_score'], model_slug, label_names).to_csv(preds_dir / f'{model_slug}_val_predictions.csv', index=False)
    predictions_to_df(test_eval['sample_ids'], 'test', test_eval['y_true'], test_eval['y_pred'], test_eval['y_score'], model_slug, label_names).to_csv(preds_dir / f'{model_slug}_test_predictions.csv', index=False)
    pd.DataFrame(fit_result['history']).to_csv(tables_dir / f'{model_slug}_epoch_logs.csv', index=False)
    classification_report_df(test_eval['y_true'], test_eval['y_pred'], labels=label_names).to_csv(tables_dir / f'{model_slug}_classification_report.csv', index=False)
    confusion_matrix_df(test_eval['y_true'], test_eval['y_pred'], labels=label_names).to_csv(tables_dir / f'{model_slug}_confusion_matrix.csv', index=False)
    run_metadata = {
        'model_name': model_slug,
        'text_column': args.text_column,
        'device': device,
        'epochs': args.epochs,
        'vocab_size': len(vocab.itos),
        'positive_label': positive_label,
        'train_time_seconds': fit_result['train_time_seconds'],
        # Persist the validation-best checkpoint summary for later reporting.
        'best_epoch': best_epoch,
        'best_val_f1_macro': best_val_f1_macro,
        'val_inference': val_eval.get('inference', {}),
        'test_inference': test_eval.get('inference', {}),
        'best_test_metrics': test_eval['metrics'],
    }
    write_json(run_metadata, metadata_dir / f'{model_slug}.run_metadata.json')
    write_json({'text_column': args.text_column, 'model': model_slug, 'device': device, 'epochs': args.epochs, 'vocab_size': len(vocab.itos), 'positive_label': positive_label, 'train_time_seconds': fit_result['train_time_seconds'], 'best_epoch': best_epoch, 'best_val_f1_macro': best_val_f1_macro, 'best_test_metrics': test_eval['metrics']}, reports_dir / 'recurrent_summary.json')
    print(f'[ok] wrote {preds_dir / f"{model_slug}_test_predictions.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
