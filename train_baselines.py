from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from utils.data_utils import resolve_positive_label, write_json
from utils.eval_utils import PREDICTION_COLUMNS, classification_report_df, compute_classification_metrics, confusion_matrix_df
from utils.train_utils import train_sparse_text_classifier


def prediction_table(sample_ids, split_name: str, y_true, y_pred, y_score, model_name: str, label_names: list[str]) -> pd.DataFrame:
    pred_score = y_score.max(axis=1) if getattr(y_score, 'ndim', 1) == 2 else y_score
    df = pd.DataFrame({
        'sample_id': sample_ids,
        'split': split_name,
        'true_label': y_true,
        'pred_label': y_pred,
        'pred_score': pred_score,
        'model_name': model_name,
        'is_correct': [int(a == b) for a, b in zip(y_true, y_pred)],
    })
    if getattr(y_score, 'ndim', 1) == 2 and y_score.shape[1] == len(label_names):
        for idx, label_name in enumerate(label_names):
            df[f'score_{label_name}'] = y_score[:, idx]
    ordered = PREDICTION_COLUMNS + [c for c in df.columns if c not in PREDICTION_COLUMNS]
    return df[ordered]


def main() -> int:
    parser = argparse.ArgumentParser(description='Train TF-IDF baselines.')
    parser.add_argument('--input', default='data/splits/split_table.csv')
    parser.add_argument('--text-column', default='clean_text')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models', nargs='*', default=['logistic_regression', 'linear_svm'])
    parser.add_argument('--max-features', type=int, default=5000)
    parser.add_argument('--min-df', type=int, default=2)
    parser.add_argument('--positive-label', default='fake')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    df = pd.read_csv(repo_root / args.input)
    label_names = sorted(df['label_name'].astype(str).unique().tolist())
    positive_label = resolve_positive_label(label_names, requested=args.positive_label)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    models_dir = repo_root / 'outputs' / 'models' / 'baselines'
    preds_dir = repo_root / 'outputs' / 'predictions'
    tables_dir = repo_root / 'outputs' / 'tables'
    reports_dir = repo_root / 'outputs' / 'reports'
    metadata_dir = reports_dir / 'metadata'
    for folder in (models_dir, preds_dir, tables_dir, reports_dir, metadata_dir):
        folder.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for model_name in args.models:
        artifacts = train_sparse_text_classifier(
            train_texts=train_df[args.text_column].fillna('').astype(str).tolist(),
            train_labels=train_df['label_name'].astype(str).tolist(),
            val_texts=val_df[args.text_column].fillna('').astype(str).tolist(),
            test_texts=test_df[args.text_column].fillna('').astype(str).tolist(),
            model_name=model_name,
            max_features=args.max_features,
            min_df=args.min_df,
            random_state=args.seed,
        )
        joblib.dump(artifacts.pipeline, models_dir / f'{model_name}_{args.text_column}.joblib')

        val_pred_df = prediction_table(val_df['sample_id'].tolist(), 'val', val_df['label_name'].astype(str).tolist(), artifacts.validation_predictions.tolist(), artifacts.validation_scores, model_name, label_names)
        test_pred_df = prediction_table(test_df['sample_id'].tolist(), 'test', test_df['label_name'].astype(str).tolist(), artifacts.test_predictions.tolist(), artifacts.test_scores, model_name, label_names)
        val_pred_df.to_csv(preds_dir / f'{model_name}_{args.text_column}_val_predictions.csv', index=False)
        test_pred_df.to_csv(preds_dir / f'{model_name}_{args.text_column}_test_predictions.csv', index=False)

        split_inference = {
            'val': artifacts.validation_inference,
            'test': artifacts.test_inference,
        }
        for split_name, split_df, pred_labels, score_matrix in [
            ('val', val_df, artifacts.validation_predictions.tolist(), artifacts.validation_scores),
            ('test', test_df, artifacts.test_predictions.tolist(), artifacts.test_scores),
        ]:
            metrics = compute_classification_metrics(
                y_true=split_df['label_name'].astype(str).tolist(),
                y_pred=pred_labels,
                y_score=score_matrix,
                labels=label_names,
                positive_label=positive_label,
            )
            metrics_rows.append({'model_name': model_name, 'split': split_name, 'text_column': args.text_column, 'max_features': args.max_features, **split_inference[split_name], **metrics})
            if split_name == 'test':
                classification_report_df(split_df['label_name'].astype(str).tolist(), pred_labels, labels=label_names).to_csv(tables_dir / f'{model_name}_{args.text_column}_classification_report.csv', index=False)
                confusion_matrix_df(split_df['label_name'].astype(str).tolist(), pred_labels, labels=label_names).to_csv(tables_dir / f'{model_name}_{args.text_column}_confusion_matrix.csv', index=False)

        write_json({
            'model_name': model_name,
            'text_column': args.text_column,
            'max_features': args.max_features,
            'min_df': args.min_df,
            'positive_label': positive_label,
            'val_inference': artifacts.validation_inference,
            'test_inference': artifacts.test_inference,
        }, metadata_dir / f'{model_name}_{args.text_column}.run_metadata.json')

    pd.DataFrame(metrics_rows).to_csv(tables_dir / f'baseline_metrics_{args.text_column}.csv', index=False)
    write_json({'text_column': args.text_column, 'models': args.models, 'max_features': args.max_features, 'min_df': args.min_df, 'positive_label': positive_label, 'metrics_file': str(tables_dir / f'baseline_metrics_{args.text_column}.csv')}, reports_dir / 'baseline_summary.json')
    print(f'[ok] wrote {tables_dir / f"baseline_metrics_{args.text_column}.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
