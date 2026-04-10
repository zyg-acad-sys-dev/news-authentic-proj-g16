from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

from utils.data_utils import resolve_positive_label, write_json
from utils.eval_utils import (
    binary_curve_tables,
    classification_report_df,
    compute_classification_metrics,
    confusion_matrix_df,
    save_bar_chart,
    save_confusion_matrix_heatmap,
    save_grouped_bar_chart,
    save_histogram,
    save_training_curve,
)

IGNORE_STEMS = {'evaluation_master_predictions', 'baseline_all_predictions'}


def prediction_files(predictions_dir: Path) -> list[Path]:
    return [path for path in sorted(predictions_dir.glob('*.csv')) if path.stem not in IGNORE_STEMS]


def labels_from_df(df: pd.DataFrame) -> list[str]:
    return sorted({str(v) for v in df['true_label'].astype(str).tolist()} | {str(v) for v in df['pred_label'].astype(str).tolist()})


def extract_score_matrix(df: pd.DataFrame, labels: list[str]):
    score_cols = [f'score_{label}' for label in labels]
    if all(col in df.columns for col in score_cols):
        return df[score_cols].to_numpy(dtype=float)
    return None


def _write_training_curves(tables_dir: Path, figures_dir: Path) -> None:
    for log_path in sorted(tables_dir.glob('*_epoch_logs.csv')):
        history_df = pd.read_csv(log_path)
        if history_df.empty or 'epoch' not in history_df.columns:
            continue
        stem = log_path.stem.replace('_epoch_logs', '')
        save_training_curve(history_df, x_col='epoch', series_cols=['train_loss', 'loss', 'eval_loss'], title=f'Training Loss Curve: {stem}', ylabel='Loss', path=figures_dir / f'training_curve_loss_{stem}.png')
        save_training_curve(history_df, x_col='epoch', series_cols=['val_f1_macro', 'eval_f1_macro', 'val_accuracy', 'eval_accuracy'], title=f'Validation Curve: {stem}', ylabel='Metric', path=figures_dir / f'training_curve_metric_{stem}.png')


def load_efficiency_rows(metadata_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not metadata_dir.exists():
        return rows
    for meta_path in sorted(metadata_dir.glob('*.json')):
        data = pd.read_json(meta_path, typ='series').to_dict()
        model_name = data.get('model_name')
        if not model_name:
            continue
        for split_name in ['val', 'test']:
            key = f'{split_name}_inference'
            split_meta = data.get(key, {}) or {}
            if not isinstance(split_meta, dict):
                continue
            rows.append({
                'model_name': str(model_name),
                'split': split_name,
                'num_samples': split_meta.get('num_samples'),
                'inference_total_seconds': split_meta.get('inference_total_seconds'),
                'inference_avg_ms_per_sample': split_meta.get('inference_avg_ms_per_sample'),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description='Aggregate metrics and build report-ready figures.')
    parser.add_argument('--split-table', default='data/splits/split_table.csv')
    parser.add_argument('--predictions-dir', default='outputs/predictions')
    parser.add_argument('--text-column', default='clean_text')
    parser.add_argument('--positive-label', default='fake')
    parser.add_argument('--export-first-n', type=int, default=100, help='Export the first N test predictions for the best test model.')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    split_df = pd.read_csv(repo_root / args.split_table)
    predictions_dir = repo_root / args.predictions_dir
    tables_dir = repo_root / 'outputs' / 'tables'
    figures_dir = repo_root / 'outputs' / 'figures'
    reports_dir = repo_root / 'outputs' / 'reports'
    metadata_dir = reports_dir / 'metadata'
    curves_dir = tables_dir / 'curves'
    for folder in (tables_dir, figures_dir, reports_dir, metadata_dir, curves_dir):
        folder.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    combined_predictions = []
    resolved_positive_label = None
    for pred_path in prediction_files(predictions_dir):
        pred_df = pd.read_csv(pred_path)
        if pred_df.empty:
            continue
        labels = labels_from_df(pred_df)
        positive_label = resolve_positive_label(labels, requested=args.positive_label)
        resolved_positive_label = positive_label
        y_score = extract_score_matrix(pred_df, labels)
        metrics = compute_classification_metrics(pred_df['true_label'].astype(str).tolist(), pred_df['pred_label'].astype(str).tolist(), y_score=y_score, labels=labels, positive_label=positive_label)
        model_name = str(pred_df['model_name'].iloc[0])
        split_name = str(pred_df['split'].iloc[0])
        metric_rows.append({'model_name': model_name, 'split': split_name, 'positive_label': positive_label, **{k: v for k, v in metrics.items() if not isinstance(v, list)}})
        classification_report_df(pred_df['true_label'], pred_df['pred_label'], labels=labels).to_csv(tables_dir / f'eval_{model_name}_{split_name}_classification_report.csv', index=False)
        confusion_matrix_df(pred_df['true_label'], pred_df['pred_label'], labels=labels).to_csv(tables_dir / f'eval_{model_name}_{split_name}_confusion_matrix.csv', index=False)
        combined_predictions.append(pred_df)

        if y_score is not None and len(labels) == 2:
            score_series = pred_df[f'score_{positive_label}'] if f'score_{positive_label}' in pred_df.columns else pred_df['pred_score']
            roc_df, pr_df = binary_curve_tables(pred_df['true_label'].astype(str).tolist(), score_series.to_numpy(dtype=float), positive_label=positive_label)
            roc_df.to_csv(curves_dir / f'roc_{model_name}_{split_name}.csv', index=False)
            pr_df.to_csv(curves_dir / f'pr_{model_name}_{split_name}.csv', index=False)

    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(['split', 'f1_macro', 'accuracy'], ascending=[True, False, False])
    metrics_df.to_csv(tables_dir / 'all_metrics.csv', index=False)
    efficiency_df = pd.DataFrame(load_efficiency_rows(metadata_dir))
    if not efficiency_df.empty:
        efficiency_df = efficiency_df.sort_values(['split', 'model_name']).reset_index(drop=True)
    efficiency_df.to_csv(tables_dir / 'efficiency_metrics.csv', index=False)
    if combined_predictions:
        pd.concat(combined_predictions, ignore_index=True).to_csv(predictions_dir / 'evaluation_master_predictions.csv', index=False)

    test_df = metrics_df[metrics_df['split'] == 'test'].copy() if not metrics_df.empty else pd.DataFrame()
    comparison_cols = [c for c in ['model_name', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted', 'roc_auc', 'average_precision'] if c in test_df.columns]
    comparison_df = test_df[comparison_cols].sort_values(['f1_macro', 'accuracy'], ascending=False) if not test_df.empty else pd.DataFrame(columns=comparison_cols)
    if not comparison_df.empty and not efficiency_df.empty:
        eff_test_df = efficiency_df[efficiency_df['split'] == 'test'].copy()
        if not eff_test_df.empty:
            comparison_df = comparison_df.merge(eff_test_df[['model_name', 'num_samples', 'inference_total_seconds', 'inference_avg_ms_per_sample']], on='model_name', how='left')
    comparison_df.to_csv(tables_dir / 'model_comparison.csv', index=False)

    class_df = split_df.groupby(['split', 'label_name']).size().reset_index(name='count')
    save_grouped_bar_chart(class_df, category_col='split', series_col='label_name', value_col='count', title='Class Distribution by Split', xlabel='Split', ylabel='Count', path=figures_dir / 'class_distribution_by_split.png')
    token_lengths = split_df[args.text_column].fillna('').astype(str).str.split().map(len)
    save_histogram(token_lengths.tolist(), title=f'Token Length Distribution ({args.text_column})', xlabel='Token count', ylabel='Frequency', path=figures_dir / 'token_length_distribution.png', bins=30)

    if not comparison_df.empty:
        top_bar_df = comparison_df.head(10).copy()
        save_bar_chart(top_bar_df, x_col='model_name', y_col='f1_macro', title='Test Macro-F1 by Model', xlabel='Model', ylabel='Macro-F1', path=figures_dir / 'model_comparison_macro_f1.png')
        if 'accuracy' in top_bar_df.columns:
            save_bar_chart(top_bar_df, x_col='model_name', y_col='accuracy', title='Test Accuracy by Model', xlabel='Model', ylabel='Accuracy', path=figures_dir / 'model_comparison_accuracy.png')

    top_models = comparison_df.head(3)['model_name'].astype(str).tolist() if not comparison_df.empty else []
    roc_curves, pr_curves = [], []
    for model_name in top_models:
        roc_path = curves_dir / f'roc_{model_name}_test.csv'
        pr_path = curves_dir / f'pr_{model_name}_test.csv'
        if roc_path.exists():
            roc_df = pd.read_csv(roc_path)
            roc_curves.append({'x': roc_df['fpr'].to_numpy(dtype=float), 'y': roc_df['tpr'].to_numpy(dtype=float), 'label': model_name})
        if pr_path.exists():
            pr_df = pd.read_csv(pr_path)
            pr_curves.append({'x': pr_df['recall'].to_numpy(dtype=float), 'y': pr_df['precision'].to_numpy(dtype=float), 'label': model_name})
    if roc_curves:
        from utils.eval_utils import save_line_chart
        save_line_chart(roc_curves, title='ROC Curves (Top Test Models)', xlabel='False Positive Rate', ylabel='True Positive Rate', path=figures_dir / 'roc_curves_top_models.png')
    if pr_curves:
        from utils.eval_utils import save_line_chart
        save_line_chart(pr_curves, title='Precision-Recall Curves (Top Test Models)', xlabel='Recall', ylabel='Precision', path=figures_dir / 'pr_curves_top_models.png')

    best_model = top_models[0] if top_models else None
    if best_model is not None:
        pred_path = predictions_dir / f'{best_model}_test_predictions.csv'
        if pred_path.exists():
            pred_df = pd.read_csv(pred_path)
            labels = labels_from_df(pred_df)
            matrix = confusion_matrix(pred_df['true_label'].astype(str), pred_df['pred_label'].astype(str), labels=labels)
            save_confusion_matrix_heatmap(matrix, labels=labels, title=f'Confusion Matrix: {best_model} (test)', path=figures_dir / f'confusion_matrix_{best_model}_test.png')
            merged = pred_df.merge(split_df[['sample_id', 'title', args.text_column]], on='sample_id', how='left')
            error_cases_df = merged[merged['is_correct'] == 0].copy()
            if not error_cases_df.empty:
                error_cases_df['text_preview'] = error_cases_df[args.text_column].fillna('').astype(str).str.slice(0, 220)
                keep_cols = ['sample_id', 'true_label', 'pred_label', 'pred_score', 'model_name', 'title', 'text_preview']
                error_cases_df[keep_cols].head(50).to_csv(tables_dir / f'error_cases_{best_model}_test.csv', index=False)

            if args.export_first_n > 0:
                first_n_df = merged.copy().head(int(args.export_first_n))
                if not first_n_df.empty:
                    first_n_df['text_preview'] = first_n_df[args.text_column].fillna('').astype(str).str.slice(0, 500)
                    keep_cols = ['sample_id', 'true_label', 'pred_label', 'pred_score', 'model_name', 'title', args.text_column, 'text_preview']
                    keep_cols = [c for c in keep_cols if c in first_n_df.columns]
                    first_n_df[keep_cols].to_csv(tables_dir / f'first_{int(args.export_first_n)}_test_predictions_{best_model}.csv', index=False)

    _write_training_curves(tables_dir, figures_dir)

    write_json({'num_evaluated_runs': int(len(metrics_df)), 'num_unique_models': int(metrics_df['model_name'].nunique()) if not metrics_df.empty else 0, 'positive_label': resolved_positive_label, 'best_test_model': comparison_df.head(1).to_dict(orient='records') if not comparison_df.empty else []}, reports_dir / 'evaluation_summary.json')
    with (reports_dir / 'report_ready_summary.md').open('w', encoding='utf-8') as f:
        f.write('# Report-Ready Summary\n\n')
        f.write(f'- Evaluated runs: {len(metrics_df)}\n')
        f.write(f'- Unique models: {metrics_df["model_name"].nunique() if not metrics_df.empty else 0}\n')
        if not efficiency_df.empty:
            f.write(f'- Runs with efficiency metadata: {efficiency_df["model_name"].nunique()}\n')
        if resolved_positive_label is not None:
            f.write(f'- Positive label for ROC / PR: `{resolved_positive_label}`\n')
        if not comparison_df.empty:
            best = comparison_df.head(1).iloc[0]
            f.write(f'- Best test model: `{best["model_name"]}`\n')
            f.write(f'- Best test macro-F1: {best.get("f1_macro")}\n')
    print(f'[ok] wrote {tables_dir / "all_metrics.csv"}')
    print(f'[ok] wrote {tables_dir / "model_comparison.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
