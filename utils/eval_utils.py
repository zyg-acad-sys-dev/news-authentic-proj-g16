from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

PREDICTION_COLUMNS = [
    'sample_id',
    'split',
    'true_label',
    'pred_label',
    'pred_score',
    'model_name',
    'is_correct',
]


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def maybe_sync_torch_device() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
            except Exception:
                pass
    except Exception:
        pass


def measure_inference_seconds(predict_fn: Callable[[], Any]) -> float:
    maybe_sync_torch_device()
    start = perf_counter()
    predict_fn()
    maybe_sync_torch_device()
    return perf_counter() - start


def summarize_inference_timing(total_seconds: float, num_samples: int) -> dict[str, float | int | None]:
    total = _safe_float(total_seconds)
    n = int(num_samples)
    avg_ms = None
    if total is not None and n > 0:
        avg_ms = _safe_float((total / n) * 1000.0)
    return {
        'num_samples': n,
        'inference_total_seconds': total,
        'inference_avg_ms_per_sample': avg_ms,
    }


def compute_classification_metrics(y_true, y_pred, y_score=None, labels=None, positive_label: str | None = None) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    unique_labels = labels or sorted({str(v) for v in y_true_arr.tolist()})
    metric_dict: dict[str, Any] = {
        'accuracy': _safe_float(accuracy_score(y_true_arr, y_pred_arr)),
        'precision_macro': _safe_float(precision_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)),
        'recall_macro': _safe_float(recall_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)),
        'f1_macro': _safe_float(f1_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)),
        'f1_weighted': _safe_float(f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)),
        'labels': unique_labels,
    }
    if y_score is not None:
        y_score = np.asarray(y_score)
        try:
            if len(unique_labels) == 2:
                resolved_positive = positive_label if positive_label in unique_labels else unique_labels[1]
                y_true_binary = (y_true_arr == resolved_positive).astype(int)
                positive_index = unique_labels.index(resolved_positive) if y_score.ndim == 2 else 0
                positive_scores = y_score[:, positive_index] if y_score.ndim == 2 else y_score
                metric_dict['positive_label'] = resolved_positive
                metric_dict['roc_auc'] = _safe_float(roc_auc_score(y_true_binary, positive_scores))
                metric_dict['average_precision'] = _safe_float(average_precision_score(y_true_binary, positive_scores))
            elif y_score.ndim == 2 and y_score.shape[1] == len(unique_labels):
                y_true_bin = label_binarize(y_true_arr, classes=unique_labels)
                metric_dict['roc_auc_ovr_macro'] = _safe_float(roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr'))
                metric_dict['average_precision_macro'] = _safe_float(average_precision_score(y_true_bin, y_score, average='macro'))
        except Exception:
            pass
    return metric_dict


def classification_report_df(y_true, y_pred, labels=None) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    rows = []
    for label_name, values in report.items():
        if isinstance(values, dict):
            rows.append({'label': label_name, **{k: _safe_float(v) for k, v in values.items()}})
        else:
            rows.append({'label': label_name, 'value': _safe_float(values)})
    return pd.DataFrame(rows)


def confusion_matrix_df(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, true_label in enumerate(labels):
        row = {'true_label': true_label}
        for j, pred_label in enumerate(labels):
            row[f'pred_{pred_label}'] = int(matrix[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_histogram(values: Iterable[float], title: str, xlabel: str, ylabel: str, path: str | Path, bins: int = 30) -> Path:
    path = ensure_parent(path)
    arr = np.asarray(list(values), dtype=float)
    plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_grouped_bar_chart(df: pd.DataFrame, category_col: str, series_col: str, value_col: str, title: str, xlabel: str, ylabel: str, path: str | Path) -> Path:
    path = ensure_parent(path)
    pivot = df.pivot(index=category_col, columns=series_col, values=value_col).fillna(0)
    x = np.arange(len(pivot.index))
    names = pivot.columns.astype(str).tolist()
    width = 0.8 / max(len(names), 1)
    plt.figure(figsize=(10, 5.5))
    for idx, name in enumerate(names):
        plt.bar(x + idx * width - (len(names) - 1) * width / 2, pivot[name].to_numpy(dtype=float), width=width, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x, pivot.index.astype(str).tolist(), rotation=20, ha='right')
    if len(names) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, path: str | Path, rotate_xticks: int = 25) -> Path:
    path = ensure_parent(path)
    plt.figure(figsize=(10, 5.5))
    plt.bar(df[x_col].astype(str), df[y_col].astype(float))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate_xticks, ha='right')
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_line_chart(curves: list[dict], title: str, xlabel: str, ylabel: str, path: str | Path) -> Path:
    path = ensure_parent(path)
    plt.figure(figsize=(8, 6))
    for curve in curves:
        plt.plot(curve['x'], curve['y'], label=curve.get('label', 'curve'))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if curves:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_confusion_matrix_heatmap(matrix: np.ndarray, labels: list[str], title: str, path: str | Path) -> Path:
    path = ensure_parent(path)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(matrix.shape[1]), yticks=np.arange(matrix.shape[0]), xticklabels=labels, yticklabels=labels, title=title, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
    thresh = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(int(matrix[i, j])), ha='center', va='center', color='white' if matrix[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_training_curve(history_df: pd.DataFrame, x_col: str, series_cols: list[str], title: str, ylabel: str, path: str | Path) -> Path:
    curves = []
    if x_col not in history_df.columns:
        return ensure_parent(path)
    for col in series_cols:
        if col in history_df.columns:
            numeric = pd.to_numeric(history_df[col], errors='coerce')
            mask = numeric.notna()
            if mask.any():
                x_values = pd.to_numeric(history_df.loc[mask, x_col], errors='coerce').to_numpy(dtype=float)
                y_values = numeric.loc[mask].to_numpy(dtype=float)
                curves.append({'x': x_values, 'y': y_values, 'label': col})
    if not curves:
        return ensure_parent(path)
    return save_line_chart(curves=curves, title=title, xlabel=x_col, ylabel=ylabel, path=path)


def binary_curve_tables(y_true, y_score, positive_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true_binary = (pd.Series(y_true).astype(str) == positive_label).astype(int).to_numpy()
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true_binary = y_true_binary[mask]
    y_score = y_score[mask]
    if y_score.size == 0 or y_true_binary.min() == y_true_binary.max():
        return pd.DataFrame(columns=['fpr', 'tpr', 'threshold']), pd.DataFrame(columns=['recall', 'precision', 'threshold'])
    fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_score)
    precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, y_score)
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': np.append(roc_thresholds, np.nan)[:len(fpr)]})
    pr_df = pd.DataFrame({'recall': recall, 'precision': precision, 'threshold': np.append(pr_thresholds, np.nan)[:len(recall)]})
    return roc_df, pr_df
