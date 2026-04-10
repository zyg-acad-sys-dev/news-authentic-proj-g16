from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset

from .eval_utils import compute_classification_metrics, measure_inference_seconds, summarize_inference_timing
from .text_utils import basic_tokenize

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(token, self.unk_id) for token in tokens]


def build_vocab(texts, max_vocab_size: int = 30000, min_freq: int = 1) -> Vocab:
    from collections import Counter

    counter = Counter()
    for text in texts:
        counter.update(basic_tokenize(text))
    sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    selected = [token for token, freq in sorted_tokens if freq >= min_freq][: max(0, max_vocab_size - 2)]
    itos = [PAD_TOKEN, UNK_TOKEN] + selected
    stoi = {token: idx for idx, token in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


class TextSequenceDataset(Dataset):
    def __init__(self, df, text_col: str, label_to_id: dict[str, int], vocab: Vocab, max_length: int = 256):
        self.sample_ids = df['sample_id'].tolist()
        self.texts = df[text_col].fillna('').astype(str).tolist()
        self.labels = [label_to_id[str(label)] for label in df['label_name'].tolist()]
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        tokens = basic_tokenize(self.texts[idx])[: self.max_length]
        input_ids = self.vocab.encode(tokens)
        if not input_ids:
            input_ids = [self.vocab.pad_id]
        return {
            'sample_id': self.sample_ids[idx],
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'length': len(input_ids),
            'label': self.labels[idx],
        }


def make_pad_collate(pad_id: int):
    def collate(batch: list[dict]):
        sample_ids = [item['sample_id'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
        sequences = [item['input_ids'] for item in batch]
        input_ids = pad_sequence(sequences, batch_first=True, padding_value=pad_id)
        return {'sample_ids': sample_ids, 'input_ids': input_ids, 'lengths': lengths, 'labels': labels}
    return collate


class TransformerTextDataset(Dataset):
    def __init__(self, df, text_col: str, label_to_id: dict[str, int]):
        self.sample_ids = df['sample_id'].tolist()
        self.texts = df[text_col].fillna('').astype(str).tolist()
        self.labels = [label_to_id[str(label)] for label in df['label_name'].tolist()]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return {'text': self.texts[idx], 'label': self.labels[idx]}


def make_transformer_collate(tokenizer, max_length: int = 256):
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def collate(batch: list[dict]):
        texts = [item['text'] for item in batch]
        encodings = tokenizer(texts, truncation=True, max_length=max_length)
        feature_batch = [{k: v[i] for k, v in encodings.items()} for i in range(len(batch))]
        padded = data_collator(feature_batch)
        padded['labels'] = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        return padded
    return collate


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = 'mean') -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha.float())
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()


def compute_balanced_class_weights(label_ids: list[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(label_ids, dtype=torch.long), minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    total = counts.sum()
    weights = total / (num_classes * counts)
    return weights / weights.mean()


def build_sequence_loss(loss_name: str, class_weights: torch.Tensor | None = None, focal_gamma: float = 2.0):
    key = loss_name.lower().strip()
    if key in {'ce', 'cross_entropy', 'weighted_ce'}:
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    if key in {'focal', 'focal_loss'}:
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)
    raise ValueError(f'Unsupported loss name: {loss_name}')


def build_tfidf_vectorizer(ngram_range=(1, 2), min_df: int = 2, max_features: int | None = 5000, analyzer: str = 'word') -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=False, strip_accents=None, ngram_range=ngram_range, min_df=min_df, max_features=max_features, analyzer=analyzer, sublinear_tf=True)


@dataclass
class SparseTrainingArtifacts:
    pipeline: Pipeline
    validation_scores: np.ndarray
    test_scores: np.ndarray
    validation_predictions: np.ndarray
    test_predictions: np.ndarray
    validation_inference: dict[str, float | int | None]
    test_inference: dict[str, float | int | None]


def _build_estimator(model_name: str, random_state: int = 42):
    if model_name == 'logistic_regression':
        return LogisticRegression(max_iter=2000, random_state=random_state)
    if model_name == 'linear_svm':
        return CalibratedClassifierCV(estimator=LinearSVC(random_state=random_state, dual='auto'), method='sigmoid', cv=3)
    raise ValueError(f'Unsupported sparse model: {model_name}')


def _extract_scores(estimator, features) -> np.ndarray:
    if hasattr(estimator, 'predict_proba'):
        return estimator.predict_proba(features)
    if hasattr(estimator, 'decision_function'):
        return estimator.decision_function(features)
    raise RuntimeError('Estimator does not expose predict_proba or decision_function.')


def train_sparse_text_classifier(train_texts, train_labels, val_texts, test_texts, model_name: str, ngram_range=(1, 2), min_df: int = 2, max_features: int | None = 5000, analyzer: str = 'word', random_state: int = 42) -> SparseTrainingArtifacts:
    pipeline = Pipeline(steps=[
        ('vectorizer', build_tfidf_vectorizer(ngram_range=ngram_range, min_df=min_df, max_features=max_features, analyzer=analyzer)),
        ('classifier', _build_estimator(model_name=model_name, random_state=random_state)),
    ])
    pipeline.fit(train_texts, train_labels)

    vectorizer = pipeline.named_steps['vectorizer']
    classifier = pipeline.named_steps['classifier']

    val_features = vectorizer.transform(val_texts)
    test_features = vectorizer.transform(test_texts)

    val_pred_box: dict[str, np.ndarray] = {}
    val_score_box: dict[str, np.ndarray] = {}
    test_pred_box: dict[str, np.ndarray] = {}
    test_score_box: dict[str, np.ndarray] = {}

    def _run_validation():
        val_pred_box['value'] = classifier.predict(val_features)
        val_score_box['value'] = _extract_scores(classifier, val_features)

    def _run_test():
        test_pred_box['value'] = classifier.predict(test_features)
        test_score_box['value'] = _extract_scores(classifier, test_features)

    val_seconds = measure_inference_seconds(_run_validation)
    test_seconds = measure_inference_seconds(_run_test)

    return SparseTrainingArtifacts(
        pipeline=pipeline,
        validation_scores=val_score_box['value'],
        test_scores=test_score_box['value'],
        validation_predictions=val_pred_box['value'],
        test_predictions=test_pred_box['value'],
        validation_inference=summarize_inference_timing(val_seconds, len(val_texts)),
        test_inference=summarize_inference_timing(test_seconds, len(test_texts)),
    )


class RecurrentTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        rnn_type: str = 'lstm',
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        dropout_between_layers = dropout if num_layers > 1 else 0.0
        rnn_type = rnn_type.lower().strip()
        if rnn_type == 'gru':
            self.encoder = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_between_layers, bidirectional=bidirectional)
            self.rnn_type = 'gru'
        elif rnn_type == 'rnn':
            self.encoder = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, nonlinearity='tanh', dropout=dropout_between_layers, bidirectional=bidirectional)
            self.rnn_type = 'rnn'
        else:
            self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_between_layers, bidirectional=bidirectional)
            self.rnn_type = 'lstm'
        direction_multiplier = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * direction_multiplier, num_classes)
        self.bidirectional = bidirectional

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.encoder(packed)
        if self.rnn_type == 'lstm':
            hidden = hidden[0]
        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            last_hidden = hidden[-1]
        return self.classifier(self.dropout(last_hidden))


def predict_recurrent_model(model, data_loader, device: str, label_names: list[str], positive_label: str | None = None) -> dict[str, Any]:
    model.eval()
    all_sample_ids: list[str] = []
    all_true: list[str] = []
    all_pred: list[str] = []
    all_prob_rows: list[np.ndarray] = []

    def _run_prediction() -> None:
        nonlocal all_sample_ids, all_true, all_pred, all_prob_rows
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                lengths = batch['lengths'].to(device)
                labels = batch['labels']
                logits = model(input_ids, lengths)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                pred_ids = probs.argmax(axis=1)
                all_sample_ids.extend(batch['sample_ids'])
                all_true.extend([label_names[i] for i in labels.tolist()])
                all_pred.extend([label_names[i] for i in pred_ids.tolist()])
                all_prob_rows.extend(list(probs))

    inference_seconds = measure_inference_seconds(_run_prediction)
    prob_array = np.vstack(all_prob_rows) if all_prob_rows else np.zeros((0, len(label_names)))
    metrics = compute_classification_metrics(all_true, all_pred, y_score=prob_array, labels=label_names, positive_label=positive_label)
    return {
        'sample_ids': all_sample_ids,
        'y_true': all_true,
        'y_pred': all_pred,
        'y_score': prob_array,
        'metrics': metrics,
        'inference': summarize_inference_timing(inference_seconds, len(all_sample_ids)),
    }


def fit_recurrent_model(
    model,
    train_loader,
    val_loader,
    device: str,
    label_names: list[str],
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    clip_norm: float | None = 1.0,
    criterion=None,
    positive_label: str | None = None,
    train_eval_loader=None,
    test_eval_loader=None,
) -> dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = criterion or torch.nn.CrossEntropyLoss()
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)
    best_state = deepcopy(model.state_dict())
    best_metric = float('-inf')
    history: list[dict[str, Any]] = []
    train_start = perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = perf_counter()
        model.train(); running_loss = 0.0; total_examples = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            if clip_norm is not None and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            batch_size = int(labels.size(0))
            running_loss += float(loss.item()) * batch_size
            total_examples += batch_size
        train_loss = running_loss / max(total_examples, 1)

        train_metrics = {}
        if train_eval_loader is not None:
            train_eval = predict_recurrent_model(model, train_eval_loader, device=device, label_names=label_names, positive_label=positive_label)
            train_metrics = train_eval.get('metrics', {}) or {}

        val_eval = predict_recurrent_model(model, val_loader, device=device, label_names=label_names, positive_label=positive_label)
        val_metrics = val_eval.get('metrics', {}) or {}
        val_macro_f1 = val_metrics.get('f1_macro') or 0.0

        test_metrics = {}
        if test_eval_loader is not None:
            test_eval = predict_recurrent_model(model, test_eval_loader, device=device, label_names=label_names, positive_label=positive_label)
            test_metrics = test_eval.get('metrics', {}) or {}

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_metrics.get('accuracy'),
            'train_f1_macro': train_metrics.get('f1_macro'),
            'val_accuracy': val_metrics.get('accuracy'),
            'val_f1_macro': val_macro_f1,
            'val_f1_weighted': val_metrics.get('f1_weighted'),
            'test_accuracy': test_metrics.get('accuracy'),
            'test_f1_macro': test_metrics.get('f1_macro'),
            'epoch_seconds': perf_counter() - epoch_start,
        })
        if val_macro_f1 > best_metric:
            best_metric = val_macro_f1
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {'model': model, 'history': history, 'best_val_f1_macro': best_metric, 'train_time_seconds': perf_counter() - train_start}


def load_sequence_classifier(
    model_name_or_path: str,
    num_labels: int,
    *,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    proxies: dict[str, str] | None = None,
):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    load_kwargs = {
        'cache_dir': cache_dir,
        'local_files_only': local_files_only,
    }
    if proxies:
        load_kwargs['proxies'] = proxies

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        **load_kwargs,
    )
    return tokenizer, model
