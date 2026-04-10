from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

TITLE_CANDIDATES = ["title", "headline", "news_title"]
TEXT_CANDIDATES = ["text", "body", "content", "article", "news_text"]
LABEL_CANDIDATES = ["label", "target", "class", "y"]
SOURCE_CANDIDATES = ["source", "publisher", "domain"]
DATE_CANDIDATES = ["date", "publish_date", "published_at"]


def pick_first_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lowered = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def parse_label_map(spec: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not spec:
        return mapping
    for chunk in str(spec).split(','):
        item = chunk.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f'Invalid label-map item: {item}. Expected KEY:VALUE.')
        key, value = item.split(':', 1)
        mapping[key.strip().lower()] = value.strip().lower()
    return mapping


def coerce_label_name(value: object, label_map: dict[str, str] | None = None) -> str:
    text = str(value).strip().lower()
    mapping = {
        'false': 'fake',
        'true': 'real',
        'fake': 'fake',
        'real': 'real',
    }
    if label_map:
        mapping.update({str(k).strip().lower(): str(v).strip().lower() for k, v in label_map.items()})
    if text in mapping:
        return mapping[text]
    if text in {'0', '1'}:
        return f'class_{text}'
    return text


def resolve_positive_label(labels: list[str], requested: str | None = 'fake') -> str:
    lowered = {str(label).lower(): str(label) for label in labels}
    if requested and requested.lower() in lowered:
        return lowered[requested.lower()]
    for candidate in ['fake', 'positive', 'true', '1', 'yes', 'real']:
        if candidate in lowered:
            return lowered[candidate]
    return str(labels[-1])


def stratified_train_val_test_split(
    df: pd.DataFrame,
    label_col: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if round(train_size + val_size + test_size, 5) != 1.0:
        raise ValueError('train_size + val_size + test_size must equal 1.0')

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        random_state=random_state,
        stratify=df[label_col],
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=random_state,
        stratify=temp_df[label_col],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def write_json(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
