from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.data_utils import (
    DATE_CANDIDATES,
    LABEL_CANDIDATES,
    SOURCE_CANDIDATES,
    TEXT_CANDIDATES,
    TITLE_CANDIDATES,
    coerce_label_name,
    parse_label_map,
    pick_first_column,
    stratified_train_val_test_split,
    write_json,
)
from utils.eval_utils import save_grouped_bar_chart, save_histogram
from utils.text_utils import basic_clean_text, basic_tokenize, merge_title_body


def main() -> int:
    parser = argparse.ArgumentParser(description='Prepare the fake-news dataset for training.')
    parser.add_argument('--input', default='data/raw/fake_news.csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-size', type=float, default=0.70)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument(
        '--label-map',
        default='0:real,1:fake',
        help='Comma-separated mapping used before training, e.g. 0:real,1:fake or 0:fake,1:real.',
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    input_path = repo_root / args.input
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    label_map = parse_label_map(args.label_map)
    df = pd.read_csv(input_path)
    original_rows = len(df)
    title_col = pick_first_column(df.columns, TITLE_CANDIDATES)
    text_col = pick_first_column(df.columns, TEXT_CANDIDATES)
    label_col = pick_first_column(df.columns, LABEL_CANDIDATES)
    source_col = pick_first_column(df.columns, SOURCE_CANDIDATES)
    date_col = pick_first_column(df.columns, DATE_CANDIDATES)
    if text_col is None or label_col is None:
        raise ValueError('Could not detect text and label columns in the input CSV.')

    prepared = pd.DataFrame({
        'sample_id': [f'sample_{i:06d}' for i in range(len(df))],
        'title': df[title_col].fillna('').astype(str) if title_col else '',
        'body_text': df[text_col].fillna('').astype(str),
        'label': df[label_col],
        'source': df[source_col].fillna('').astype(str) if source_col else '',
        'date': df[date_col].fillna('').astype(str) if date_col else '',
    })
    prepared['label_name'] = prepared['label'].map(lambda value: coerce_label_name(value, label_map=label_map))
    unresolved = sorted({v for v in prepared['label_name'].astype(str).unique().tolist() if v.startswith('class_')})
    if unresolved:
        raise ValueError(
            'Label semantics are still generic after applying --label-map. '
            f'Please confirm the dataset semantics. Unresolved labels: {unresolved}'
        )
    prepared = prepared[(prepared['title'].str.strip() != '') | (prepared['body_text'].str.strip() != '')].copy()
    duplicate_subset = ['body_text', 'label_name']
    if title_col is not None:
        duplicate_subset = ['title', 'body_text', 'label_name']
    prepared = prepared.drop_duplicates(subset=duplicate_subset).reset_index(drop=True)
    prepared['headline_only'] = prepared['title'].fillna('').astype(str)
    prepared['merged_text'] = [merge_title_body(t, b) for t, b in zip(prepared['title'], prepared['body_text'])]
    prepared['clean_text'] = prepared['merged_text'].map(basic_clean_text)

    train_df, val_df, test_df = stratified_train_val_test_split(
        prepared,
        label_col='label_name',
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
    )
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values('sample_id').reset_index(drop=True)
    split_df['token_count'] = split_df['clean_text'].map(lambda x: len(basic_tokenize(x)))

    processed_dir = repo_root / 'data' / 'processed'
    splits_dir = repo_root / 'data' / 'splits'
    figures_dir = repo_root / 'outputs' / 'figures'
    reports_dir = repo_root / 'outputs' / 'reports'
    tables_dir = repo_root / 'outputs' / 'tables'
    for folder in (processed_dir, splits_dir, figures_dir, reports_dir, tables_dir):
        folder.mkdir(parents=True, exist_ok=True)

    prepared.to_csv(processed_dir / 'news_text_fields.csv', index=False)
    split_df.to_csv(splits_dir / 'split_table.csv', index=False)

    class_df = split_df.groupby(['split', 'label_name']).size().reset_index(name='count')
    class_df.to_csv(tables_dir / 'dataset_class_distribution.csv', index=False)
    save_grouped_bar_chart(
        class_df,
        category_col='split',
        series_col='label_name',
        value_col='count',
        title='Class Distribution by Split',
        xlabel='Split',
        ylabel='Count',
        path=figures_dir / 'class_distribution_by_split.png',
    )
    save_histogram(
        split_df['token_count'].tolist(),
        title='Token Length Distribution (clean_text)',
        xlabel='Token count',
        ylabel='Frequency',
        path=figures_dir / 'token_length_distribution.png',
        bins=30,
    )

    write_json(
        {
            'input_file': str(input_path),
            'original_rows': int(original_rows),
            'retained_rows': int(len(prepared)),
            'detected_columns': {
                'title_col': title_col,
                'text_col': text_col,
                'label_col': label_col,
                'source_col': source_col,
                'date_col': date_col,
            },
            'label_map': label_map,
            'split_sizes': split_df['split'].value_counts().to_dict(),
            'label_distribution': split_df['label_name'].value_counts().to_dict(),
        },
        reports_dir / 'prepare_summary.json',
    )
    print(f'[ok] wrote {processed_dir / "news_text_fields.csv"}')
    print(f'[ok] wrote {splits_dir / "split_table.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
