from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text)).strip()


def strip_basic_html(text: str) -> str:
    return normalize_whitespace(re.sub(r'<[^>]+>', ' ', text))


def remove_urls(text: str) -> str:
    return normalize_whitespace(re.sub(r'https?://\S+|www\.\S+', ' ', text))


def basic_clean_text(text: str, lowercase: bool = True) -> str:
    text = unicodedata.normalize('NFKC', text or '')
    text = strip_basic_html(text)
    text = remove_urls(text)
    if lowercase:
        text = text.lower()
    return normalize_whitespace(text)


def merge_title_body(title: str, body: str) -> str:
    title = normalize_whitespace(title or '')
    body = normalize_whitespace(body or '')
    if title and body:
        return f'{title} {body}'
    return title or body


def basic_tokenize(text: str) -> list[str]:
    return [token for token in str(text).split() if token]
