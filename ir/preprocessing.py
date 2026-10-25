"""Text preprocessing: case folding, tokenization, stopword removal."""

import re
from typing import List

from .stopwords import STOPWORDS

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def case_folding(text: str) -> str:
    """Ubah seluruh teks menjadi huruf kecil."""
    return text.lower()


def tokenize(text: str) -> List[str]:
    """Pecah teks menjadi token alfanumerik (tanda baca dibuang)."""
    return _TOKEN_RE.findall(_URL_RE.sub(" ", text))


def remove_stopwords(tokens: List[str], min_len: int = 2) -> List[str]:
    """Buang stopword dan token yang terlalu pendek."""
    return [t for t in tokens if len(t) >= min_len and t not in STOPWORDS]


def preprocess(text: str) -> List[str]:
    """Pipeline lengkap: case folding -> tokenisasi -> stopword removal."""
    if not isinstance(text, str):
        return []
    return remove_stopwords(tokenize(case_folding(text)))


def preprocess_to_string(text: str) -> str:
    """Versi string (token digabung spasi) untuk Whoosh & CountVectorizer."""
    return " ".join(preprocess(text))
