"""Pembentukan index: Whoosh (inverted index) + matriks BoW (CountVectorizer)."""

import os
import pickle
import shutil
import time

from scipy import sparse  # dependensi bawaan scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
from whoosh import index
from whoosh.analysis import RegexTokenizer
from whoosh.fields import ID, NUMERIC, STORED, TEXT, Schema

from .loader import load_all
from .preprocessing import preprocess_to_string

MATRIX_FILE = "bow_matrix.npz"
VECTORIZER_FILE = "vectorizer.pkl"


def split_tokens(text: str):
    """Tokenizer CountVectorizer: teks sudah dipreproses, cukup split spasi."""
    return text.split()


def make_schema() -> Schema:
    return Schema(
        row=NUMERIC(int, stored=True),                  # indeks baris di matriks BoW
        dataset=ID(stored=True),
        title=TEXT(stored=True),
        snippet=STORED,
        source=STORED,
        content=TEXT(analyzer=RegexTokenizer(r"\S+")),  # teks terpreproses
    )


def build_index(datasets_dir: str, index_dir: str,
                limit_per_dataset=None, log=print) -> int:
    """Muat dataset -> preprocessing -> Whoosh index + BoW. Return jumlah dokumen."""
    t0 = time.time()
    log("[1/4] Memuat dataset...")
    records = load_all(datasets_dir, limit_per_dataset, log)
    if not records:
        raise RuntimeError("Tidak ada dokumen yang berhasil dimuat. "
                           "Periksa folder datasets/.")

    log(f"[2/4] Preprocessing {len(records)} dokumen...")
    docs, kept = [], []
    for r in records:
        processed = preprocess_to_string(f"{r['title']} {r['text']}")
        if processed:
            docs.append(processed)
            kept.append(r)

    log("[3/4] Membangun matriks Bag of Words (CountVectorizer)...")
    vectorizer = CountVectorizer(tokenizer=split_tokens, token_pattern=None,
                                 lowercase=False)
    matrix = vectorizer.fit_transform(docs).tocsr()

    log("[4/4] Membangun index Whoosh...")
    if os.path.isdir(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir)
    ix = index.create_in(index_dir, make_schema())
    writer = ix.writer(limitmb=256)
    for i, (r, processed) in enumerate(zip(kept, docs)):
        writer.add_document(
            row=i, dataset=r["dataset"], title=r["title"],
            snippet=" ".join(r["text"].split())[:300],
            source=r["source"], content=processed,
        )
    writer.commit()

    sparse.save_npz(os.path.join(index_dir, MATRIX_FILE), matrix)
    with open(os.path.join(index_dir, VECTORIZER_FILE), "wb") as f:
        pickle.dump(vectorizer, f)

    log(f"Selesai: {len(kept)} dokumen, {len(vectorizer.vocabulary_)} term "
        f"({time.time() - t0:.1f} detik).")
    return len(kept)
