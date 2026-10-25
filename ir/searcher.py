"""Pencarian: kandidat dari Whoosh, ranking akhir dengan cosine similarity."""

import os
import pickle
from typing import Dict, List, Optional

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from whoosh import index
from whoosh.query import And, Or, Term

from .indexer import MATRIX_FILE, VECTORIZER_FILE, split_tokens  # noqa: F401 (untuk unpickle)
from .preprocessing import preprocess


class Searcher:
    def __init__(self, index_dir: str):
        if not index.exists_in(index_dir):
            raise FileNotFoundError(
                f"Index tidak ditemukan di '{index_dir}'. Jalankan menu [1] dulu.")
        self.ix = index.open_dir(index_dir)
        self.matrix = sparse.load_npz(os.path.join(index_dir, MATRIX_FILE)).tocsr()
        with open(os.path.join(index_dir, VECTORIZER_FILE), "rb") as f:
            self.vectorizer = pickle.load(f)

    def search(self, query: str, top_k: int = 5, candidates: int = 2000,
               dataset: Optional[str] = None) -> List[Dict]:
        """Preprocess query -> kandidat Whoosh -> ranking cosine similarity."""
        tokens = preprocess(query)
        if not tokens:
            return []

        q = Or([Term("content", t) for t in dict.fromkeys(tokens)])
        if dataset:
            q = And([q, Term("dataset", dataset)])

        with self.ix.searcher() as s:
            hits = s.search(q, limit=candidates)
            found = [dict(h.fields()) for h in hits]
        if not found:
            return []

        rows = [h["row"] for h in found]
        qvec = self.vectorizer.transform([" ".join(tokens)])
        sims = cosine_similarity(qvec, self.matrix[rows]).ravel()

        order = sims.argsort()[::-1][:top_k]
        results = []
        for rank, idx in enumerate(order, start=1):
            doc = found[idx]
            doc["score"] = float(sims[idx])
            doc["rank"] = rank
            results.append(doc)
        return results
