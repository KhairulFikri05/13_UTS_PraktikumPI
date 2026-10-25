"""Pemuat dataset. Membaca folder datasets/<nama>/ secara fleksibel.

Format yang didukung: .csv, .tsv, .json, .jsonl, .txt
Kolom judul & isi dideteksi otomatis (lihat TITLE_COLS / BODY_COLS).
"""

import json
import os
from typing import Dict, List, Optional

import pandas as pd

DATASETS = ["etd-usk", "etd-ugm", "kompas", "tempo", "mojok"]

TITLE_COLS = ["title", "judul", "headline", "judul_artikel"]
BODY_COLS = [
    "content", "isi", "text", "body", "article", "konten", "artikel",
    "abstract", "abstrak", "abstract_id", "abstract_en", "description",
    "deskripsi", "ringkasan", "summary", "isi_berita", "full_text",
]
SUPPORTED_EXT = (".csv", ".tsv", ".json", ".jsonl", ".txt")

def _read_text_file(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def _read_table(path: str, limit: Optional[int]) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(path, sep=sep, dtype=str, encoding=enc,
                                   nrows=limit, on_bad_lines="skip")
            except UnicodeDecodeError:
                continue
        return pd.DataFrame()
    if ext == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False, nrows=limit)
    # .json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        data = lists[0] if lists else [data]
    df = pd.DataFrame(data)
    return df.head(limit) if limit else df


def _rows_to_records(df: pd.DataFrame, dataset: str, source: str) -> List[Dict]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    title_col = next((cols[c] for c in TITLE_COLS if c in cols), None)
    body_cols = [cols[c] for c in BODY_COLS if c in cols]
    if not body_cols:  # fallback: semua kolom teks selain judul
        body_cols = [c for c in df.columns
                     if c != title_col and df[c].dtype == object]
    records = []
    for i, row in enumerate(df.to_dict("records")):
        title = str(row[title_col]).strip() if title_col and pd.notna(row[title_col]) else ""
        body = " ".join(str(row[c]) for c in body_cols if pd.notna(row[c]))
        if not (title or body.strip()):
            continue
        records.append({
            "dataset": dataset,
            "title": title or f"(tanpa judul) {source}#{i}",
            "text": body,
            "source": f"{source}#{i}",
        })
    return records


def _load_file(path: str, name: str, rel: str, limit: Optional[int]) -> List[Dict]:
    if path.lower().endswith(".txt"):
        text = _read_text_file(path)
        if not text.strip():
            return []
        return [{"dataset": name, "title": text.strip().splitlines()[0][:120],
                 "text": text, "source": rel}]
    try:
        df = _read_table(path, limit)
    except Exception as exc:  # file rusak jangan hentikan seluruh proses
        print(f"  [!] Gagal membaca {rel}: {exc}")
        return []
    print(f"      kolom {rel}: {list(df.columns)}")
    return _rows_to_records(df, name, rel)


def load_dataset(path: str, name: str, limit: Optional[int] = None) -> List[Dict]:
    """Muat satu dataset (satu file ATAU satu folder berisi banyak file)."""
    if os.path.isfile(path):
        return _load_file(path, name, os.path.basename(path), limit)
    records: List[Dict] = []
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            if not fn.lower().endswith(SUPPORTED_EXT):
                continue
            if limit and len(records) >= limit:
                return records[:limit]
            full = os.path.join(root, fn)
            records.extend(_load_file(full, name, os.path.relpath(full, path), limit))
    return records[:limit] if limit else records


def find_dataset_path(datasets_dir: str, name: str) -> Optional[str]:
    """Cari folder/file dataset; toleran terhadap '-' vs '_' dan ekstensi."""
    variants = {name, name.replace("-", "_"), name.replace("_", "-")}
    for v in variants:
        folder = os.path.join(datasets_dir, v)
        if os.path.isdir(folder):
            return folder
        for ext in SUPPORTED_EXT:
            f = os.path.join(datasets_dir, v + ext)
            if os.path.isfile(f):
                return f
    return None


def load_all(datasets_dir: str, limit_per_dataset: Optional[int] = None,
             log=print) -> List[Dict]:
    """Muat kelima dataset dari folder datasets/."""
    all_records: List[Dict] = []
    for name in DATASETS:
        path = find_dataset_path(datasets_dir, name)
        if path is None:
            log(f"  [!] Dataset '{name}' tidak ditemukan di {datasets_dir} (dilewati)")
            continue
        recs = load_dataset(path, name, limit_per_dataset)
        log(f"  - {name:<9}: {len(recs):>6} dokumen")
        all_records.extend(recs)
    return all_records