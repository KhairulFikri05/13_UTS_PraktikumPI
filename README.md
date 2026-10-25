# Multi-Dataset Information Retrieval System (CLI)

UTS Praktikum Penelusuran Informasi Departemen Informatika, FMIPA, Universitas Syiah Kuala.

## Fitur
- Preprocessing: case folding, tokenisasi, stopword removal (Indonesia + Inggris, daftar buatan sendiri).
- Representasi dokumen: Bag of Words (`CountVectorizer`).
- Indexing & kandidat pencarian: `Whoosh` (inverted index).
- Ranking akhir: `cosine_similarity` antara vektor BoW query dan dokumen.
- Menampilkan 5 dokumen teratas beserta skor, dataset, judul, sumber, dan cuplikan.

## Struktur Proyek
```
main.py            # CLI interaktif
ir/preprocessing.py  # case folding, tokenisasi, stopword removal
ir/stopwords.py      # daftar stopword
ir/loader.py         # pembaca dataset (csv/tsv/json/jsonl/txt)
ir/indexer.py        # BoW + index Whoosh
ir/searcher.py       # pencarian + ranking cosine
datasets/            # etd-usk, etd-ugm, kompas, tempo, mojok
index/               # dibuat otomatis oleh menu [1]
```

## Instalasi & Menjalankan
```
pip install -r requirements.txt
python main.py
```
1. Pilih `[1] Load & Index Dataset` (Enter untuk default folder `datasets`).
   Opsional: isi batas dokumen per dataset untuk uji cepat.
2. Pilih `[2] Search Query`, ketik query, (opsional) filter dataset.
3. `[3] Exit` untuk keluar.

## Alur Sistem
1. **Load**, kelima dataset dibaca; kolom judul & isi dideteksi otomatis.
2. **Preprocess**, judul + isi diubah ke huruf kecil, ditokenisasi, stopword dibuang.
3. **BoW**,  `CountVectorizer` membentuk matriks dokumen x term (disimpan di `index/`).
4. **Index**, teks terpreproses diindeks di Whoosh.
5. **Search**, query dipreproses dengan pipeline yang sama; Whoosh mengambil
   dokumen kandidat (OR antar term, maks. 2000); kandidat diurutkan berdasarkan
   cosine similarity terhadap vektor BoW query; 5 teratas ditampilkan.

## Catatan
- Hanya memakai library standar Python + scikit-learn, Whoosh, pandas (scipy ikut terpasang bersama scikit-learn).

## Anggota Tim
1. Rijauddin Abdul Ghani (2408107010008)
2. Fawwaz Ziyadi Ilmi (2408107010021)
3. Khairul Fikri (2408107010032)
