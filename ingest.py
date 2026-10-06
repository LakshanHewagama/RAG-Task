"""Read every PDF and Excel file in data/, chunk it, embed it and store it in Chroma."""
import os
import re

import chromadb
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, EMBED_MODEL

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
ROWS_PER_CHUNK = 10


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into ~size character chunks with overlap, never cutting a word."""
    # keep line breaks as their own tokens so table rows stay on separate lines
    words = text.replace("\n", " \n ").split(" ")
    words = [w for w in words if w]
    chunks, current = [], []
    length = 0
    for word in words:
        if length + len(word) + 1 > size and current:
            chunks.append(" ".join(current))
            # keep the last words (about `overlap` characters) for the next chunk
            tail, tail_len = [], 0
            for w in reversed(current):
                if tail_len + len(w) + 1 > overlap:
                    break
                tail.insert(0, w)
                tail_len += len(w) + 1
            current, length = tail, tail_len
        current.append(word)
        length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return [c.replace(" \n ", "\n").replace("\n ", "\n").strip() for c in chunks]


def clean_layout_text(text):
    """Layout-mode text keeps table columns apart with many spaces; turn those gaps into ' | '."""
    lines = []
    for line in text.splitlines():
        line = re.sub(r" {3,}", " | ", line.strip())
        if line:
            lines.append(line)
    return "\n".join(lines)


def read_pdf(path, name):
    docs = []
    reader = PdfReader(path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = clean_layout_text(page.extract_text(extraction_mode="layout") or "")
        if not text:
            continue
        for i, chunk in enumerate(chunk_text(text)):
            docs.append({
                "id": f"{name}-p{page_num}-c{i}",
                "text": chunk,
                "meta": {"source": name, "page": page_num, "type": "pdf"},
            })
    return docs


def read_excel(path, name):
    docs = []
    sheets = pd.read_excel(path, sheet_name=None)
    for sheet, df in sheets.items():
        df = df.dropna(how="all")
        rows = []
        for idx, row in df.iterrows():
            parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip()]
            if parts:
                # +2 = Excel row number (1-based and header row)
                rows.append((idx + 2, ", ".join(parts)))
        for start in range(0, len(rows), ROWS_PER_CHUNK):
            group = rows[start:start + ROWS_PER_CHUNK]
            row_start, row_end = group[0][0], group[-1][0]
            text = f"Sheet: {sheet}\n" + "\n".join(f"Row {r}: {t}" for r, t in group)
            docs.append({
                "id": f"{name}-{sheet}-r{row_start}-{row_end}",
                "text": text,
                "meta": {"source": name, "sheet": str(sheet), "row_start": int(row_start),
                         "row_end": int(row_end), "type": "excel"},
            })
    return docs


def load_documents():
    docs, counts = [], {}
    for name in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, name)
        ext = os.path.splitext(name)[1].lower()
        try:
            if ext == ".pdf":
                file_docs = read_pdf(path, name)
            elif ext in (".xlsx", ".xls"):
                file_docs = read_excel(path, name)
            else:
                continue
        except Exception as e:
            print(f"  ! Could not read {name}: {e}")
            continue
        counts[name] = len(file_docs)
        docs.extend(file_docs)
    return docs, counts


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"Folder '{DATA_DIR}/' not found. Create it and add your files.")
        return

    docs, counts = load_documents()
    if not docs:
        print(f"No PDF or Excel content found in '{DATA_DIR}/'.")
        return

    print(f"Embedding {len(docs)} chunks with {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    batch = 1000
    for i in range(0, len(docs), batch):
        part = docs[i:i + batch]
        collection.add(
            ids=[d["id"] for d in part],
            documents=[d["text"] for d in part],
            metadatas=[d["meta"] for d in part],
            embeddings=embeddings[i:i + batch],
        )

    print("\nSummary")
    print(f"Files processed: {len(counts)}")
    for name, n in counts.items():
        print(f"  {name}: {n} chunks")
    print(f"Total chunks: {len(docs)}")


if __name__ == "__main__":
    main()
