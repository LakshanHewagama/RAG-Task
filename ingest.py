"""Read every PDF and Excel file in data/, chunk it, embed it and store it in Chroma."""
import os
import re

import chromadb
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, EMBED_MODEL, MAX_EXCEL, MAX_PDFS

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
ROWS_PER_CHUNK = 10
# Pages up to this size stay as one chunk so tables are never cut in half
MAX_PAGE_CHUNK = 2000


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


ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}


def normalize_numbers(text):
    """'Semester V' / 'Semester 05' -> 'Semester 5' so they match questions like 'semester 5'."""
    text = re.sub(r"\b(Semester|Year)\s+(VIII|VII|VI|IV|V|III|II|I)\b",
                  lambda m: f"{m.group(1)} {ROMAN[m.group(2)]}", text)
    return re.sub(r"\b(Semester|Year)\s+0(\d)\b", r"\1 \2", text)


def read_pdf(path, name):
    docs = []
    reader = PdfReader(path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = normalize_numbers(clean_layout_text(page.extract_text(extraction_mode="layout") or ""))
        if not text:
            continue
        # The page title (first line) is repeated in every chunk so table rows keep their context
        title = text.splitlines()[0][:150]
        page_chunks = [text] if len(text) <= MAX_PAGE_CHUNK else chunk_text(text)
        for i, chunk in enumerate(page_chunks):
            if i > 0:
                chunk = f"{title}\n{chunk}"
            docs.append({
                "id": f"{name}-p{page_num}-c{i}",
                "text": chunk,
                "meta": {"source": name, "page": page_num, "type": "pdf"},
            })
    return docs


def read_excel(path, name):
    docs = []
    # header=None: read every row as data, then find the real header row ourselves
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    for sheet, df in sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        cells = [[str(v).strip() if pd.notna(v) and str(v).strip() else None for v in r]
                 for r in df.itertuples(index=False)]
        excel_rows = [i + 1 for i in df.index]  # pandas index 0 = Excel row 1

        # Header = first row where at least half the columns are filled with text (not numbers).
        # Rows above it (e.g. a "Software Evolution" title) become the sheet title.
        ncols = len(df.columns)
        h = next((i for i, r in enumerate(cells)
                  if sum(1 for v in r if v and not v.replace(".", "").isdigit()) >= max(2, ncols / 2)), 0)
        headers = [v or f"Column {j + 1}" for j, v in enumerate(cells[h])]
        title = " ".join(v for r in cells[:h] for v in r if v)

        rows = []
        for r, excel_row in zip(cells[h + 1:], excel_rows[h + 1:]):
            parts = [f"{col}: {val}" for col, val in zip(headers, r) if val]
            if parts:
                rows.append((excel_row, ", ".join(parts)))
        heading = f"Sheet: {sheet}" + (f" ({title})" if title else "")
        for start in range(0, len(rows), ROWS_PER_CHUNK):
            group = rows[start:start + ROWS_PER_CHUNK]
            row_start, row_end = group[0][0], group[-1][0]
            text = heading + "\n" + "\n".join(f"Row {r}: {t}" for r, t in group)
            docs.append({
                "id": f"{name}-{sheet}-r{row_start}-{row_end}",
                "text": text,
                "meta": {"source": name, "sheet": str(sheet), "row_start": int(row_start),
                         "row_end": int(row_end), "type": "excel"},
            })
    return docs


def file_kind(name):
    """'pdf', 'excel' or None for a file name."""
    ext = os.path.splitext(name)[1].lower()
    return "pdf" if ext == ".pdf" else "excel" if ext in (".xlsx", ".xls") else None


def limit_error(names):
    """Message if names has more PDFs or Excel files than allowed, else None."""
    pdfs = sum(1 for n in names if file_kind(n) == "pdf")
    excels = sum(1 for n in names if file_kind(n) == "excel")
    if pdfs > MAX_PDFS or excels > MAX_EXCEL:
        return (f"Only {MAX_PDFS} PDFs and {MAX_EXCEL} Excel files are allowed "
                f"(found {pdfs} PDFs and {excels} Excel files).")
    return None


def load_documents():
    docs, counts = [], {}
    names = sorted(os.listdir(DATA_DIR))
    if error := limit_error(names):
        raise ValueError(error)
    for name in names:
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

    try:
        docs, counts = load_documents()
    except ValueError as e:
        print(e)
        return
    if not docs:
        print(f"No PDF or Excel content found in '{DATA_DIR}/'.")
        # Drop the old collection so deleted files are no longer answered from
        try:
            chromadb.PersistentClient(path=CHROMA_DIR).delete_collection(COLLECTION_NAME)
        except Exception:
            pass
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
