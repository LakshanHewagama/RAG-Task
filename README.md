# Knowledge Base Chatbot (RAG)

Answers questions using only the PDFs and Excel files in `data/`, and shows the source (file + page, or sheet + rows) for every answer.

## Setup
1. Create and activate a virtual environment
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
2. Install: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY` and a free `OPENROUTER_MODEL` (id ending in `:free`, from https://openrouter.ai/models).
4. Put the 5 PDFs and 3 Excel files into `data/`.
5. Build the knowledge base: `python ingest.py`
6. Start the app: `streamlit run app.py`

Quick terminal test: `python rag.py`

## Architecture
1. `ingest.py` reads PDFs page by page (pypdf) and Excel rows (pandas), 10 rows per chunk.
2. PDF pages up to 2000 characters stay whole (tables are not cut); longer pages are split into ~800 character chunks with 150 overlap. "Semester V" / "Semester 05" are normalised to "Semester 5".
3. Chunks are embedded locally with `all-MiniLM-L6-v2` and stored in ChromaDB (`chroma_db/`).
4. `rag.py` uses hybrid search (vector + keyword match, e.g. "semester 5", "SE3102") to get the top-k chunks and sends them to an OpenRouter free model.
5. `app.py` is a Streamlit chat that shows the answer and a "Sources" expander.

Questions whose best match is farther than `MAX_DISTANCE` (default 0.75) get "I could not find this in the documents." without calling the LLM.
`FALLBACK_MODELS` lists models OpenRouter tries if the main one fails.

## Known limitations
- Free OpenRouter models have rate limits (429) and their ids change over time.
- Scanned PDFs (images only) have no text; OCR is not included.
- Very large Excel sheets create many chunks; numeric questions over many rows (sums, averages) may be incomplete because only top-k chunks are sent.
