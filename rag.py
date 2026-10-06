"""Retrieve relevant chunks from Chroma and ask the LLM on OpenRouter."""
import re
import time

import chromadb
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError
from sentence_transformers import SentenceTransformer

from config import (CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, FALLBACK_MODELS, MAX_DISTANCE,
                    OPENROUTER_API_KEY, OPENROUTER_MODEL, TOP_K)

NOT_FOUND = "I could not find this in the documents."

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer ONLY using the provided context. "
    "If the answer is not in the context, say you could not find it in the documents. "
    "Cite sources like [1], [2]. Tables in the context use ' | ' between columns: "
    "match each value to its column header carefully and copy numbers exactly. "
    "If a table row has no column headers in the context, do not guess what its numbers mean; "
    "prefer values that are explicitly labelled (e.g. 'Credit Value: 02')."
)

_model = None
_collection = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def source_label(meta):
    if meta.get("type") == "excel":
        return f"{meta['source']}, sheet {meta['sheet']}, rows {meta['row_start']}-{meta['row_end']}"
    return f"{meta['source']}, page {meta['page']}"


def retrieve(question, k=TOP_K):
    embedding = get_model().encode([question]).tolist()
    res = get_collection().query(query_embeddings=embedding, n_results=k)
    return [
        {"text": doc, "meta": meta, "distance": dist}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def build_prompt(question, chunks):
    blocks = [f"[{i}] {source_label(c['meta'])}\n{c['text']}" for i, c in enumerate(chunks, start=1)]
    return "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def ask(question, history=None):
    """Return (answer, sources). `history` is a list of previous user questions (optional)."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.startswith("sk-or-xxx"):
        return "OpenRouter API key is missing. Add it to the .env file.", []
    if not OPENROUTER_MODEL:
        return "OPENROUTER_MODEL is missing. Add a free model id to the .env file.", []

    # Conversation memory: add the last 3 questions to the retrieval query
    search_query = " ".join((history or [])[-3:] + [question])
    try:
        # Relevance check uses the question alone, so earlier questions can't make an unrelated one pass
        best = retrieve(question, 1)
        chunks = retrieve(search_query)
    except Exception as e:
        return f"Could not search the knowledge base. Did you run `python ingest.py`? ({e})", []

    # Nothing in the documents is close to the question: answer without calling the LLM
    if not best or best[0]["distance"] > MAX_DISTANCE:
        return NOT_FOUND, []

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, chunks)},
    ]

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=OPENROUTER_MODEL, messages=messages, temperature=0.2,
                # OpenRouter tries these in order if the main model fails
                extra_body={"models": FALLBACK_MODELS} if FALLBACK_MODELS else None,
            )
            answer = (resp.choices[0].message.content or "").strip()
            # Some models cite like 【1†L1-L4】; turn that into [1]
            answer = re.sub(r"【(\d+)[^】]*】", r"[\1]", answer)
            return answer or "The model returned an empty answer. Please try again.", chunks
        except RateLimitError:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            return "The free model is rate limited right now. Wait a bit or switch model.", chunks
        except APIConnectionError:
            return "Network error: could not reach OpenRouter. Check your connection.", chunks
        except APIStatusError as e:
            if e.status_code == 401:
                return "OpenRouter rejected the API key (401). Check .env.", chunks
            if e.status_code == 404:
                return (f"Model '{OPENROUTER_MODEL}' not available (404). Set OPENROUTER_MODEL=openrouter/free "
                        "in .env and restart the app."), chunks
            return f"OpenRouter error {e.status_code}: {e.message}", chunks
        except Exception as e:
            return f"Unexpected error: {e}", chunks


if __name__ == "__main__":
    q = input("Question: ") or "What is this document about?"
    answer, sources = ask(q)
    print("\nAnswer:\n" + answer)
    print("\nSources:")
    for i, s in enumerate(sources, start=1):
        print(f"[{i}] {source_label(s['meta'])}")
