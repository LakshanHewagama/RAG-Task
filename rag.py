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
    "prefer values that are explicitly labelled (e.g. 'Credit Value: 02'). "
    "When the question asks who/which/how many/list, check EVERY context block and EVERY row, "
    "and include ALL matches (one per line, each with its citation), not just the first one found."
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


STOPWORDS = {"the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "is", "are", "was", "what",
              "which", "who", "how", "many", "much", "do", "does", "tell", "me", "about", "list", "give",
              "all", "with", "this", "that", "there", "subjects", "subject", "courses", "course", "modules"}


def keywords(text):
    """Lowercase words and numbers (minus stopwords), plus word pairs like 'semester 5'."""
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
    return set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])}


def keyword_scores(question):
    """Score every chunk by how many query keywords/pairs appear in it (pairs count double)."""
    terms = keywords(question)
    data = get_collection().get(include=["documents", "metadatas"])
    scored = []
    for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        text = " ".join(re.findall(r"[a-z0-9]+", doc.lower()))
        # IDs/codes (letters + digits, e.g. BSSE0122011287, SE4107) count 5, word pairs 2, single words 1
        score = sum((5 if re.search(r"[a-z]", t) and re.search(r"\d", t) and " " not in t
                     else 2 if " " in t else 1)
                    for t in terms if re.search(rf"\b{t}\b", text))
        if score:
            scored.append((score, cid, doc, meta))
    scored.sort(key=lambda x: -x[0])
    return scored


def cited_sources(answer, chunks):
    """Return [(number, chunk)] for the chunks the answer cites as [n]; all chunks if none are cited."""
    nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(chunks)})
    if not nums:
        return list(enumerate(chunks, start=1))
    return [(n, chunks[n - 1]) for n in nums]


def fix_citations(answer, chunks):
    """The model sometimes cites the wrong block: check each answer line's word pairs (e.g. a name)
    against the chunks and point its [n] at the chunk that really contains them."""
    flat = {n: " ".join(re.findall(r"[a-z0-9]+", c["text"].lower())) for n, c in enumerate(chunks, start=1)}
    out = []
    for line in answer.splitlines():
        cites = [int(n) for n in re.findall(r"\[(\d+)\]", line) if int(n) in flat]
        pairs = {p for p in keywords(re.sub(r"\[\d+\]", " ", line)) if " " in p}
        has = lambda n: any(re.search(rf"\b{p}\b", flat[n]) for p in pairs)
        if cites and pairs and not any(has(n) for n in cites):
            right = [n for n in flat if has(n)]
            if right:
                tail = line[len(line.rstrip()):]  # keep markdown line-break spaces
                line = re.sub(r"(\s*\[\d+\])+", "", line).rstrip() + " " + "".join(f"[{n}]" for n in right) + tail
        out.append(line)
    return "\n".join(out)


def snippet(text, question, answer="", limit=1500):
    """Lines of the chunk that contain words from the question/answer (e.g. the matching table row)."""
    def terms(t):
        return {w for w in keywords(t) if " " not in w and (len(w) > 3 or w.isdigit())}
    # Drop citations like [2] so their numbers don't match unrelated rows
    q_terms, a_terms = terms(question), terms(re.sub(r"\[\d+\]", " ", answer))
    lines = text.splitlines()
    # Answer words not in the question (e.g. a person's name) pin down the exact row the answer used.
    # Full word pairs ('amaya amarasinghe') come first so a shared surname doesn't pull in other rows.
    answer_clean = re.sub(r"\[\d+\]", " ", answer)
    pairs = {p for p in keywords(answer_clean) if " " in p} - keywords(question)
    flat = [" ".join(re.findall(r"[a-z0-9]+", line.lower())) for line in lines]
    candidates = [i for i, f in enumerate(flat) if any(re.search(rf"\b{p}\b", f) for p in pairs)] \
        or [i for i, f in enumerate(flat) if any(t in f for t in a_terms - q_terms)] \
        or range(len(lines))
    # Words from the question count 3x, words from the answer 1x
    scored = [(3 * sum(t in lines[i].lower() for t in q_terms) + sum(t in lines[i].lower() for t in a_terms), i)
              for i in candidates]
    top = max((s for s, _ in scored), default=0)
    best = sorted(i for s, i in sorted(scored, reverse=True)[:6] if s > 0 and s >= top / 2)
    if not best:
        return text[:limit] + ("..." if len(text) > limit else "")
    # Keep whole lines so a table row is never cut off mid-way
    out = [lines[best[0]]]
    for i in best[1:]:
        if len("\n".join(out)) + len(lines[i]) > limit:
            break
        out.append(lines[i])
    return "\n".join(out)


def exact_label(meta, snippet_text):
    """Source label narrowed to the exact Excel rows shown in the snippet (e.g. 'row 6')."""
    rows = [int(n) for n in re.findall(r"^Row (\d+):", snippet_text, re.MULTILINE)]
    if meta.get("type") != "excel" or not rows:
        return source_label(meta)
    which = f"row {rows[0]}" if len(rows) == 1 else "rows " + ", ".join(map(str, rows))
    return f"{meta['source']}, sheet {meta['sheet']}, {which}"


def retrieve(question, k=TOP_K):
    """Hybrid search: vector results (rank-based score) + keyword results (match-count score)."""
    embedding = get_model().encode([question]).tolist()
    res = get_collection().query(query_embeddings=embedding, n_results=max(k * 4, 20))
    found = {}
    for rank, (cid, doc, meta, dist) in enumerate(zip(res["ids"][0], res["documents"][0],
                                                       res["metadatas"][0], res["distances"][0])):
        found[cid] = {"text": doc, "meta": meta, "distance": dist, "score": 1 / (60 + rank)}
    kw = keyword_scores(question)[:20]
    for score, cid, doc, meta in kw:
        item = found.setdefault(cid, {"text": doc, "meta": meta, "distance": 1.0, "score": 0})
        # Exact keyword matches (e.g. "semester 5", "SE3102") get up to 2x the weight of the top vector hit
        item["score"] += 2 * (score / kw[0][0]) / 60
    return sorted(found.values(), key=lambda c: -c["score"])[:k]


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
        best = retrieve(question)
        # The question's own results always come first; history only adds a few extra chunks
        # for follow-ups, so earlier unrelated questions can't push the right chunks out
        chunks = list(best)
        if history:
            seen = {(c["meta"]["source"], c["text"]) for c in best}
            extra = [c for c in retrieve(search_query) if (c["meta"]["source"], c["text"]) not in seen]
            chunks += extra[:3]
    except Exception as e:
        return f"Could not search the knowledge base. Did you run `python ingest.py`? ({e})", []

    # Nothing in the documents is close to the question: answer without calling the LLM
    id_match = any(score >= 5 for score, *_ in keyword_scores(question)[:1])
    if not best or (min(c["distance"] for c in best) > MAX_DISTANCE and not id_match):
        return NOT_FOUND, []

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, chunks)},
    ]

    # Try the main model, then each fallback, until one gives a real answer
    models = [OPENROUTER_MODEL] + [m for m in FALLBACK_MODELS if m != OPENROUTER_MODEL]
    problem = "No model could answer right now. Please try again in a minute."
    for model in models:
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
                answer = clean_answer(resp.choices[0].message.content)
                if answer:
                    return fix_citations(answer, chunks), chunks
                problem = "The free models returned no usable answer. Please try again."
                break  # empty or safety label: try the next model
            except RateLimitError:
                problem = "The free models are rate limited right now. Wait a bit and try again."
                if attempt == 0:
                    time.sleep(3)
                    continue
                break
            except APIConnectionError:
                return "Network error: could not reach OpenRouter. Check your connection.", chunks
            except APIStatusError as e:
                if e.status_code == 401:
                    return "OpenRouter rejected the API key (401). Check .env.", chunks
                # 404 (model removed), 5xx, etc.: try the next model
                problem = f"OpenRouter error {e.status_code} for model '{model}'. Check the model ids in .env."
                break
            except Exception as e:
                problem = f"Unexpected error: {e}"
                break
    return problem, chunks


def clean_answer(text):
    """Return the answer text, or '' if it is empty or a safety-classifier label instead of an answer."""
    answer = (text or "").strip()
    # Some models cite like 【1†L1-L4】; turn that into [1]
    answer = re.sub(r"【(\d+)[^】]*】", r"[\1]", answer)
    # Guard models reply e.g. "User Safety: safe" instead of answering
    if re.match(r"^(user|response|prompt)?\s*safety\s*:", answer, re.I) or answer.lower() in ("safe", "unsafe"):
        return ""
    return answer


if __name__ == "__main__":
    q = input("Question: ") or "What is this document about?"
    answer, sources = ask(q)
    print("\nAnswer:\n" + answer)
    print("\nSources:")
    for i, s in enumerate(sources, start=1):
        print(f"[{i}] {source_label(s['meta'])}")
