"""Streamlit chat UI for the knowledge base."""
import os

import streamlit as st

import ingest
import rag
from config import CHROMA_DIR, DATA_DIR, OPENROUTER_MODEL

st.set_page_config(page_title="Knowledge Base Chatbot")
st.title("Knowledge Base Chatbot")


@st.cache_resource
def load_model():
    return rag.get_model()


@st.cache_resource
def load_collection():
    return rag.get_collection()


def show_sources(sources, question="", answer=""):
    """Show only the chunks the answer cites, with the lines that match the question."""
    if not sources:
        return
    with st.expander("Sources"):
        for i, s in rag.cited_sources(answer, sources):
            text = rag.snippet(s["text"], question, answer)
            st.markdown(f"**[{i}] {rag.exact_label(s['meta'], text)}**")
            st.text(text)


# Sidebar
with st.sidebar:
    st.subheader("Settings")
    st.write(f"**Model:** {OPENROUTER_MODEL or 'not set'}")

    db_ready = os.path.isdir(CHROMA_DIR)
    collection = None
    if db_ready:
        try:
            collection = load_collection()
            rag._collection = collection
            st.write(f"**Chunks in DB:** {collection.count()}")
        except Exception:
            db_ready = False

    st.write("**Documents:**")
    files = [f for f in sorted(os.listdir(DATA_DIR)) if f.lower().endswith((".pdf", ".xlsx", ".xls"))] \
        if os.path.isdir(DATA_DIR) else []
    for f in files:
        name_col, del_col = st.columns([5, 1], vertical_alignment="center")
        name_col.caption(f"📄 {f}")
        if del_col.button("🗑️", key=f"delete_{f}", help=f"Delete {f}"):
            os.remove(os.path.join(DATA_DIR, f))
            with st.spinner("Re-ingesting..."):
                ingest.main()
            load_collection.clear()
            rag._collection = None
            st.session_state.upload_done = f"Deleted: {f}"
            st.rerun()
    if not files:
        st.caption("No files in data/")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    # Changing the uploader's key after a save gives an empty upload box
    st.session_state.setdefault("upload_key", 0)
    uploads = st.file_uploader("Upload files", type=["pdf", "xlsx", "xls"], accept_multiple_files=True,
                               key=f"uploader_{st.session_state.upload_key}")
    # Uploads replace same-named files, so count the combined set of names
    upload_error = ingest.limit_error(set(files) | {f.name for f in uploads}) if uploads else None
    if upload_error:
        st.error(upload_error + " Remove some files first.")
    if uploads and not upload_error and st.button("Save & re-ingest"):
        os.makedirs(DATA_DIR, exist_ok=True)
        for f in uploads:
            with open(os.path.join(DATA_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())
        with st.spinner("Ingesting..."):
            ingest.main()
        load_collection.clear()
        rag._collection = None
        st.session_state.upload_key += 1
        st.session_state.upload_done = f"Saved and ingested: {', '.join(f.name for f in uploads)}"
        st.rerun()
    if "upload_done" in st.session_state:
        st.success(st.session_state.pop("upload_done"))

if not db_ready:
    st.warning("Knowledge base not found. Run `python ingest.py` first.")
    st.stop()

rag._model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources"), msg.get("question", ""), msg["content"])
        # Deleting an answer also deletes its question, so it no longer affects later searches
        if msg["role"] == "assistant" and st.button("🗑️ Delete", key=f"del_msg_{idx}"):
            start = idx - 1 if idx > 0 and st.session_state.messages[idx - 1]["role"] == "user" else idx
            del st.session_state.messages[start:idx + 1]
            st.rerun()

if question := st.chat_input("Ask a question about the documents"):
    history = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = rag.ask(question, history)
        st.markdown(answer)
        show_sources(sources, question, answer)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "question": question})
    st.rerun()  # redraw from history so the new answer gets its delete button
