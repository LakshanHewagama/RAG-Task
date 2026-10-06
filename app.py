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


def show_sources(sources):
    if not sources:
        return
    with st.expander("Sources"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {rag.source_label(s['meta'])}**")
            snippet = s["text"][:400] + ("..." if len(s["text"]) > 400 else "")
            st.caption(snippet)


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
        st.caption(f"📄 {f}")
    if not files:
        st.caption("No files in data/")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    uploads = st.file_uploader("Upload files", type=["pdf", "xlsx", "xls"], accept_multiple_files=True)
    if uploads and st.button("Save & re-ingest"):
        os.makedirs(DATA_DIR, exist_ok=True)
        for f in uploads:
            with open(os.path.join(DATA_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())
        with st.spinner("Ingesting..."):
            ingest.main()
        load_collection.clear()
        rag._collection = None
        st.success("Knowledge base updated.")
        st.rerun()

if not db_ready:
    st.warning("Knowledge base not found. Run `python ingest.py` first.")
    st.stop()

rag._model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources"))

if question := st.chat_input("Ask a question about the documents"):
    history = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = rag.ask(question, history)
        st.markdown(answer)
        show_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
