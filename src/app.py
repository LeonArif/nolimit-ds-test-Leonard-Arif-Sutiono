from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import streamlit as st
from dotenv import load_dotenv

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag import HFInferenceClient, RagIndex, SentenceEmbedder, load_pdf_bytes


load_dotenv(PROJECT_ROOT / ".env")

SAMPLE_PDF = PROJECT_ROOT / "data" / "sample" / "CV_Leonard Arif Sutiono.pdf"
DEFAULT_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct").strip()
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()


@dataclass(frozen=True)
class SourceFile:
    name: str
    content: bytes


def _source_file_from_uploaded(uploaded_file) -> SourceFile:
    return SourceFile(name=uploaded_file.name, content=uploaded_file.getvalue())


def _source_file_from_path(path: Path) -> SourceFile:
    return SourceFile(name=path.name, content=path.read_bytes())


def _compute_signature(files: Sequence[SourceFile]) -> str:
    digest = hashlib.sha256()
    for source_file in files:
        digest.update(source_file.name.encode("utf-8"))
        digest.update(source_file.content)
    return digest.hexdigest()


def _build_rag_index(files: Sequence[SourceFile], embedder: SentenceEmbedder) -> tuple[RagIndex, list]:
    chunks = []
    for source_file in files:
        chunks.extend(load_pdf_bytes(source_file.content, source_file.name))
    return RagIndex.build(chunks, embedder), chunks


def _load_active_sources() -> list[SourceFile]:
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    sources: list[SourceFile] = []
    if uploaded_files:
        sources.extend(_source_file_from_uploaded(file) for file in uploaded_files)

    if st.sidebar.checkbox("Include sample PDF", value=not bool(uploaded_files)) and SAMPLE_PDF.exists():
        sources.append(_source_file_from_path(SAMPLE_PDF))

    return sources


def _render_sources(retrieved_chunks) -> None:
    if not retrieved_chunks:
        st.info("No supporting sources were retrieved for this answer.")
        return

    with st.expander("Retrieved sources", expanded=True):
        for index, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(
                f"**Source {index}:** {chunk.file_name}  \n"
                f"**Page:** {chunk.page_number}  \n"
                f"**Score:** {chunk.score:.3f}"
            )
            st.write(chunk.text)
            st.divider()


def _build_fallback_answer(retrieved_chunks, error_message: str | None = None) -> str:
    if not retrieved_chunks:
        return "Informasi ini tidak ditemukan dalam dokumen yang tersedia."

    lines = []
    if error_message:
        lines.append(f"Mode retrieval-only aktif: {error_message}")
    else:
        lines.append("Model generatif tidak tersedia, jadi ini ringkasan hasil retrieval dari dokumen:")

    seen_sources: set[tuple[str, int]] = set()
    unique_chunks = []
    for chunk in retrieved_chunks:
        source_key = (chunk.file_name, chunk.page_number)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        unique_chunks.append(chunk)
        if len(unique_chunks) == 3:
            break

    for chunk in unique_chunks:
        snippet = " ".join(chunk.text.split())
        if len(snippet) > 320:
            snippet = snippet[:320].rsplit(" ", 1)[0] + "..."
        lines.append(f"- {chunk.file_name} halaman {chunk.page_number}: {snippet}")

    return "\n".join(lines)


def main() -> None:
    st.set_page_config(page_title="NoLimit RAG Chatbot", page_icon="📄", layout="wide")
    st.title("NoLimit RAG Chatbot")
    st.caption("PDF question answering with sentence-transformers, FAISS or cosine fallback, and Hugging Face generation.")

    with st.sidebar:
        st.header("Configuration")
        model_id = st.text_input("HF model id", value=DEFAULT_MODEL_ID)
        st.text_input("Embedding model", value=DEFAULT_EMBEDDING_MODEL, disabled=True)
        top_k = st.slider("Top-k retrieval", min_value=1, max_value=5, value=3)
        st.write("Provide your `HF_TOKEN` via `.env` or Streamlit secrets.")

    sources = _load_active_sources()
    if not sources:
        st.warning("Upload at least one PDF or enable the sample PDF to start indexing.")
        return

    token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")
    if not token:
        st.warning("HF_TOKEN is missing. Generative answers will fail until the token is configured.")
    embedding_model = DEFAULT_EMBEDDING_MODEL

    if "embedder" not in st.session_state or st.session_state.get("embedding_model_name") != embedding_model:
        with st.spinner("Loading embedding model..."):
            try:
                st.session_state.embedder = SentenceEmbedder(embedding_model)
            except Exception:
                if embedding_model != DEFAULT_EMBEDDING_MODEL:
                    st.warning(f"Embedding model '{embedding_model}' failed to load. Falling back to '{DEFAULT_EMBEDDING_MODEL}'.")
                    embedding_model = DEFAULT_EMBEDDING_MODEL
                    st.session_state.embedder = SentenceEmbedder(embedding_model)
                else:
                    raise
            st.session_state.embedding_model_name = embedding_model
            st.session_state.rag_index = None
            st.session_state.rag_chunks = []
            st.session_state.rag_signature = None
            st.session_state.chat_history = []

    current_signature = _compute_signature(sources)
    if st.session_state.get("rag_signature") != current_signature:
        with st.spinner("Building document index..."):
            rag_index, chunks = _build_rag_index(sources, st.session_state.embedder)
            st.session_state.rag_index = rag_index
            st.session_state.rag_chunks = chunks
            st.session_state.rag_signature = current_signature
            st.session_state.chat_history = []

    st.success(f"Indexed {len(st.session_state.rag_chunks)} chunks from {len(sources)} PDF file(s).")

    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"])

    query = st.chat_input("Ask a question about the uploaded PDFs")
    if not query:
        return

    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            retrieved_chunks = st.session_state.rag_index.search(query, st.session_state.embedder, top_k=top_k)
            if token:
                client = HFInferenceClient(model_id=model_id, token=token)
                try:
                    answer = client.generate_answer(query, retrieved_chunks)
                except Exception as error:
                    answer = _build_fallback_answer(retrieved_chunks, str(error))
            else:
                answer = _build_fallback_answer(retrieved_chunks, "HF_TOKEN is missing")

        st.markdown(answer)
        _render_sources(retrieved_chunks)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved_chunks,
        }
    )


if __name__ == "__main__":
    main()