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

from rag import GeminiClient, RagIndex, SentenceEmbedder, load_pdf_bytes


load_dotenv(PROJECT_ROOT / ".env")

SAMPLE_PDF = PROJECT_ROOT / "data" / "sample" / "CV_Leonard Arif Sutiono.pdf"
DEFAULT_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash").strip()
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
TOP_K_RETRIEVAL = 3


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
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    sources: list[SourceFile] = []
    if uploaded_files:
        sources.extend(_source_file_from_uploaded(file) for file in uploaded_files)

    if SAMPLE_PDF.exists():
        sources.append(_source_file_from_path(SAMPLE_PDF))

    return sources


def _render_sources(retrieved_chunks) -> None:
    if not retrieved_chunks:
        st.info("No supporting sources were retrieved for this answer.")
        return

    with st.expander("Retrieved sources", expanded=False):
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
    st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")
    st.title("RAG Chatbot")
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #2563eb;
            --secondary-background-color: #1f2937;
        }

        .stApp {
            background: #0f1117;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        [data-testid="stFileUploaderDropzone"] {
            min-height: 0;
            padding: 0;
            background: transparent;
            border: 0;
        }

        [data-testid="stFileUploaderDropzone"] button {
            min-height: 3.25rem;
            background: #2563eb;
            border-color: #3b82f6;
            color: #ffffff;
        }

        [data-testid="stFileUploaderDropzone"] small {
            line-height: 3.25rem;
        }

        [data-testid="stFileUploaderDropzone"] button:hover {
            background: #1d4ed8;
            border-color: #60a5fa;
            color: #ffffff;
        }

        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button,
        [data-testid="stFileUploader"] button[aria-label*="Remove"],
        [data-testid="stFileUploader"] button[title*="Remove"] {
            min-width: 1.75rem !important;
            width: 1.75rem !important;
            min-height: 1.75rem !important;
            height: 1.75rem !important;
            padding: 0 !important;
            border-radius: 999px !important;
            background: #1f2937 !important;
            border: 1px solid #475569 !important;
            color: #e5e7eb !important;
        }

        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button:hover,
        [data-testid="stFileUploader"] button[aria-label*="Remove"]:hover,
        [data-testid="stFileUploader"] button[title*="Remove"]:hover {
            background: #2563eb !important;
            border-color: #3b82f6 !important;
            color: #ffffff !important;
        }

        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button svg,
        [data-testid="stFileUploader"] button[aria-label*="Remove"] svg,
        [data-testid="stFileUploader"] button[title*="Remove"] svg {
            width: 1rem !important;
            height: 1rem !important;
        }

        div[data-testid="stFormSubmitButton"] {
            position: absolute;
            width: 1px;
            height: 1px;
            overflow: hidden;
            clip: rect(0 0 0 0);
            white-space: nowrap;
        }

        div[data-testid="stForm"] {
            border: 0;
            padding: 0;
            background: transparent;
        }

        div[data-testid="stForm"] > div {
            width: 100%;
        }

        div[data-testid="stForm"] [data-testid="stTextInput"] {
            margin: 0;
        }

        div[data-baseweb="input"] {
            min-height: 3.25rem;
            border-color: #475569;
        }

        div[data-baseweb="input"] input {
            min-height: 3.25rem;
        }

        div[data-baseweb="input"]:focus-within {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 1px #3b82f6 !important;
        }

        div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {
            background-color: #2563eb;
        }

        div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
            background-color: #0ea5e9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"])

    input_area = st.container()
    with input_area:
        upload_col, input_col = st.columns([1, 3], vertical_alignment="top")
        with upload_col:
            sources = _load_active_sources()
        with input_col:
            with st.form("question_form", clear_on_submit=True):
                query = st.text_input(
                    "Ask a question about the uploaded PDFs",
                    placeholder="Ask a question about the uploaded PDFs",
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("Send", use_container_width=True)

    if not sources:
        st.warning("Upload at least one PDF or enable the sample PDF to start indexing.")
        return

    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.warning("GEMINI_API_KEY is missing. Generative answers will fail until the key is configured.")
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

    if not submitted or not query:
        return

    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.spinner("Retrieving context and generating an answer..."):
        retrieved_chunks = st.session_state.rag_index.search(
            query,
            st.session_state.embedder,
            top_k=TOP_K_RETRIEVAL,
        )
        if api_key:
            client = GeminiClient(model_id=DEFAULT_MODEL_ID, api_key=api_key)
            try:
                answer = client.generate_answer(query, retrieved_chunks)
            except Exception as error:
                answer = _build_fallback_answer(retrieved_chunks, str(error))
        else:
            answer = _build_fallback_answer(retrieved_chunks, "GEMINI_API_KEY is missing")

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved_chunks,
        }
    )
    st.rerun()


if __name__ == "__main__":
    main()
