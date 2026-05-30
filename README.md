# NoLimit RAG Chatbot

RAG chatbot for the NoLimit Indonesia data scientist test. The app loads PDF documents, extracts and chunks the text, embeds the chunks with `sentence-transformers/all-MiniLM-L6-v2`, retrieves the most relevant passages with cosine similarity or FAISS, and answers with `GeminiClient` through the `google-genai` SDK.

Live demo: https://nolimit-ragchatbot.streamlit.app/

## Project Structure

- `src/app.py` - Streamlit entry point that handles upload, indexing, retrieval, and chat responses
- `src/rag/pdf_loader.py` - PDF loading, text normalization, and chunking
- `src/rag/embedder.py` - SentenceTransformer wrapper for embedding generation
- `src/rag/vector_store.py` - Vector search with FAISS support and cosine fallback
- `src/rag/llm_client.py` - Gemini prompt builder and generation client
- `src/rag/__init__.py` - Re-exports the RAG utilities used by the app
- `data/sample/CV_Leonard Arif Sutiono.pdf` - Local sample PDF for verification
- `Flowchart.png` - End-to-end pipeline diagram

## Dataset Source

Source: the included sample document is the author's CV included for local verification and demonstration. File path: `data/sample/CV_Leonard Arif Sutiono.pdf`.

License: author-owned sample content used only for this technical test submission.

## Models Used

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Generative model: `gemini-2.5-flash` by default, configurable with `GEMINI_MODEL_ID`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file and set `GEMINI_API_KEY`.
4. Optionally set `GEMINI_MODEL_ID` or `EMBEDDING_MODEL` to override the defaults.
5. Start the app:

```bash
python -m streamlit run src/app.py
```

## How It Works

1. Upload one or more PDF files, or use the bundled sample PDF.
2. The app extracts text per page and splits it into overlapping chunks.
3. Chunks are embedded and indexed.
4. A query retrieves the top relevant chunks.
5. The Gemini model answers using only the retrieved context and cites file/page sources.
6. If the API key is missing or generation fails, the app falls back to a retrieval-only summary.

## Usage (quick)

- Live demo: https://nolimit-ragchatbot.streamlit.app/
- Run locally from the project root:

```bash
python -m streamlit run src/app.py
```

- On the app page: upload one or more PDF files (or use the included sample CV), enter a question in the input box, and press Send.
- The assistant will answer using only the retrieved document context and will list cited sources (file name + page number) under the assistant message.
- If the answer is not present in the documents, the app returns the exact fallback message:

`Informasi ini tidak ditemukan dalam dokumen yang tersedia.`

## Output Rules

The assistant is instructed to answer only from the document context. If the answer is not found, it returns:

`Informasi ini tidak ditemukan dalam dokumen yang tersedia.`

When generation fails, the app shows a retrieval-only answer built from the top retrieved chunks.

## Flowchart

See [Flowchart.png](Flowchart.png) for the end-to-end pipeline diagram.

## Notes

- On Windows, the app uses cosine similarity if FAISS is unavailable.
- On Linux or Hugging Face Spaces, `faiss-cpu` can be installed and used automatically.