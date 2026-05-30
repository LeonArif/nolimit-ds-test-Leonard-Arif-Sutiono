# NoLimit RAG Chatbot

RAG chatbot for the NoLimit Indonesia data scientist test. The app ingests PDF documents, chunks the text, embeds the chunks with `sentence-transformers/all-MiniLM-L6-v2`, retrieves the most relevant passages with cosine similarity or FAISS, and sends the retrieved context to `Qwen/Qwen2.5-7B-Instruct` through the Hugging Face Inference API.

## Project Structure

- `src/app.py` - Streamlit entry point
- `src/rag/pdf_loader.py` - PDF loading and chunking
- `src/rag/embedder.py` - SentenceTransformer wrapper
- `src/rag/vector_store.py` - Vector search with FAISS fallback
- `src/rag/llm_client.py` - Hugging Face Inference API client
- `data/sample/CV_Leonard Arif Sutiono.pdf` - Local sample PDF for verification
- `flowchart.png` - End-to-end pipeline diagram

## Dataset Source

The included sample document is a personal CV PDF provided in the repository for local verification. No external public dataset is required for the test submission.

## Models Used

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Generative model: `Qwen/Qwen2.5-7B-Instruct`

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set `HF_TOKEN`.
4. Start the app:

```bash
streamlit run src/app.py
```

## How It Works

1. Upload one or more PDF files, or use the sample PDF.
2. The app extracts text per page and splits it into overlapping chunks.
3. Chunks are embedded and indexed.
4. A query retrieves the top relevant chunks.
5. The Hugging Face model answers using only the retrieved context and cites file/page sources.

## Output Rules

The assistant is instructed to answer only from the document context. If the answer is not found, it returns:

`Informasi ini tidak ditemukan dalam dokumen yang tersedia.`

## Flowchart

See [flowchart.png](flowchart.png).

## Notes

- On Windows, the app uses cosine similarity if FAISS is unavailable.
- On Linux or Hugging Face Spaces, `faiss-cpu` can be installed and used automatically.