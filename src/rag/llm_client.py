from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from google import genai
from google.genai import types

from .vector_store import SearchResult


SYSTEM_PROMPT = (
    "You are a precise document assistant. Answer ONLY using the provided context from the uploaded PDF documents. "
    "If the answer cannot be found in the context, respond exactly with: "
    '"Informasi ini tidak ditemukan dalam dokumen yang tersedia." ' 
    "Do not use external knowledge. If the user asks about anything unrelated to the provided context, "
    "respond exactly with: \"Informasi ini tidak ditemukan dalam dokumen yang tersedia.\" "
    "Always cite the file name and page number in the answer when you do answer from the context."
)


def build_prompt(query: str, retrieved_chunks: Sequence[SearchResult]) -> str:
    context_parts: list[str] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {index}: {chunk.file_name}, Page {chunk.page_number}]\n{chunk.text}"
        )

    context = "\n\n".join(context_parts) if context_parts else "(no context available)"
    return (
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer only from the context above. If the answer is not in the context, return the exact Indonesian fallback sentence. "
        "Keep the answer concise and factual, and include file name and page number citations when relevant."
    )


def _clean_generated_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^<\|assistant\|>\s*", "", cleaned)
    cleaned = re.sub(r"^Assistant:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


@dataclass
class GeminiClient:
    model_id: str
    api_key: str
    max_output_tokens: int = 512

    def generate_answer(self, query: str, retrieved_chunks: Sequence[SearchResult]) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Set it in .env or Streamlit secrets.")

        if not retrieved_chunks:
            return 'Informasi ini tidak ditemukan dalam dokumen yang tersedia.'

        prompt = build_prompt(query, retrieved_chunks)
        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=self.max_output_tokens,
                temperature=0.0,
                top_p=1.0,
            ),
        )
        generated_text = self._extract_generated_text(response)

        if not generated_text:
            return 'Informasi ini tidak ditemukan dalam dokumen yang tersedia.'

        return _clean_generated_text(generated_text)

    @staticmethod
    def _extract_generated_text(response: object) -> str:
        text = getattr(response, "text", "")
        if text:
            return str(text)

        candidates = getattr(response, "candidates", None)
        if isinstance(candidates, list) and candidates:
            first_candidate = candidates[0]
            content = getattr(first_candidate, "content", None)
            if content is not None:
                parts = getattr(content, "parts", None)
                if isinstance(parts, list):
                    texts = [str(getattr(part, "text", "")) for part in parts if getattr(part, "text", "")]
                    return "".join(texts)

        return ""