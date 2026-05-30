from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import json
import os
import re
import socket
import urllib.error
import urllib.request

from .vector_store import SearchResult


SYSTEM_PROMPT = (
    "You are a precise document assistant. Answer ONLY using the provided context from the uploaded PDF documents. "
    "If the answer cannot be found in the context, respond exactly with: "
    '"Informasi ini tidak ditemukan dalam dokumen yang tersedia." ' 
    "Do not use external knowledge. Always cite the file name and page number in the answer."
)


def build_prompt(query: str, retrieved_chunks: Sequence[SearchResult]) -> str:
    context_parts: list[str] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {index}: {chunk.file_name}, Page {chunk.page_number}]\n{chunk.text}"
        )

    context = "\n\n".join(context_parts) if context_parts else "(no context available)"
    return (
        f"System:\n{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer in a concise, factual way. Include source citations using file name and page number."
    )


def _clean_generated_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^<\|assistant\|>\s*", "", cleaned)
    cleaned = re.sub(r"^Assistant:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


@dataclass
class HFInferenceClient:
    model_id: str
    token: str
    timeout: int = 120

    def generate_answer(self, query: str, retrieved_chunks: Sequence[SearchResult]) -> str:
        if not self.token:
            raise RuntimeError("HF_TOKEN is missing. Set it in .env or Streamlit secrets.")

        if not retrieved_chunks:
            return 'Informasi ini tidak ditemukan dalam dokumen yang tersedia.'

        prompt = build_prompt(query, retrieved_chunks)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.2,
                "top_p": 0.95,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        response = self._post(payload)
        generated_text = self._extract_generated_text(response)

        if not generated_text:
            return 'Informasi ini tidak ditemukan dalam dokumen yang tersedia.'

        return _clean_generated_text(generated_text)

    def _post(self, payload: dict) -> object:
        url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Hugging Face Inference API error ({error.code}): {body}") from error
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", None)
            if isinstance(reason, socket.gaierror):
                raise RuntimeError(
                    "Could not resolve the Hugging Face API host. This is a local DNS/network problem, "
                    "not a model-id problem. Check internet access, proxy, VPN, or firewall settings."
                ) from error

            raise RuntimeError(
                "Could not reach Hugging Face Inference API. Check internet access, proxy, VPN, or firewall settings."
            ) from error

    @staticmethod
    def _extract_generated_text(response: object) -> str:
        if isinstance(response, list) and response:
            first_item = response[0]
            if isinstance(first_item, dict):
                return str(first_item.get("generated_text", ""))

        if isinstance(response, dict):
            if "error" in response:
                raise RuntimeError(str(response["error"]))
            if "generated_text" in response:
                return str(response["generated_text"])
            if "summary_text" in response:
                return str(response["summary_text"])

        return ""